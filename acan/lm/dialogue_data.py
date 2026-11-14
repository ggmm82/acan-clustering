"""Italian and English data to teach a model to converse, prepared in the worker.

Every file comes from a pinned revision of a public dataset with a permissive licence and is
checked against its SHA-256 before use. Files are downloaded one at a time, read in batches and
deleted, so the disk holds one source file plus the growing corpus. The preparation is resumable:
it records the file and row reached, and a pause or a restart of the worker continues from there.

The corpus is JSONL with ``text`` and ``split``:

* conversations are written in the chat format of the SmolLM2 tokenizer
  (``<|im_start|>user ... <|im_end|>``), so the model learns where a turn ends;
* encyclopaedic text (Wikipedia, FineWeb-2) keeps the language and the facts;
* the official test files of the sources, and about 1% + 1% of everything else chosen by a hash
  of the text, are the validation and test splits: the fixed reference on which every checkpoint
  of a long training is measured.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

HF = 'https://huggingface.co/datasets/'


def _files(repo, revision, entries):
    return [dict(url=f'{HF}{repo}/resolve/{revision}/{path}', path=path, bytes=size, sha256=digest)
            for path, size, digest in entries]


SMOLTALK = _files('HuggingFaceTB/smol-smoltalk', 'f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc', [
    ('data/test-00000-of-00001.parquet', 48177022, 'be6773dcce145f3918ff14237b1f765affa427b0b13f6a02d397e665ac908b9a'),
    ('data/train-00000-of-00004.parquet', 229710350, '498cd4580014f42c40cfde066573564ed28e3e1548a5566715b615f5ea932856'),
    ('data/train-00001-of-00004.parquet', 230298184, '5412c03df579fc7dd7d06c7f5249627d99d0621882fb8f2cd5c13582eb2a80f6'),
    ('data/train-00002-of-00004.parquet', 230587255, 'aefcac6571abf275b4e1629baa01f326c0fe279d35cd0349b7439319923815a7'),
    ('data/train-00003-of-00004.parquet', 231879938, 'c15edfca888c9924a626dad7b7f4e9f08d8f406f997ac13be46bc9be0a6192db')])
AYA = _files('CohereLabs/aya_dataset', 'f9ea04583f02a8f86404ff6c58bf75fe637df8a2', [
    ('data/test-00000-of-00001.parquet', 977517, '52bbb79461bff9887c9987a5737d9a06f5ec6c78c08d770828c37fc182ce2cca'),
    ('data/train-00000-of-00001.parquet', 137195800, '51baa85043b569ba117f41516b32d1b0e4e2647fb203a08933f98a556fb1fb06')])
OASST = _files('OpenAssistant/oasst2', '179dd21fc55192153d94adb0e0ce8f69e222bf75', [
    ('data/train-00000-of-00001-88ba0162028a73fc.parquet', 63494018, '9638227b41b0a4fb69afa5d5c4b52e3fb6abfd56c9870d81bae32c9981e769b1'),
    ('data/validation-00000-of-00001-1deeef95c3248fe0.parquet', 3180111, 'fd46c63babc0756909a957ee40975bcf5610d58c5d1ab5ff2c57c0f57be1f273')])
REDIX = [dict(url=f'{HF}ReDiX/everyday-conversations-ita/resolve/5e60d2c503a87dc9589c80bab29bcb3badff9ae2/everyday-conversations.jsonl',
              path='everyday-conversations.jsonl', bytes=6653879,
              sha256='6bdc3190c605828c01df0184e9a914f338e6c976f40352c4ed2f1d6c5db162b7')]
FINEWEB2_IT = _files('HuggingFaceFW/fineweb-2', 'af9c13333eb981300149d5ca60a8e9d659b276b9', [
    ('data/ita_Latn/test/000_00000.parquet', 63821346, '2daa4433781837cb71c9d2caa23ac89c0f6c2228fe4366ad54df3002b7e40c42')])
_SOURCES_FILE = Path(__file__).with_name('dialogue_sources.json')   # long lists: SYNTH and Wikipedia shards
_LISTS = json.loads(_SOURCES_FILE.read_text())
SYNTH, WIKI_IT, WIKI_EN = _LISTS['synth'], _LISTS['wiki_it'], _LISTS['wiki_en']

SOURCES = {
    'smoltalk': dict(title='SmolTalk (versione per modelli piccoli) · conversazioni in inglese', languages=['en'],
                     license='Apache 2.0', page='https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk', files=SMOLTALK),
    'synth': dict(title='SYNTH (PleIAs) · domande e risposte sintetiche da Wikipedia, anche in italiano', languages=['it', 'en'],
                  license='CC BY 4.0', page='https://huggingface.co/datasets/PleIAs/SYNTH', files=SYNTH),
    'aya': dict(title='Aya dataset · istruzioni scritte da persone, parte italiana e inglese', languages=['it', 'en'],
                license='Apache 2.0', page='https://huggingface.co/datasets/CohereLabs/aya_dataset', files=AYA),
    'oasst': dict(title='OpenAssistant OASST2 · conversazioni scritte da volontari, italiano e inglese', languages=['it', 'en'],
                  license='Apache 2.0', page='https://huggingface.co/datasets/OpenAssistant/oasst2', files=OASST),
    'redix': dict(title='Everyday conversations ITA (ReDiX) · conversazioni quotidiane in italiano', languages=['it'],
                  license='Apache 2.0', page='https://huggingface.co/datasets/ReDiX/everyday-conversations-ita', files=REDIX),
    'wiki_it': dict(title='Wikipedia in italiano (novembre 2023)', languages=['it'], license='CC BY-SA 3.0 / GFDL',
                    page='https://huggingface.co/datasets/wikimedia/wikipedia', files=WIKI_IT),
    'wiki_en': dict(title='Wikipedia in inglese (novembre 2023)', languages=['en'], license='CC BY-SA 3.0 / GFDL',
                    page='https://huggingface.co/datasets/wikimedia/wikipedia', files=WIKI_EN),
    'fineweb2_it': dict(title='FineWeb-2 italiano · testo web filtrato (file di test ufficiale)', languages=['it'],
                        license='ODC-By 1.0', page='https://huggingface.co/datasets/HuggingFaceFW/fineweb-2', files=FINEWEB2_IT),
}

MB = 1024*1024
# For each source: which files and, per language, the most megabytes of corpus it may write.
# Measured on the files: a SYNTH file gives about 7.8 MB of Italian conversations (4,900) and 208 MB
# of English; Aya 0.3 MB of Italian, OASST2 0.4 MB, ReDiX 5.6 MB. Italian conversations are the
# scarce part: the Italian language itself comes mostly from Wikipedia and FineWeb-2.
MIXTURES = {
    'prova': dict(title='Prova · circa 300 MB · da qualche giorno a qualche settimana sulla CPU', parts={
        'smoltalk': dict(files=[0, 1], caps=dict(en=80)), 'synth': dict(files=list(range(6)), caps=dict(it=40, en=15)),
        'aya': dict(files=[0, 1], caps=dict(it=5, en=5)), 'oasst': dict(files=[0, 1], caps=dict(it=5, en=10)),
        'redix': dict(files=[0], caps=dict(it=10)), 'wiki_it': dict(files=[9], caps=dict(it=70)),
        'wiki_en': dict(files=[1], caps=dict(en=40)), 'fineweb2_it': dict(files=[0], caps=dict(it=30))}),
    'base': dict(title='Base · circa 2 GB · mesi sulla CPU, giorni su una GPU', parts={
        'smoltalk': dict(files=[0, 1, 2, 3, 4], caps=dict(en=500)), 'synth': dict(files=list(range(30)), caps=dict(it=230, en=100)),
        'aya': dict(files=[0, 1], caps=dict(it=5, en=5)), 'oasst': dict(files=[0, 1], caps=dict(it=5, en=40)),
        'redix': dict(files=[0], caps=dict(it=10)), 'wiki_it': dict(files=[0, 1, 2, 3, 4], caps=dict(it=500)),
        'wiki_en': dict(files=[0, 1, 2, 3], caps=dict(en=400)), 'fineweb2_it': dict(files=[0], caps=dict(it=150))}),
    'esteso': dict(title='Esteso · circa 6 GB · per una GPU', parts={
        'smoltalk': dict(files=[0, 1, 2, 3, 4], caps=dict(en=900)), 'synth': dict(files=list(range(60)), caps=dict(it=460, en=300)),
        'aya': dict(files=[0, 1], caps=dict(it=5, en=5)), 'oasst': dict(files=[0, 1], caps=dict(it=5, en=40)),
        'redix': dict(files=[0], caps=dict(it=10)), 'wiki_it': dict(files=list(range(10)), caps=dict(it=2000)),
        'wiki_en': dict(files=list(range(12)), caps=dict(en=1700)), 'fineweb2_it': dict(files=[0], caps=dict(it=150))}),
}
REFERENCE_NOTE = ("Il riferimento fisso: i file di test ufficiali (SmolTalk, Aya, OASST2) e circa l'1% di ogni fonte per la "
                  "validazione e l'1% per il test, scelti con un hash del testo. Sono gli stessi per tutta la durata "
                  "dell'addestramento, quindi i miglioramenti fra checkpoint si confrontano su testo mai visto.")


def catalog():
    sources = [dict(id=key, title=item['title'], languages=item['languages'], license=item['license'], page=item['page'])
               for key, item in SOURCES.items()]
    mixtures = []
    for key, item in MIXTURES.items():
        download = sum(SOURCES[name]['files'][index]['bytes'] for name, part in item['parts'].items() for index in part['files'])
        size = sum(sum(part['caps'].values()) for part in item['parts'].values())
        mixtures.append(dict(id=key, title=item['title'], corpus_mb=size, download_mb=round(download/MB),
                             languages={language: sum(part['caps'].get(language, 0) for part in item['parts'].values())
                                        for language in ('it', 'en')}))
    return dict(sources=sources, mixtures=mixtures, reference=REFERENCE_NOTE)


# ---- turning records into training text -----------------------------------------------------------------------

def chat(turns):
    """Turns as (role, text) in the chat format of the SmolLM2 tokenizer."""
    names = {'human': 'user', 'user': 'user', 'prompter': 'user', 'gpt': 'assistant', 'assistant': 'assistant', 'system': 'system'}
    parts = [f"<|im_start|>{names[role]}\n{text.strip()}<|im_end|>" for role, text in turns if text and text.strip()]
    return '\n'.join(parts) if len(parts) >= 2 else None


def split_of(text, official_test=False):
    if official_test:
        return 'test'
    bucket = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big') % 1000
    return 'validation' if bucket < 10 else 'test' if bucket < 20 else 'train'


def records(source, path, file):
    """(language, text, official_test) for every usable record of one downloaded file."""
    import pyarrow.parquet as parquet
    test = '/test' in file['path'] or 'validation' in file['path']
    if source == 'redix':
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    item = json.loads(line)
                    text = chat([(turn['from'], turn['value']) for turn in item.get('conversations', [])])
                    if text:
                        yield 'it', text, False
        return
    if source == 'oasst':
        yield from _oasst(parquet.read_table(path, columns=['message_id', 'parent_id', 'text', 'role', 'lang', 'deleted']).to_pylist(), test)
        return
    columns = {'smoltalk': ['messages'], 'synth': ['language', 'query', 'synthetic_answer'],
               'aya': ['inputs', 'targets', 'language'], 'wiki_it': ['title', 'text'], 'wiki_en': ['title', 'text'],
               'fineweb2_it': ['text']}[source]
    reader = parquet.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=512, columns=columns):
        for row in batch.to_pylist():
            if source == 'smoltalk':
                text = chat([(turn['role'], turn['content']) for turn in row['messages']])
                language = 'en'
            elif source == 'synth':
                language = row['language']
                text = chat([('user', row['query'] or ''), ('assistant', row['synthetic_answer'] or '')])
            elif source == 'aya':
                language = {'Italian': 'it', 'English': 'en'}.get(row['language'])
                text = chat([('user', row['inputs'] or ''), ('assistant', row['targets'] or '')])
            elif source in ('wiki_it', 'wiki_en'):
                language = source[-2:]
                text = f"{row['title']}\n\n{row['text']}".strip() if row['text'] else None
            else:
                language, text = 'it', row['text']
            if text and language in ('it', 'en'):
                yield language, text, test


def _oasst(rows, test):
    """Every conversation from the first message to an assistant answer, in Italian or English."""
    by_id = {row['message_id']: row for row in rows if not row['deleted'] and row['lang'] in ('it', 'en')}
    parents = {row['parent_id'] for row in by_id.values()}
    for row in by_id.values():
        if row['role'] != 'assistant' or row['message_id'] in parents:
            continue                                            # whole threads only, ending with an answer
        thread, current = [], row
        while current is not None:
            thread.append((current['role'], current['text']))
            current = by_id.get(current['parent_id'])
        text = chat(reversed(thread))
        if text:
            yield row['lang'], text, test


# ---- download --------------------------------------------------------------------------------------------------

def download(file, target, check=lambda: None, attempts=5):
    """Download with resume (HTTP Range) and verify size and SHA-256; returns the path."""
    target = Path(target)
    partial = target.with_suffix(target.suffix+'.part')
    for attempt in range(attempts):
        try:
            done = partial.stat().st_size if partial.exists() else 0
            if done < file['bytes']:
                request = Request(file['url'], headers={'Range': f'bytes={done}-'} if done else {})
                with urlopen(request, timeout=60) as response, partial.open('ab' if done else 'wb') as stream:
                    if done and response.status != 206:                 # no resume offered: start again
                        stream.truncate(0)
                    while True:
                        check()
                        chunk = response.read(4*MB)
                        if not chunk:
                            break
                        stream.write(chunk)
            if partial.stat().st_size != file['bytes']:
                raise URLError('download incompleto')
            digest = hashlib.sha256()
            with partial.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8*MB), b''):
                    digest.update(chunk)
            if digest.hexdigest() != file['sha256']:
                partial.unlink()
                raise ValueError(f"File modificato o corrotto: {file['path']} (checksum diverso).")
            os.replace(partial, target)
            return target
        except (URLError, OSError, TimeoutError) as exc:
            if attempt == attempts-1:
                raise ValueError(f"Download non riuscito: {file['path']} ({exc}). Riprendi più tardi: il lavoro fatto resta.") from None
            time.sleep(5*(attempt+1))


# ---- the preparation, resumable --------------------------------------------------------------------------------

class Paused(Exception):
    pass


def plan(mixture):
    """The ordered list of (source, file index) of a mixture."""
    return [(name, index) for name, part in MIXTURES[mixture]['parts'].items() for index in part['files']]


def build(directory, mixture, check=lambda: None, report=lambda phase, state: None):
    """Write ``directory/corpus.jsonl``; ``state.json`` makes it resumable. Returns the final state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path, output = directory/'state.json', directory/'corpus.jsonl'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        mixture=mixture, position=0, row=0, written=0, by_part={}, records={'train': 0, 'validation': 0, 'test': 0},
        done=False)
    if state['mixture'] != mixture:
        raise ValueError('La preparazione salvata è di un\'altra miscela.')
    steps = plan(mixture)

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state))
        os.replace(temporary, state_path)
    with output.open('ab') as stream:
        stream.truncate(state['written'])                       # drop a write interrupted after the last save
        stream.seek(state['written'])
        while state['position'] < len(steps):
            name, index = steps[state['position']]
            part, file = MIXTURES[mixture]['parts'][name], SOURCES[name]['files'][index]
            caps = {language: megabytes*MB for language, megabytes in part['caps'].items()}
            used = lambda language: state['by_part'].get(f'{name}:{language}', 0)
            if all(used(language) >= cap for language, cap in caps.items()):
                state.update(position=state['position']+1, row=0)
                save()
                continue
            free = shutil.disk_usage(directory).free
            if free < file['bytes'] + 2*1024**3:
                raise ValueError(f"Spazio su disco insufficiente per scaricare {file['path']} "
                                 f"({file['bytes']/MB:.0f} MB più 2 GB di riserva).")
            report(f"Scarico {SOURCES[name]['title']} · file {index+1} ({file['bytes']/MB:.0f} MB)", state)
            local = download(file, directory/('source-'+Path(file['path']).name), check)
            report(f"Leggo {SOURCES[name]['title']} · file {index+1}", state)
            for row, (language, text, test) in enumerate(records(name, local, file)):
                if row < state['row']:
                    continue
                if language in caps and used(language) < caps[language]:
                    role = split_of(text, test)
                    line = (json.dumps(dict(text=text, split=role), ensure_ascii=False)+'\n').encode()
                    stream.write(line)
                    state['written'] += len(line)
                    state['by_part'][f'{name}:{language}'] = used(language)+len(line)
                    state['records'][role] += 1
                state['row'] = row+1
                if state['row'] % 2000 == 0:
                    stream.flush()
                    os.fsync(stream.fileno())
                    save()
                    report(f"Leggo {SOURCES[name]['title']} · file {index+1}", state)
                    check()
                if all(used(language) >= cap for language, cap in caps.items()):
                    break
            stream.flush()
            os.fsync(stream.fileno())
            local.unlink()
            state.update(position=state['position']+1, row=0)
            save()
    state['done'] = True
    save()
    return state


def run_in_worker(root, identifier, stop_event=None):
    """The preparation requested from the laboratory; at the end the corpus is ready to train on."""
    from .store import Store
    store = Store(root)
    item = store.build(identifier)
    directory = store.directory('builds', identifier)

    def check():
        if stop_event is not None and stop_event.is_set():
            raise Paused()
        if store.build(identifier)['status'] == 'pausing':
            raise Paused()

    def report(phase, state):
        store.update_build(identifier, phase=phase, result=dict(progress=progress(state)))
    try:
        store.update_build(identifier, status='running', error=None)
        state = build(directory, item['request']['mixture'], check, report)
        name = f"dialogo-{item['request']['mixture']}.jsonl"
        corpus = store.adopt_corpus(item['owner'], name, directory/'corpus.jsonl')
        provenance = dict(mixture=item['request']['mixture'], title=MIXTURES[item['request']['mixture']]['title'],
                          sources={key: dict(title=SOURCES[key]['title'], license=SOURCES[key]['license'], page=SOURCES[key]['page'],
                                             files=[SOURCES[key]['files'][index] for index in part['files']])
                                   for key, part in MIXTURES[item['request']['mixture']]['parts'].items()},
                          records=state['records'], bytes=state['written'], reference=REFERENCE_NOTE, prepared_at=time.time())
        (store.directory('corpora', corpus['id'])/'dialogue-source.json').write_text(json.dumps(provenance, indent=1))
        store.update_build(identifier, status='completed', phase='Pronto: il testo è fra quelli da addestrare',
                           result=dict(progress=progress(state), corpus_id=corpus['id'], corpus_name=corpus['name']))
    except Paused:
        store.update_build(identifier, status='paused', phase='In pausa: riprendi quando vuoi, il lavoro fatto resta')
    except Exception as exc:
        store.update_build(identifier, status='failed', error=str(exc))
        raise


def progress(state):
    steps = plan(state['mixture'])
    total = sum(sum(part['caps'].values()) for part in MIXTURES[state['mixture']]['parts'].values())*MB
    return dict(files_done=state['position'], files=len(steps), written_mb=round(state['written']/MB, 1),
                target_mb=round(total/MB), records=state['records'])
