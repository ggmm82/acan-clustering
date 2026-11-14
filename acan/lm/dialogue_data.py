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
# Other kinds of data, chosen on top of the Italian and English text.
CODE = _files('sahil2801/CodeAlpaca-20k', '152bb5e9a29651266b018106053980070a0521a1', [
    ('code_alpaca_20k.json', 8057896, '58d136bb7ea8e4b7c5b6ef08831da905d0abfafeb56f8fa519083089dd6a210e')])
BASH = _files('neulab/tldr', '62e85e84102b5ebb6a412cf0f4245227ea897a08', [
    ('tldr-test.jsonl', 524944, 'a398a34e2728d6990ac93ef74ed4dffe22154083bc27ec64fdad32084e9de818'),
    ('tldr-train.jsonl', 4756295, '823ffdc1e6a0f92a0cf6914b342e7f100f2f842f01978843ae0fddfe19edbe15')])
MUSIC = _files('sander-wood/irishman', '30902e69ca45266207f8466e0d04e4bc742c5604', [
    ('validation.json', 796938, 'a33935f9047edc976e087b146737ed3737a9315c2bf52ff02d8873b057c575ae'),
    ('train.json', 79975421, '5706880c0f935e69108244e010784935b7aba855f37ae2ce7f4e4d170851835c')])
FASHION = _files('zalando-datasets/fashion_mnist', '531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2', [
    ('fashion_mnist/test-00000-of-00001.parquet', 5175617, 'fb36d8b5b74a491bce99357dcee62a32d347a356ea2a35d1c9057e7a7435ca5a'),
    ('fashion_mnist/train-00000-of-00001.parquet', 30931277, 'a5769a70af055b15ada8bd3dad9057557554fdd30e17cba45e7ef0f8bf79c5bb')])
MNIST = _files('ylecun/mnist', '77f3279092a1c1579b2250db8eafed0ad422088c', [
    ('mnist/test-00000-of-00001.parquet', 2595890, 'd49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b'),
    ('mnist/train-00000-of-00001.parquet', 15561616, 'f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf')])
DIGITS_SPOKEN = _files('mteb/free-spoken-digit-dataset', 'c34455c99604d35cb8d27328c267be1478efc903', [
    ('data/test-00000-of-00001.parquet', 1980617, 'eeef2d13b4307e65a82a43773758b6872e3ebcefb2a928ab42f1594929f1a1e6'),
    ('data/train-00000-of-00001.parquet', 17651752, '0131a53186276d2c42f99f1d7b6f95d271928083bcf2ced690f863ab14ed941e')])
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
    'code': dict(title='Code Alpaca · 20.000 richieste di programmazione con la soluzione (inglese)', languages=['en'],
                 license='CC BY 4.0', page='https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k', files=CODE),
    'bash': dict(title='tldr (DocPrompting) · comandi bash spiegati in una frase, dalle pagine tldr', languages=['en'],
                 license='MIT (pagine tldr: CC BY 4.0)', page='https://huggingface.co/datasets/neulab/tldr', files=BASH),
    'music': dict(title='IrishMAN · melodie in notazione ABC', languages=['abc'], license='MIT',
                  page='https://huggingface.co/datasets/sander-wood/irishman', files=MUSIC),
    'fashion': dict(title='Fashion-MNIST · immagini di capi di abbigliamento con il loro nome', languages=['it', 'en'],
                    license='MIT', page='https://huggingface.co/datasets/zalando-datasets/fashion_mnist', files=FASHION),
    'digits': dict(title='MNIST · cifre scritte a mano con la cifra', languages=['it', 'en'], license='MIT',
                   page='https://huggingface.co/datasets/ylecun/mnist', files=MNIST),
    'spoken': dict(title='Free Spoken Digit Dataset · cifre pronunciate (WAV 8 kHz)', languages=['en'], license='CC BY-SA 4.0',
                   page='https://huggingface.co/datasets/mteb/free-spoken-digit-dataset', files=DIGITS_SPOKEN),
    'moving': dict(title='Cifre in movimento · brevi video 16×16 generati dalle cifre MNIST', languages=['it', 'en'],
                   license='MIT (da MNIST)', page='https://huggingface.co/datasets/ylecun/mnist', files=MNIST),
}
# Other kinds of data the long training can add to the text; each part caps the megabytes it writes.
DOMAINS = {
    'code': dict(title='Codice (Python e altri linguaggi)', parts={'code': dict(files=[0], caps=dict(en=12))}),
    'bash': dict(title='Comandi bash', parts={'bash': dict(files=[0, 1], caps=dict(en=8))}),
    'music': dict(title='Musica in notazione ABC', parts={'music': dict(files=[0, 1], caps=dict(abc=60))}),
    'images': dict(title='Immagini 32×32 con il loro nome', parts={'fashion': dict(files=[0, 1], caps=dict(it=60, en=60)),
                                                                   'digits': dict(files=[0, 1], caps=dict(it=30, en=30))}),
    'audio': dict(title='Suoni: cifre pronunciate', parts={'spoken': dict(files=[0, 1], caps=dict(en=120))}),
    'video': dict(title='Video: cifre in movimento', parts={'moving': dict(files=[1], caps=dict(it=40, en=40))}),
}
FASHION_NAMES = [('maglietta', 't-shirt'), ('pantaloni', 'trousers'), ('maglione', 'pullover'), ('vestito', 'dress'),
                 ('cappotto', 'coat'), ('sandalo', 'sandal'), ('camicia', 'shirt'), ('scarpa da ginnastica', 'sneaker'),
                 ('borsa', 'bag'), ('stivaletto', 'ankle boot')]

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
    domains = []
    for key, item in DOMAINS.items():
        download = sum(SOURCES[name]['files'][index]['bytes'] for name, part in item['parts'].items() for index in part['files'])
        domains.append(dict(id=key, title=item['title'], corpus_mb=sum(sum(part['caps'].values()) for part in item['parts'].values()),
                            download_mb=round(download/MB), sources=[SOURCES[name]['title'] for name in item['parts']]))
    return dict(sources=sources, mixtures=mixtures, domains=domains, reference=REFERENCE_NOTE)


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
    name = file['path'].rsplit('/', 1)[-1]
    test = '/test' in file['path'] or 'validation' in file['path'] or name.startswith(('test', 'tldr-test'))
    extra = _extra_records(source, path, test)
    if extra is not None:
        yield from extra
        return
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


def _extra_records(source, path, test):
    """Records of the other kinds of data, or None for the text sources."""
    import pyarrow.parquet as parquet
    if source == 'code':
        def code():
            for item in json.loads(Path(path).read_text(encoding='utf-8')):
                question = item['instruction'] + (f"\n\n{item['input']}" if item.get('input') else '')
                text = chat([('user', question), ('assistant', item['output'])])
                if text:
                    yield 'en', text, test
        return code()
    if source == 'bash':
        def bash():
            with open(path, encoding='utf-8') as stream:
                for line in stream:
                    item = json.loads(line)
                    command = item['cmd'].replace('{{', '').replace('}}', '')
                    yield 'en', chat([('user', item['nl']), ('assistant', command)]), test
        return bash()
    if source == 'music':
        def music():
            for item in json.loads(Path(path).read_text(encoding='utf-8')):
                if item.get('abc notation'):
                    yield 'abc', item['abc notation'].strip(), test
        return music()
    if source in ('fashion', 'digits', 'spoken', 'moving'):
        def media_records():
            from .media import encode_audio, encode_image
            column = 'audio' if source == 'spoken' else 'image'
            rows = parquet.ParquetFile(path)
            index = 0
            for batch in rows.iter_batches(batch_size=256, columns=[column, 'label']):
                for row in batch.to_pylist():
                    label, raw = row['label'], row[column]['bytes']
                    language = 'it' if index % 2 == 0 else 'en'          # Italian and English requests in turn
                    index += 1
                    if source == 'fashion':
                        names = FASHION_NAMES[label]
                        question = f'Disegna: {names[0]}.' if language == 'it' else f'Draw: {names[1]}.'
                        yield language, chat([('user', question), ('assistant', encode_image(raw))]), test
                    elif source == 'digits':
                        question = f'Disegna la cifra {label}.' if language == 'it' else f'Draw the digit {label}.'
                        yield language, chat([('user', question), ('assistant', encode_image(raw))]), test
                    elif source == 'spoken':
                        # 4 kHz without the silence around it, so a whole word fits in a piece of 2,048 tokens.
                        yield 'en', chat([('user', f'Say the digit {label}.'), ('assistant', encode_audio(raw, 4000, trim=True))]), test
                    else:
                        question = f'Mostra la cifra {label} che si muove.' if language == 'it' else f'Show the digit {label} moving.'
                        yield language, chat([('user', question), ('assistant', moving_digit(raw, index))]), test
        return media_records()
    return None


def moving_digit(raw, seed, frames=7):
    """A short video: a handwritten digit, shrunk to 8 x 8, bouncing inside a 16 x 16 frame.

    Seven frames are about 1,900 tokens: a whole video fits in a piece of at most 2,048 tokens.
    """
    import random
    from PIL import Image
    from .media import VIDEO_SIZE, _colour
    digit = Image.open(__import__('io').BytesIO(raw)).convert('L').resize((8, 8))
    rng = random.Random(seed)
    x, y = rng.randint(0, VIDEO_SIZE-8), rng.randint(0, VIDEO_SIZE-8)
    dx, dy = rng.choice((-1, 1)), rng.choice((-1, 1))
    parts = []
    for _ in range(frames):
        canvas = Image.new('L', (VIDEO_SIZE, VIDEO_SIZE))
        canvas.paste(digit, (x, y))
        pixels = list(canvas.getdata())
        parts.append('<|frame|>' + ''.join(''.join(f'<|p{_colour(value, value, value)}|>' for value in pixels[row*VIDEO_SIZE:(row+1)*VIDEO_SIZE])
                                           + '<|row|>' for row in range(VIDEO_SIZE)))
        if not 0 <= x+dx <= VIDEO_SIZE-8:
            dx = -dx
        if not 0 <= y+dy <= VIDEO_SIZE-8:
            dy = -dy
        x, y = x+dx, y+dy
    return '<|video|>' + ''.join(parts) + '<|/video|>'


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


def parts_of(mixture, extras=()):
    """Every part of a mixture and of the chosen other kinds of data, as (name, part)."""
    found = list(MIXTURES[mixture]['parts'].items())
    for domain in extras:
        found += list(DOMAINS[domain]['parts'].items())
    return found


def plan(mixture, extras=()):
    """The ordered list of (source, file index, part) of a mixture with its extras."""
    return [(name, index, part) for name, part in parts_of(mixture, extras) for index in part['files']]


def build(directory, mixture, check=lambda: None, report=lambda phase, state: None, extras=()):
    """Write ``directory/corpus.jsonl``; ``state.json`` makes it resumable. Returns the final state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path, output = directory/'state.json', directory/'corpus.jsonl'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        mixture=mixture, extras=list(extras), position=0, row=0, written=0, by_part={},
        records={'train': 0, 'validation': 0, 'test': 0}, done=False)
    if state['mixture'] != mixture or state.get('extras', []) != list(extras):
        raise ValueError('La preparazione salvata è di un\'altra miscela.')
    steps = plan(mixture, extras)

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state))
        os.replace(temporary, state_path)
    with output.open('ab') as stream:
        stream.truncate(state['written'])                       # drop a write interrupted after the last save
        stream.seek(state['written'])
        while state['position'] < len(steps):
            name, index, part = steps[state['position']]
            file = SOURCES[name]['files'][index]
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
        mixture, extras = item['request']['mixture'], item['request'].get('extras', [])
        state = build(directory, mixture, check, report, extras)
        name = f"dialogo-{mixture}" + ''.join(f'-{domain}' for domain in extras) + '.jsonl'
        corpus = store.adopt_corpus(item['owner'], name, directory/'corpus.jsonl')
        provenance = dict(mixture=mixture, extras=extras, title=MIXTURES[mixture]['title'],
                          sources={key: dict(title=SOURCES[key]['title'], license=SOURCES[key]['license'], page=SOURCES[key]['page'],
                                             files=[SOURCES[key]['files'][index] for index in part['files']])
                                   for key, part in parts_of(mixture, extras)},
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
    extras = state.get('extras', [])
    steps = plan(state['mixture'], extras)
    total = sum(sum(part['caps'].values()) for _, part in parts_of(state['mixture'], extras))*MB
    return dict(files_done=state['position'], files=len(steps), written_mb=round(state['written']/MB, 1),
                target_mb=round(total/MB), records=state['records'])
