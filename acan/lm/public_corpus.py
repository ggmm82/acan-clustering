"""Explicit, bounded public-corpus experiments; no crawling or arbitrary URLs.

No change to the model/checkpoint contract. Source bytes are verified before a
workspace-local corpus is created. The existing durable training worker is used.
"""
import fcntl
import hashlib
import json
import io
import time
from urllib.error import URLError
from urllib.request import urlopen

from .manager import launch
from .store import atomic_json, digest_file
from .research_options import validate_options

CATALOG = {
    'dante-it': dict(
        title='Divina Commedia · italiano letterario', language='it',
        url='https://www.gutenberg.org/cache/epub/1000/pg1000.txt',
        source='https://www.gutenberg.org/ebooks/1000',
        license='Opera originale di pubblico dominio; edizione Project Gutenberg',
        license_url='https://www.gutenberg.org/policy/license.html',
        sha256='4669dcc00ee61ceffe92d871e61ea430cec87b35cbab24f19a4c0b1c7da521b2',
        bytes=597903, format='gutenberg',
        note='Poesia medievale: prova tecnica, non addestramento di un assistente italiano moderno.',
        prompt='Nel mezzo del cammin'),
    'tiny-shakespeare': dict(
        title='Tiny Shakespeare · inglese', language='en',
        url='https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt',
        source='https://github.com/karpathy/char-rnn/tree/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare',
        license='Testi originali di Shakespeare di pubblico dominio; raccolta char-rnn',
        license_url='https://github.com/karpathy/char-rnn/blob/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/LICENSE',
        sha256='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed',
        bytes=1115394, format='text',
        note='Piccolo benchmark narrativo/teatrale inglese; non insegna a rispondere come un assistente.',
        prompt='First Citizen:\n'),
}
CATALOG['wikitext-2'] = {'title': 'WikiText-2 · inglese · train/valid/test ufficiali', 'language': 'en', 'url': 'https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1', 'source': 'https://huggingface.co/datasets/Salesforce/wikitext', 'license': 'CC BY-SA 3.0 / GFDL · articoli Wikipedia', 'license_url': 'https://huggingface.co/datasets/Salesforce/wikitext#licensing-information', 'sha256': '5cd2a8282508e6c9c61f146c25d07da32c735e2e185ac3023c738bf630d95b54', 'bytes': 7747362, 'format': 'wikitext', 'note': 'Benchmark linguistico inglese con split ufficiali separati; non insegna da solo a rispondere come un assistente.', 'prompt': 'The history of', 'files': [{'split': 'train', 'bytes': 6357543, 'sha256': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 'url': 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet'}, {'split': 'validation', 'bytes': 657209, 'sha256': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 'url': 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/validation-00000-of-00001.parquet'}, {'split': 'test', 'bytes': 732610, 'sha256': '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91', 'url': 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet'}]}

PRESETS = {
    'check': dict(title='Prova breve · massimo 20 passi · 1.280 token elaborati', total_steps=20,
                  sequence_length=64, accumulation=1, save_steps=5),
    'practice': dict(title='Esercizio · massimo 200 passi · 102.400 token elaborati', total_steps=200,
                     sequence_length=128, accumulation=4, save_steps=20),
}
MAX_BYTES = 8 * 1024 * 1024


def catalog():
    return dict(datasets=[dict(id=key, **value) for key, value in CATALOG.items()],
                presets=[dict(id=key, **value) for key, value in PRESETS.items()])


def download(spec):
    started = time.monotonic()
    chunks, size = [], 0
    try:
        with urlopen(spec['url'], timeout=8) as response:
            while True:
                chunk = response.read(256*1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES or time.monotonic()-started > 15:
                    raise ValueError('Download interrotto: limite di dimensione o tempo raggiunto.')
                chunks.append(chunk)
    except (OSError, URLError) as exc:
        raise ValueError('Corpus pubblico non raggiungibile; riprova più tardi.') from exc
    return b''.join(chunks)


def prepare_source(raw, spec):
    if len(raw) != spec['bytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('Il corpus pubblico è cambiato o è incompleto: checksum non valido. Nessun training avviato.')
    text = raw.decode('utf-8-sig').replace('\r\n', '\n')
    if spec['format'] == 'gutenberg':
        start = '*** START OF THE PROJECT GUTENBERG EBOOK '
        end = '*** END OF THE PROJECT GUTENBERG EBOOK '
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError('Marcatori del corpus Gutenberg non riconosciuti.')
        text = text.split(start, 1)[1].split('\n', 1)[1].split(end, 1)[0]
    # Preserve paragraph/document boundaries and remove empty edge lines only.
    return (text.strip()+'\n').encode('utf-8')


def prepare_wikitext(spec):
    import pyarrow.parquet as parquet
    records=[]
    for file in spec['files']:
        raw=download(file)
        if len(raw)!=file['bytes'] or hashlib.sha256(raw).hexdigest()!=file['sha256']:
            raise ValueError('Split WikiText modificato o incompleto: '+file['split'])
        lines=parquet.read_table(io.BytesIO(raw),columns=['text'])['text'].to_pylist()
        current=[]
        def emit():
            text='\n'.join(current).strip()
            if text:
                records.append(json.dumps(dict(text=text,split=file['split']),ensure_ascii=False))
        for line in lines:
            heading=line.strip()
            if heading.startswith('= ') and heading.endswith(' =') and heading.count('=')==2 and current:
                emit();current=[]
            current.append(line.rstrip())
        emit()
    content=('\n'.join(records)+'\n').encode()
    if len(content)>32*1024*1024:
        raise ValueError('Corpus preparato oltre il budget di importazione.')
    return content


def experiment(store, owner, dataset, preset):
    if not isinstance(dataset, str) or not isinstance(preset, str) or dataset not in CATALOG or preset not in PRESETS:
        raise ValueError('Seleziona un corpus pubblico e un profilo disponibili.')
    if store.busy():
        raise ValueError('Worker occupato: metti in pausa il progetto attivo prima di avviare una prova pubblica.')
    spec, settings = CATALOG[dataset], PRESETS[preset]
    # A double click/retry cannot create overlapping imports for this workspace.
    with (store.root / ('public-'+owner+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Preparazione di un corpus pubblico già in corso.') from None
        corpus, provenance = None, None
        for candidate in store.corpora(owner):
            metadata = store.directory('corpora', candidate['id']) / 'public-source.json'
            if candidate['status'] != 'ready' or not metadata.exists():
                continue
            origin = json.loads(metadata.read_text())
            path = metadata.parent / 'corpus'
            if (origin.get('dataset') == dataset and origin.get('source_sha256') == spec['sha256']
                    and digest_file(path) == candidate['sha256'] == origin.get('prepared_sha256')):
                corpus, provenance = candidate, origin
                break
        if corpus is None:
            if spec['format']=='wikitext':
                content=prepare_wikitext(spec)
            else:
                raw=download(spec)
                content=prepare_source(raw,spec)
            corpus=store.create_corpus(owner,dataset+('.jsonl' if spec['format']=='wikitext' else '.txt'),len(content))
            for offset in range(0,len(content),2*1024*1024):
                store.append_corpus(corpus['id'],owner,offset,content[offset:offset+2*1024*1024])
            corpus = store.finish_corpus(corpus['id'], owner)
            provenance = dict(dataset=dataset, title=spec['title'], source=spec['source'],
                              url=spec['url'], license=spec['license'], license_url=spec['license_url'],
                              source_sha256=spec['sha256'], prepared_sha256=corpus['sha256'],
                              source_bytes=spec['bytes'], prepared_bytes=len(content), files=spec.get('files'),
                              preprocessing=('wikitext-articles-official-splits-v1' if spec['format']=='wikitext' else 'utf8-newlines-v1; Gutenberg header/footer removed when present'),
                              imported_at=time.time())
            atomic_json(store.directory('corpora', corpus['id'])/'public-source.json', provenance)
        config = validate_options(dict(corpus_id=corpus['id'], name=spec['title']+' · '+settings['title'],
                                       base_model='HuggingFaceTB/SmolLM2-135M', batch_size=1,
                                       **{k:v for k,v in settings.items() if k != 'title'}))
        job = store.create_job(owner, config)
        atomic_json(store.directory('jobs', job['id'])/'public-source.json', provenance)
        # Another workspace may acquire the accelerator while the download runs.
        # Keep the created project selectable and resumable rather than lose it.
        message = 'Corpus verificato; training avviato. Puoi mettere in pausa e riprendere lo stesso progetto.'
        try:
            launch(store, job)
        except ValueError as exc:
            message = f'Corpus e progetto salvati. Training non avviato: {exc} Usa Riprendi quando il worker è libero.'
        return dict(job=store.job(job['id'], owner), corpus=corpus, source=provenance,
                    prompt=spec['prompt'], message=message)
