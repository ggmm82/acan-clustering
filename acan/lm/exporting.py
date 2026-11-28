"""Workspace-authorized inspectable model deltas, without corpus or optimizer."""
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
import zipfile
from . import checkpoints
from .store import atomic_json, digest_file


def export_bundle(store, job, name):
    from .manager import snapshots
    if name not in {item['name'] for item in snapshots(store, job)}:
        raise ValueError('Checkpoint non disponibile: aggiorna la lista.')
    directory = store.directory('jobs', job['id'])
    snapshot = directory/'checkpoints'/name
    receipt = checkpoints.verify(snapshot)
    if job['config'].get('architecture') == 'concepts':
        return _export_concepts(directory, snapshot, name, receipt, job)
    state = json.loads((snapshot/'state.json').read_text())
    from .research_model import check_inference_contract
    check_inference_contract(job['config'],state)
    base = json.loads((directory/'base.json').read_text())
    native = base.get('native', False)
    if native:                                                  # born from zero: all its weights and its tokenizer travel together
        from .native import SIZES
        size = SIZES[job['config']['native_size']]
        config = dict(hidden_size=size['hidden'], num_hidden_layers=size['layers'], vocab_size=state['graft'].get('vocabulary'))
    else:
        from huggingface_hub import snapshot_download
        base_path = Path(snapshot_download(base['repo'], revision=base['revision'],
                                          cache_dir=str(store.root/'hub'), local_files_only=True))
        config_path = base_path/'config.json'
        if digest_file(config_path) != base['files_sha256']['config.json']:
            raise ValueError('Configurazione del modello originale modificata.')
        config = json.loads(config_path.read_text())
    from safetensors import safe_open
    tensors = {}
    with safe_open(snapshot/'trainables.safetensors', framework='pt', device='cpu') as reader:
        for key in reader.keys():
            tensor = reader.get_slice(key)
            tensors[key] = dict(shape=tensor.get_shape(), dtype=tensor.get_dtype())
    replaced = state['graft']['replaced_layers']
    architecture = dict(base_model=base['repo'], base_revision=base['revision'],
                        hidden_size=config['hidden_size'], vocabulary=config['vocab_size'],
                        slots=job['config']['slots'], local_window=job['config']['window'],
                        layers=[dict(index=i, attention='ACANConceptAttentionLayer' if i in replaced else 'original dense attention')
                                for i in range(config['num_hidden_layers'])],
                        trainable_parameters=sum(math.prod(item['shape']) for item in tensors.values()),
                        trainable_tensors=tensors, total_parameters=state.get('total_parameters'),
                        note='Slots/channels are latent numerical variables, not certified linguistic explanations.')
    exports = directory/'exports'; exports.mkdir(exist_ok=True)
    destination = exports/(name+'.zip')
    if destination.exists():
        return destination
    payloads = {
        'trainables.safetensors': (snapshot/'trainables.safetensors').read_bytes(),
        'base.json': json.dumps(base, indent=2).encode(),
        'config.json': json.dumps(job['config'], indent=2).encode(),
        'architecture.json': json.dumps(architecture, indent=2).encode(),
        'training-summary.json': json.dumps({key:state.get(key) for key in ('step','training_nll','training_objective','training_cp_penalty','cp_weight','concept_counts','validation_nll','baseline_nll','tokens_trained','early_stop_reason')}, indent=2).encode(),
        'README.txt': ('Pacchetto di ispezione ACAN-LM Hybrid. Contiene i pesi appresi ACAN/LoRA e il riferimento esatto al modello originale, NON tutti i pesi originali.\n'
                       'Non contiene corpus, chiave workspace, token del servizio, stato RNG o optimizer pickle.\n'
                       'Installa questo progetto con extra [llm]. Estrai lo ZIP in una cartella e usa:\n'
                       'from acan.lm.exporting import load_export\nmodel, tokenizer = load_export("cartella_estratta", device="cpu")\n'
                       'Il primo caricamento scarica i pesi originali della revisione registrata. architecture.json descrive livelli e tensori.\n').encode(),
    }
    if native:
        for item in sorted((directory/'tokenizer').iterdir()):
            payloads['tokenizer-'+item.name] = item.read_bytes()
        payloads['README.txt'] = ('Modello a concetti ACAN nativo, nato da zero: contiene TUTTI i pesi appresi e il suo tokenizer.\n'
                                  'Non contiene corpus, chiave workspace, token del servizio, stato RNG o optimizer.\n'
                                  'Installa questo progetto con extra [llm]. Estrai lo ZIP in una cartella e usa:\n'
                                  'from acan.lm.exporting import load_export\nmodel, tokenizer = load_export("cartella_estratta", device="cpu")\n').encode()
    provenance = directory/'public-source.json'
    if provenance.exists():
        payloads['public-source.json'] = provenance.read_bytes()
    manifest = dict(format=1, core_contract=receipt['contract'],model_contract=state.get('model_contract'), checkpoint=name, identity=receipt['identity'],
                    files_sha256={key:hashlib.sha256(value).hexdigest() for key,value in payloads.items()})
    temporary = exports/(uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for key, value in payloads.items():
                archive.writestr(key, value)
            archive.writestr('export-manifest.json', json.dumps(manifest, indent=2))
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_export(directory, device='cpu', cache_dir=None):
    """Reconstruct the actual grafted model; never unpickle optimizer state."""
    import torch
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from .research_model import load_model as load_base
    from .grafting import reset_streams
    directory = Path(directory)
    manifest = json.loads((directory/'export-manifest.json').read_text())
    if manifest.get('format') != 1 or manifest.get('core_contract') != checkpoints.contract():
        raise ValueError('Export incompatibile: usare la versione del worker che lo ha creato.')
    required = {'base.json','config.json','architecture.json','trainables.safetensors','training-summary.json','README.txt'}
    if not required.issubset(manifest['files_sha256']):
        raise ValueError('Export incompleto.')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest_file(directory/name) != expected:
            raise ValueError('File export modificato: '+name)
    base = json.loads((directory/'base.json').read_text())
    config = json.loads((directory/'config.json').read_text())
    from .research_model import check_inference_contract
    check_inference_contract(config,manifest)
    if base.get('native'):
        import tempfile, shutil
        folder = Path(tempfile.mkdtemp(prefix='acan-tokenizer-'))
        for name in manifest['files_sha256']:
            if name.startswith('tokenizer-'):
                shutil.copy(directory/name, folder/name[len('tokenizer-'):])
        model, _ = load_base(str(folder), config, device)
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
        path = None
    else:
        path = snapshot_download(base['repo'], revision=base['revision'], cache_dir=cache_dir,
                             allow_patterns=['*.json','*.safetensors','*.model','tokenizer.*','vocab.*','merges.txt'])
    if path is not None:
        for name, expected in base['files_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or digest_file(Path(path)/name) != expected:
                raise ValueError('File del modello originale modificato: '+name)
        model, _ = load_base(path, config, device)
        tokenizer = None
    tensors = load_file(directory/'trainables.safetensors', device=device)
    parameters = {key:parameter for key,parameter in model.named_parameters() if parameter.requires_grad}
    if set(tensors) != set(parameters):
        raise ValueError('Tensori addestrabili incompatibili.')
    with torch.no_grad():
        for key, parameter in parameters.items():
            if parameter.shape != tensors[key].shape:
                raise ValueError('Dimensioni incompatibili: '+key)
            parameter.copy_(tensors[key])
    model.eval();reset_streams(model)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    return model, tokenizer


CONCEPT_README = """Modello linguistico a concetti ACAN.

concepts.safetensors   i parametri della rete a concetti (le voci net.*): i concetti dei token e di ogni strato,
                       imparati col gradiente, e la memoria del documento. Tutto si legge con numpy, senza PyTorch.
                       (safetensors: un formato standard e sicuro per tabelle di numeri)
tokenizer/             il tokenizer che trasforma il testo in token e viceversa
scrivi.py              uno script pronto: carica il modello e scrive
config.json            le impostazioni dell'apprendimento
training-summary.json  passi, token letti, errore su testo mai visto, concetti per strato
export-manifest.json   le impronte SHA-256 di ogni file

Non contiene il testo di addestramento.

Per usarlo serve Python con il pacchetto acan (lo stesso del server), transformers e safetensors:
    pip install safetensors transformers
    python scrivi.py "Ciao! Chi sei?"
    python scrivi.py --testo "C'era una volta"

In Python:
    from acan.lm.concept_lm import from_checkpoint_arrays
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer
    model = from_checkpoint_arrays(load_file("concepts.safetensors"))
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    ids = tokenizer("C'era una volta")["input_ids"]
    print(tokenizer.decode(model.generate(ids, 64, temperature=0.7, seed=0)))
    p = model.distribution(ids)      # la probabilità di ogni token del vocabolario
"""

CONCEPT_SCRIPT = """\"\"\"Scrive con il modello a concetti di questa cartella: python scrivi.py "domanda" (conversazione) o --testo "inizio".\"\"\"
import argparse
from pathlib import Path

from safetensors.numpy import load_file
from transformers import AutoTokenizer

from acan.lm.concept_lm import from_checkpoint_arrays

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('domanda', nargs='?', default='Ciao! Chi sei?')
parser.add_argument('--testo', help='continua questo testo invece di rispondere a una domanda')
parser.add_argument('--token', type=int, default=128)
parser.add_argument('--temperatura', type=float, default=0.7)
parser.add_argument('--seme', type=int, default=0)
args = parser.parse_args()
arrays = load_file(str(here/'concepts.safetensors'))
model = from_checkpoint_arrays(arrays)
tokenizer = AutoTokenizer.from_pretrained(str(here/'tokenizer'), local_files_only=True)
prompt = args.testo if args.testo else f"<|im_start|>user\\n{args.domanda.strip()}<|im_end|>\\n<|im_start|>assistant\\n"
stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>'), tokenizer.convert_tokens_to_ids('<|im_start|>')} - {None, tokenizer.unk_token_id}
produced = model.generate(tokenizer(prompt)['input_ids'], args.token, stop=stop, temperature=args.temperatura, seed=args.seme)
print(tokenizer.decode(produced, skip_special_tokens=True))
"""


def _export_concepts(directory, snapshot, name, receipt, job):
    """The concept language model: its sums, its weights and its tokenizer, readable with numpy alone."""
    exports = directory/'exports'; exports.mkdir(exist_ok=True)
    destination = exports/(name+'-v3.zip')                     # v3: the script also stops at the start of a new turn
    if destination.exists():
        return destination
    state = json.loads((snapshot/'state.json').read_text())
    payloads = {'concepts.safetensors': (snapshot/'trainables.safetensors').read_bytes(),
                'config.json': json.dumps(job['config'], indent=2).encode(),
                'training-summary.json': json.dumps({key: state.get(key) for key in ('step', 'tokens_trained', 'validation_nll', 'baseline_nll',
                                                     'concepts_per_level', 'concepts_total', 'context_tokens',
                                                     'statistics_bytes')}, indent=2).encode(),
                'scrivi.py': CONCEPT_SCRIPT.encode(),
                'README.txt': CONCEPT_README.encode()}
    for item in sorted((directory/'tokenizer').iterdir()):
        payloads['tokenizer/'+item.name] = item.read_bytes()
    manifest = dict(format=1, core_contract=receipt['contract'], model_contract=state.get('model_contract'), checkpoint=name,
                    identity=receipt['identity'], kind=job['config']['routing_profile'],
                    files_sha256={key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()})
    temporary = exports/(uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for key, value in payloads.items():
                archive.writestr(key, value)
            archive.writestr('export-manifest.json', json.dumps(manifest, indent=2))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


HF_LOADER = '''"""ACAN concept language model: load and write with numpy alone (no PyTorch, no server).

    from acan_lm import load, answer
    model, tokenizer = load(".")
    print(answer(model, tokenizer, "Ciao! Chi sei?"))
"""
from pathlib import Path

from .net_text import NetText


def load(folder="."):
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer
    folder = Path(folder)
    model = NetText.from_arrays(load_file(str(folder / "concepts.safetensors")))
    return model, AutoTokenizer.from_pretrained(str(folder))


def stop_tokens(tokenizer):
    """Where an answer ends: end of text, end of the turn, or the start of a new turn."""
    ids = {tokenizer.eos_token_id}
    for marker in ("<|im_end|>", "<|im_start|>"):
        value = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(value, int) and value != tokenizer.unk_token_id:
            ids.add(value)
    return ids - {None}


def answer(model, tokenizer, question, max_new_tokens=128, temperature=0.7, seed=0):
    """An answer in the chat format the model was trained on."""
    prompt = f"<|im_start|>user\\n{question.strip()}<|im_end|>\\n<|im_start|>assistant\\n"
    produced = model.generate(tokenizer(prompt)["input_ids"], max_new_tokens, stop=stop_tokens(tokenizer),
                              temperature=temperature, seed=seed)
    return tokenizer.decode(produced, skip_special_tokens=True).strip()


def continue_text(model, tokenizer, text, max_new_tokens=128, temperature=0.7, seed=0):
    produced = model.generate(tokenizer(text)["input_ids"], max_new_tokens, temperature=temperature, seed=seed)
    return tokenizer.decode(produced, skip_special_tokens=True)
'''

CHAT_TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                 "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def queue_provenance(store, job):
    """The texts the model learned (the whole queue up to this one) with their sources and licences, when recorded."""
    series = job['config'].get('series') or {}
    members = [job]
    if series.get('id'):
        members = sorted((other for other in store.jobs(job['owner']) if (other['config'].get('series') or {}).get('id') == series['id']
                          and other['config']['series']['order'] <= series.get('order', 0)),
                         key=lambda other: other['config']['series']['order'])
    texts = []
    for member in members:
        corpus = store.corpus(member['config']['corpus_id'], job['owner'])
        recorded = store.directory('corpora', corpus['id'])/'dialogue-source.json'
        sources = json.loads(recorded.read_text()).get('sources', {}) if recorded.exists() else None
        texts.append(dict(name=corpus['name'], sources=sources))
    return texts


def weights_license(texts):
    """A licence for the weights compatible with the data: share-alike if a source asks for it; «other» when a text
    has no recorded source (its licence must be checked by hand)."""
    if any(text['sources'] is None for text in texts):
        return 'other'
    licences = ' '.join(source['license'] for text in texts for source in text['sources'].values())
    return 'cc-by-sa-4.0' if 'SA' in licences else 'cc-by-4.0'


def test_error(store, job, model, tokenizer_dir, limit=64):
    """The error on the test documents of the text (never read, never used to decide), overall and by kind of token."""
    import numpy as np
    from transformers import AutoTokenizer
    from .media import error_by_kind, token_kinds
    from .research_data import Blocks
    folder = store.directory('jobs', job['id'])
    blocks = Blocks(folder, job['config']['sequence_length'], job['config']['seed'], job['config']['validation_batches'])
    values, targets = [], []
    for index in blocks.evaluation_indices('test', limit):
        piece = np.asarray(blocks._block('test', index), np.int64)
        values.append(model.logp(piece, np.arange(1, len(piece))))
        targets.append(piece[1:])
    if not values:
        return None
    every = np.concatenate(values)
    kinds = token_kinds(AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True))
    return dict(nll=float(-every.mean()), tokens=int(len(every)), by_kind=error_by_kind(every, np.concatenate(targets), kinds))


def model_card(job, state, net_config, texts, licence, test):
    """README.md for the Hugging Face Hub: what the model is, how to use it, what it learned from, its measures and limits."""
    import math
    from .dialogue_data import SOURCES
    languages = sorted({language for text in texts for key in (text['sources'] or {})
                        for language in SOURCES.get(key, {}).get('languages', []) if len(language) == 2})
    datasets = sorted({source['page'].split('huggingface.co/datasets/')[1] for text in texts for source in (text['sources'] or {}).values()
                       if 'huggingface.co/datasets/' in source.get('page', '')})
    header = ['---', f'license: {licence}', 'library_name: acan', 'pipeline_tag: text-generation',
              'tags:', '- acan', '- concept-network', '- numpy']
    if languages:
        header += ['language:'] + [f'- {language}' for language in languages]
    if datasets:
        header += ['datasets:'] + [f'- {dataset}' for dataset in datasets]
    header.append('---')
    nll = state.get('validation_nll')
    lines = ['# ACAN concept language model', '',
             'A language model made of **concepts** (ACAN, Algebraic Concept Attention Network), not a transformer: every token '
             'and every position of the text belong to Gaussian concepts organised in layers; each layer reads the concepts of '
             'the layer above, and a memory of the document can copy what followed similar positions. It runs with **numpy '
             'alone**.', '',
             '## Network', '',
             f"- width {net_config['dim']}, {net_config['layers']} layers, {net_config['heads']} head{'s' if net_config['heads'] != 1 else ''}, "
             f"concepts per head per layer {net_config['concepts']}",
             f"- vocabulary {net_config.get('vocabulary', '?')} tokens (byte-level BPE learned on the training texts; images, sounds "
             'and videos are written with fixed tokens: 256 colours, 256 sound levels)',
             f"- trained on {state.get('tokens_trained', 0):,} tokens", '',
             '## Use', '', '```python', 'from huggingface_hub import snapshot_download',
             'import sys; folder = snapshot_download("<user>/<model>"); sys.path.insert(0, folder)',
             'from acan_lm import load, answer', 'model, tokenizer = load(folder)',
             'print(answer(model, tokenizer, "Ciao! Chi sei?"))', '```', '',
             'Requirements: `numpy`, `safetensors`, `transformers` (only for the tokenizer). The code of the model is in '
             '`acan_lm/` (MIT licence). The chat format is `<|im_start|>role\\n...<|im_end|>`.', '',
             '## Measures', '']
    if nll is not None:
        lines.append(f"- validation error {nll:.3f} nats per token (perplexity {math.exp(nll):,.1f})")
    if test:
        lines.append(f"- **test** error {test['nll']:.3f} nats per token (perplexity {math.exp(test['nll']):,.1f}) on {test['tokens']:,} "
                     'tokens never read and never used to decide anything')
        english = {'testo': 'text', 'segni del dialogo': 'chat markers', 'immagini e video': 'images and videos', 'audio': 'audio'}
        for kind, value in test['by_kind'].items():
            lines.append(f"  - {english.get(kind, kind)}: {value['nll']:.3f} (perplexity {math.exp(value['nll']):,.1f}; "
                         f"{value['share']:.0%} of the tokens)")
        if len(test['by_kind']) > 2:
            lines.append('- the overall figure mixes text with images, videos and audio, which are easier to predict: the error '
                         'on **text** is the one that tells the quality of the language')
    lines += ['', '## Training data', '']
    for text in texts:
        if text['sources'] is None:
            lines.append(f"- `{text['name']}`: uploaded by the author; **source and licence not recorded** - check before publishing.")
            continue
        lines.append(f"- `{text['name']}`:")
        for source in text['sources'].values():
            lines.append(f"  - [{source['title']}]({source['page']}) - {source['license']}")
    lines += ['', '## Limits', '',
              '- Small: most parameters are the concepts of the tokens; it is a research model, not an assistant.',
              '- It can repeat the question or copy from the prompt: its memory of the document learned to copy real text.',
              '- Images (32 x 32, 256 colours), sounds (8 kHz, 8 bit) and videos (16 x 16, 4 frames a second) use simple fixed codes.',
              '- It can reproduce pieces of its training text; it was not filtered for safety.', '']
    return '\n'.join(header) + '\n\n' + '\n'.join(lines)


def export_huggingface(store, job, name):
    """A folder for the Hugging Face Hub, zipped: weights, tokenizer, a clean configuration, the code to run the model
    with numpy alone and the model card with sources, licences, measures and limits."""
    from .concept_lm import load
    from .manager import snapshots
    if job['config'].get('architecture') != 'concepts':
        raise ValueError('La pubblicazione per Hugging Face è pronta solo per il modello a concetti.')
    if name not in {item['name'] for item in snapshots(store, job)}:
        raise ValueError('Checkpoint non disponibile: aggiorna la lista.')
    directory = store.directory('jobs', job['id'])
    snapshot = directory/'checkpoints'/name
    receipt = checkpoints.verify(snapshot)
    exports = directory/'exports'; exports.mkdir(exist_ok=True)
    destination = exports/(name+'-huggingface.zip')
    if destination.exists():
        return destination
    model, state = load(snapshot)
    net_config = dict(model.model.net.config, vocabulary=model.vocabulary)
    texts = queue_provenance(store, job)
    licence = weights_license(texts)
    test = test_error(store, job, model, directory/'tokenizer')
    config = dict(model_type='acan-concepts', format=1, vocab_size=model.vocabulary, eos_token_id=model.eos,
                  context_tokens=model.max_context, network={key: net_config[key] for key in ('dim', 'layers', 'heads', 'head_dim', 'concepts')
                                                               if key in net_config},
                  chat_format='<|im_start|>role\\n...<|im_end|>', tokens_trained=state.get('tokens_trained'),
                  validation_nll=state.get('validation_nll'), test=test, weights_license=licence)
    payloads = {'concepts.safetensors': (snapshot/'trainables.safetensors').read_bytes(),
                'config.json': json.dumps(config, indent=2).encode(),
                'README.md': model_card(job, state, net_config, texts, licence, test).encode()}
    for item in sorted((directory/'tokenizer').iterdir()):
        content = item.read_bytes()
        if item.name == 'tokenizer_config.json':                # the chat format, for apply_chat_template
            content = json.dumps(dict(json.loads(content), chat_template=CHAT_TEMPLATE), indent=2).encode()
        payloads[item.name] = content
    source = Path(__file__).resolve().parent
    payloads['acan_lm/__init__.py'] = HF_LOADER.encode()
    for module in ('concept_net.py', 'net_text.py'):
        payloads['acan_lm/'+module] = (source/module).read_bytes()
    payloads['acan_lm/LICENSE'] = b'MIT License - the code of the ACAN concept language model (acan-clustering).\n'
    manifest = dict(format=1, checkpoint=name, identity=receipt['identity'], weights_license=licence,
                    files_sha256={key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()})
    temporary = exports/(uuid.uuid4().hex+'.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for key, value in payloads.items():
                archive.writestr(key, value)
            archive.writestr('export-manifest.json', json.dumps(manifest, indent=2))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
