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
