"""A native concept language model: born from zero, every layer reads through concepts.

No pretrained weights and no adapters. The model is a decoder (embeddings, blocks of attention and
feed-forward, output layer tied to the embeddings) whose attention in every block is the concept
layer (acan/lm/concept_attention.py): the last ``window`` tokens exact, the older ones summarized
in at most ``slots`` concepts. Every weight is initialized at random from the project seed and
every weight learns.

The tokenizer is learned too, on an equal share of every text of the queue (byte-level BPE, so any
character can be written), with the tokens of the conversation format (``<|endoftext|>``,
``<|im_start|>``, ``<|im_end|>``) and those of images, sounds and videos (acan/lm/media.py). Learned on
Italian, English and code together, none of them pays the extra tokens of a tokenizer built for another.
"""
import hashlib
import json
from pathlib import Path
import shutil

SIZES = {
    'micro': dict(hidden=256, layers=4, heads=4, intermediate=1024, title='Micro · circa 12 milioni di parametri · CPU'),
    'small': dict(hidden=512, layers=8, heads=8, intermediate=2048, title='Piccolo · circa 50 milioni di parametri · CPU lenta o GPU'),
    'base': dict(hidden=768, layers=12, heads=12, intermediate=3072, title='Base · circa 138 milioni di parametri · GPU'),
}
SPECIAL = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
SAMPLE_BYTES = 200*1024*1024


def parameters(size, vocabulary=32000):
    """Parameters of a size (embeddings tied to the output layer)."""
    item = SIZES[size]
    block = 4*item['hidden']**2 + 3*item['hidden']*item['intermediate'] + 2*item['hidden']
    return vocabulary*item['hidden'] + item['layers']*block + item['hidden']


def _texts(path, limit=SAMPLE_BYTES):
    """Up to ``limit`` bytes of text of a corpus (JSONL ``text`` fields or plain text), media tokens left out."""
    import re
    read = 0
    with open(path, encoding='utf-8', errors='replace') as stream:
        for line in stream:
            read += len(line)
            if read > limit:
                break
            text = line
            if line.startswith('{'):
                try:
                    record = json.loads(line)
                    text = record.get('text', '') if isinstance(record, dict) else ''
                except ValueError:
                    pass
            text = re.sub(r'<\|[a-z/]{1,6}\d{0,3}\|>', ' ', text)      # pixels and samples are tokens of their own
            if text.strip():
                yield text


def _training_texts(path, name, limit):
    """Use the same document boundaries, deduplication and split as the token files."""
    from .research_data import document
    seen = set()
    with open(path, 'rb') as stream:
        while stream.tell() < limit:
            item = document(stream, name.lower().endswith('.jsonl'))
            if item is None:
                return
            text, role = item
            digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            if role == 'train' or (role is None and int(digest[:8], 16)%100 < 80):
                yield text


def train_tokenizer(corpora, directory, vocabulary=32000, training_names=None):
    """Byte-level BPE learned on an equal share of every corpus; saved so that ``AutoTokenizer`` loads it.

    ``corpora`` is a path or a list of paths: with Italian, English and code in the first texts of a
    queue, each gets its own frequent pieces. The conversation markers and the media tokens
    (acan/lm/media.py) are added whole.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from .media import TOKENS
    paths = [corpora] if isinstance(corpora, (str, Path)) else list(corpora)
    share = SAMPLE_BYTES//max(len(paths), 1)

    def texts():
        for i, path in enumerate(paths):
            yield from (_training_texts(path, training_names[i], share) if training_names else _texts(path, share))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocabulary, special_tokens=SPECIAL+TOKENS, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token='<|endoftext|>', pad_token='<|endoftext|>',
                                      additional_special_tokens=SPECIAL[1:]+TOKENS)
    directory = Path(directory)
    temporary = directory.with_name(directory.name+'.tmp')
    shutil.rmtree(temporary, ignore_errors=True)
    wrapped.save_pretrained(temporary)
    shutil.rmtree(directory, ignore_errors=True)
    temporary.rename(directory)
    return directory


def _hashes(directory):
    return {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in sorted(Path(directory).iterdir()) if item.is_file()}


def resolve_base(store, job, options):
    """The base of a project: the tokenizer of a native model, or the pretrained model of the others.

    trainer.resolve_base is part of the checkpoint contract and stays unchanged; callers use this.
    """
    if options.get('architecture') in ('native', 'concepts'):          # both learn their tokenizer from the texts
        return resolve(store, job, options)
    from .trainer import resolve_base as pretrained
    return pretrained(store, job, options)


def resolve(store, job, options):
    """The tokenizer of a native project: learned on its text, or the one of the previous item of its queue."""
    from .store import atomic_json
    directory = store.directory('jobs', job['id'])
    folder, recorded = directory/'tokenizer', directory/'base.json'
    if recorded.exists() and folder.exists():
        base = json.loads(recorded.read_text())
        if _hashes(folder) != base['files_sha256']:
            raise ValueError('Tokenizer del progetto modificato: ripristinare il backup.')
        return str(folder), base
    if options.get('control_of'):
        # The control of a native model reads the same tokens: it takes the tokenizer of its project.
        source = store.directory('jobs', options['control_of'])/'tokenizer'
        if not source.exists():
            raise ValueError('Il progetto da controllare non ha ancora un tokenizer: avvialo prima del controllo.')
        shutil.copytree(source, folder, dirs_exist_ok=True)
    elif options.get('warm_start'):
        from .research_trainer import warm_source
        source = warm_source(store, job).parent.parent/'tokenizer'
        if not source.exists():
            raise ValueError('Il testo precedente della coda non ha un tokenizer: ripristinare il backup.')
        shutil.copytree(source, folder, dirs_exist_ok=True)
    else:
        # Learned on every text of the queue known now, an equal share each (only its own text otherwise).
        series = options.get('series') or {}
        ids = [options['corpus_id']]
        if series.get('id'):
            members = sorted((other for other in store.jobs(job['owner']) if (other['config'].get('series') or {}).get('id') == series['id']),
                             key=lambda other: other['config']['series']['order'])
            ids = list(dict.fromkeys(other['config']['corpus_id'] for other in members)) or ids
        corpora = [store.corpus(identifier, job['owner'], ready=True) for identifier in ids]
        paths = [store.directory('corpora', item['id'])/'corpus' for item in corpora]
        training_only = options.get('routing_profile') == 'concept-table-v1' or options.get('context_refinement')
        names = [item['name'] for item in corpora] if training_only else None
        train_tokenizer(paths, folder, options.get('vocabulary', 32000), training_names=names)
    hashes = _hashes(folder)
    base = dict(repo='acan-native', revision='tokenizer-'+hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16],
                native=True, size=options['native_size'], files_sha256=hashes)
    atomic_json(recorded, base)
    return str(folder), base


def build(path, options, device='cpu', evaluate_base=None):
    """A model with random weights from the seed and concept attention in every block."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig
    from .full_attention import FullConceptAttention, FullConceptLlama
    from .research_model import shared_adapter
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    item = SIZES[options['native_size']]
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=item['hidden'], intermediate_size=item['intermediate'],
                         num_hidden_layers=item['layers'], num_attention_heads=item['heads'], num_key_value_heads=item['heads'],
                         max_position_embeddings=max(4096, options['sequence_length']+1), tie_word_embeddings=True,
                         bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                         rope_theta=10000., attention_bias=False, mlp_bias=False)
    config._attn_implementation = 'eager'
    state = torch.random.get_rng_state()
    torch.manual_seed(options['seed'])                          # the same seed gives the same model, also when reloading
    dense = options.get('native_attention') == 'dense'          # the control: the same model with ordinary attention
    try:
        if dense:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM(config)
        else:
            model = FullConceptLlama(config)
            for index, layer in enumerate(model.model.layers):
                layer.self_attn = FullConceptAttention(config, index, slots=options['slots'], window=options['window'])
    finally:
        torch.random.set_rng_state(state)
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = model.to(device=device, dtype=dtype)
    report = dict(profile='acan-native-dense-control-v1' if dense else 'acan-native-v1',
                  replaced_layers=[] if dense else list(range(item['layers'])), total_layers=item['layers'],
                  slots=0 if dense else options['slots'], window=0 if dense else options['window'], fraction=0. if dense else 1.,
                  size=options['native_size'], vocabulary=len(tokenizer), baseline_nll=None)
    if not dense:
        model, report = shared_adapter(model, report, options)
    report['baseline_nll'] = evaluate_base(model) if evaluate_base else None
    return model, report
