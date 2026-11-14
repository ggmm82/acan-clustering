"""A native concept language model: born from zero, every layer reads through concepts.

No pretrained weights and no adapters. The model is a decoder (embeddings, blocks of attention and
feed-forward, output layer tied to the embeddings) whose attention in every block is the concept
layer (acan/lm/concept_attention.py): the last ``window`` tokens exact, the older ones summarized
in at most ``slots`` concepts. Every weight is initialized at random from the project seed and
every weight learns.

The tokenizer is learned too, on the first text of the training (byte-level BPE, so any character
can be written), with the three tokens of the conversation format: ``<|endoftext|>``,
``<|im_start|>`` and ``<|im_end|>``. It is learned on Italian and English together, so Italian
does not pay the extra tokens of a tokenizer built for English.
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
    """Up to ``limit`` bytes of text of a corpus (JSONL ``text`` fields or plain text)."""
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
            if text.strip():
                yield text


def train_tokenizer(corpus, directory, vocabulary=32000):
    """Byte-level BPE learned on the corpus; saved so that ``AutoTokenizer`` loads it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocabulary, special_tokens=SPECIAL, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(_texts(corpus), trainer=trainer)
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token='<|endoftext|>', pad_token='<|endoftext|>',
                                      additional_special_tokens=SPECIAL[1:])
    directory = Path(directory)
    temporary = directory.with_name(directory.name+'.tmp')
    shutil.rmtree(temporary, ignore_errors=True)
    wrapped.save_pretrained(temporary)
    shutil.rmtree(directory, ignore_errors=True)
    temporary.rename(directory)
    return directory


def _hashes(directory):
    return {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in sorted(Path(directory).iterdir()) if item.is_file()}


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
    if options.get('warm_start'):
        from .research_trainer import warm_source
        source = warm_source(store, job).parent.parent/'tokenizer'
        if not source.exists():
            raise ValueError('Il testo precedente della coda non ha un tokenizer: ripristinare il backup.')
        shutil.copytree(source, folder, dirs_exist_ok=True)
    else:
        corpus = store.corpus(options['corpus_id'], job['owner'], ready=True)
        train_tokenizer(store.directory('corpora', corpus['id'])/'corpus', folder, options.get('vocabulary', 32000))
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
    try:
        model = FullConceptLlama(config)
        for index, layer in enumerate(model.model.layers):
            layer.self_attn = FullConceptAttention(config, index, slots=options['slots'], window=options['window'])
    finally:
        torch.random.set_rng_state(state)
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = model.to(device=device, dtype=dtype)
    report = dict(profile='acan-native-v1', replaced_layers=list(range(item['layers'])), total_layers=item['layers'],
                  slots=options['slots'], window=options['window'], fraction=1., size=options['native_size'],
                  vocabulary=len(tokenizer), baseline_nll=None)
    model, report = shared_adapter(model, report, options)
    report['baseline_nll'] = evaluate_base(model) if evaluate_base else None
    return model, report
