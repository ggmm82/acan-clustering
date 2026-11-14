"""What a model in training answers to the same questions, at every save.

A long training is followed with two signals: the error on the fixed reference text (a number)
and the answers themselves (what a person reads). The answers are generated greedily, so two saves
differ only because the model changed. The training does not stop for this: it takes a few
seconds per question at every save.
"""
import json
import time
from pathlib import Path

QUESTIONS = (
    'Ciao! Chi sei?',
    'Spiegami in due frasi che cos\'è la fotosintesi.',
    'Qual è la capitale della Francia?',
    'Hello! What can you do?',
    'Give me three tips to sleep better.',
)
NEW_TOKENS = 64


def chat_prompt(question):
    """A question in the chat format the conversation data is written in (acan/lm/dialogue_data.py)."""
    return f"<|im_start|>user\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n"


def stop_tokens(tokenizer):
    ids = [tokenizer.eos_token_id]
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if isinstance(end, int) and end != tokenizer.unk_token_id:
        ids.append(end)
    return [value for value in ids if value is not None]


def answer(model, tokenizer, device, question, new_tokens=NEW_TOKENS):
    import torch
    from .grafting import reset_streams
    inputs = tokenizer(chat_prompt(question), return_tensors='pt')['input_ids'].to(device)
    reset_streams(model)
    with torch.no_grad():
        output = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=new_tokens, do_sample=False,
                                num_beams=1, use_cache=True, eos_token_id=stop_tokens(tokenizer), pad_token_id=tokenizer.eos_token_id)
    reset_streams(model)
    return tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip()


def record_answers(directory, model, tokenizer, device, state, questions=None):
    """Append the answers of this save to ``answers.jsonl`` of the project."""
    was_training = model.training
    model.eval()
    started = time.monotonic()
    try:
        answers = [dict(question=question, answer=answer(model, tokenizer, device, question)) for question in (questions or QUESTIONS)]
    finally:
        if was_training:
            model.train()
    entry = dict(step=state['step'], tokens_trained=state.get('tokens_trained'), validation_nll=state.get('validation_nll'),
                 seconds=time.monotonic()-started, answers=answers)
    with (Path(directory)/'answers.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False)+'\n')
    return entry


def read_answers(directory, limit=50):
    """The latest ``limit`` saves, oldest first."""
    path = Path(directory)/'answers.jsonl'
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()[-limit:]
    return [json.loads(line) for line in lines if line.strip()]
