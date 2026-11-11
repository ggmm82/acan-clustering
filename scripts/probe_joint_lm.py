"""Matched NLL-only / NLL+CP mechanical smoke on a tiny local random Llama.

No pretrained quality claim; no downloads. Held-out blocks are fixed before training.
"""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import tempfile
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from acan.lm.research_model import load_model
from acan.lm.cp_objective import regularize,clear


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('reports/joint-lm.json'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    rng=torch.Generator().manual_seed(42)
    # Distinct random blocks, with an easy arithmetic next-token pattern.
    raw=torch.randint(0,30,(48,4),generator=rng)+2
    blocks=torch.stack((raw,(raw-1)%30+2),dim=-1).reshape(48,8)
    train,validation,test=blocks[:32],blocks[32:40],blocks[40:]
    assert len(set(map(tuple,blocks.tolist())))==len(blocks)
    rows=[]
    with tempfile.TemporaryDirectory() as directory:
        torch.manual_seed(7)
        base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64))
        base.save_pretrained(directory,safe_serialization=True)
        for fraction in (.5,1.):
            for weight in (0.,.1):
                torch.manual_seed(13)
                model,_=load_model(directory,dict(architecture='hybrid',graft_fraction=fraction,sequence_length=8,
                    slots=3,window=4,lora_rank=2,routing_profile='cp-joint-v1'))
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
                initial=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
                history=[]
                model.train()
                for step in range(16):
                    batch=train[2*step:2*step+2]
                    optimizer.zero_grad(set_to_none=True)
                    result=model(batch,labels=batch,use_cache=False)
                    loss,penalty=regularize(model,result.loss,weight)
                    loss.backward()
                    gradients=[p.grad for p in model.parameters() if p.grad is not None]
                    assert all(torch.isfinite(g).all() for g in gradients)
                    cp_gradients=[p.grad for name,p in model.named_parameters() if name.endswith('.G') and p.grad is not None]
                    history.append(dict(step=step+1,nll=float(result.loss.detach()),objective=float(loss.detach()),
                        cp_penalty=float(penalty.detach()),prototype_gradient_norm=float(sum(g.square().sum() for g in cp_gradients).sqrt())))
                    optimizer.step();clear(model)
                model.eval()
                with torch.no_grad():
                    validation_nll=float(model(validation,labels=validation,use_cache=False).loss)
                    output=model(test,labels=test,use_cache=False)
                    test_nll=float(output.loss)
                    accuracy=float((output.logits[:,:-1].argmax(-1)==test[:,1:]).float().mean())
                rows.append(dict(graft_fraction=fraction,cp_weight=weight,initial_parameter_sha256=initial,
                    validation_nll=validation_nll,held_out_nll=test_nll,held_out_next_token_accuracy=accuracy,history=history))
    report=dict(versions=dict(python=platform.python_version(),torch=version('torch'),transformers=version('transformers'),peft=version('peft')),scope='Mechanical smoke only: random local Llama, not pretrained language quality.',
        routing_profile='cp-joint-v1',seed=42,train_blocks=32,validation_blocks=8,test_blocks=8,sequence_length=8,steps=16,
        matched_initial_parameters=all(rows[i]['initial_parameter_sha256']==rows[i+1]['initial_parameter_sha256'] for i in (0,2)),arms=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='arms'}))
    for row in rows:print({key:value for key,value in row.items() if key!='history'})


if __name__=='__main__':
    main()
