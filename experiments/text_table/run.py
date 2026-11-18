"""Prova riproducibile del primo passo richiesto, separata dal prodotto.

python -m experiments.text_table.run --data /path/to/parquets --output reports/text-table.json
Il tokenizer si impara solo sul train; validation sceglie kappa, test misura.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd
from tokenizers import Tokenizer,models,pre_tokenizers,trainers,decoders
from threadpoolctl import threadpool_limits
from acan.concept_algebra import KAPPA_GRID
from acan.lm.concept_lm import ConceptLM
from .model import semantic_vectors,VectorTable


def blocks(ids,count=64,length=512,seed=42):
    # Non-overlapping blocks, selected once and saved as exact token offsets.
    available=np.arange(len(ids)//length)
    if len(available)<count:raise ValueError('Testo insufficiente per il protocollo.')
    starts=np.sort(np.random.default_rng(seed).choice(available,count,replace=False))*length
    return [ids[start:start+length] for start in starts],starts


def evaluate(model,pieces,vector=False):
    losses=[]
    for ids in pieces:
        positions=np.arange(1,len(ids))
        if vector:loss=-np.log(np.maximum(model.true_probability(ids,positions),1e-300))
        else:loss=-model.logp(ids,positions)
        losses.append(loss)
    matrix=np.stack(losses)
    return dict(nll=float(matrix.mean()),positions=int(matrix.size),
                bands={f'{a}-{b}':float(matrix[:,max(1,a)-1:b-1].mean()) for a,b in [(0,32),(32,128),(128,256),(256,512)]})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rows',type=int,default=200000);p.add_argument('--dimensions',type=int,default=32)
    p.add_argument('--context',type=int,default=8);p.add_argument('--concepts',type=int,default=256)
    p.add_argument('--judged-rows',type=int,default=512);p.add_argument('--refine-rows',type=int,default=20000)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(seed=0,options={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},runs=[],source_sha256={})
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    texts={}
    for name in ['train','validation','test']:
        file=a.data/(name+'.parquet');report['source_sha256'][name]=hashlib.sha256(file.read_bytes()).hexdigest()
        texts[name]=[str(t) for t in pd.read_parquet(file)['text'] if str(t).strip()]
    tokenizer_path=a.data/'bpe-8000.json'
    # Cache identity records the training source; never reuse a tokenizer from another text.
    identity=a.data/'bpe-source.sha256'
    if tokenizer_path.exists() and identity.exists() and identity.read_text()==report['source_sha256']['train']:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'));tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder=decoders.ByteLevel()
        tokenizer.train_from_iterator(texts['train'],trainers.BpeTrainer(vocab_size=8000,special_tokens=['[UNK]','[EOS]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tokenizer.save(str(tokenizer_path));identity.write_text(report['source_sha256']['train'])
    eos=tokenizer.token_to_id('[EOS]');arrays={}
    for name,lines in texts.items():
        encoded=tokenizer.encode_batch(lines)
        arrays[name]=np.concatenate([np.asarray(piece.ids+[eos],np.int32) for piece in encoded])
    report['tokenizer_sha256']=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    report['tokens']={name:len(ids) for name,ids in arrays.items()}
    v=tokenizer.get_vocab_size();train=arrays['train']
    valid,valid_starts=blocks(arrays['validation']);test,test_starts=blocks(arrays['test'])
    report.update(validation_offsets=valid_starts.tolist(),test_offsets=test_starts.tolist(),
        protocol='64 non-overlapping blocks512, seed42; position0 excluded because no prefix; validation first32 selects, last32 checks; test64 fixed',
        historical_comparison='Historical tokenizer and offsets were not supplied; old neural numbers are context, not a matched rerun.')
    save();print(json.dumps(dict(phase='prepared',tokens=report['tokens'])),flush=True)
    with threadpool_limits(limits=2):
        for scope,ids in [('same_rows',train[:a.rows+a.context]),('full_train',train)]:
            start=time.monotonic();baseline=ConceptLM(v,depth=6,document=4)
            for offset in range(0,len(ids),32768):baseline.add(ids[offset:offset+32768])
            baseline.settle();baseline.judge(valid[:32])
            record=dict(kind='exact_counter_control',scope=scope,train_tokens=len(ids),seconds=time.monotonic()-start,
                        validation=evaluate(baseline,valid[32:]),test=evaluate(baseline,test),bytes=baseline.nbytes())
            report['runs'].append(record);save();print(json.dumps(record),flush=True)
        start=time.monotonic();vectors,singular=semantic_vectors(train,v,a.dimensions)
        report['vectors']=dict(source='full training only',dimensions=a.dimensions,singular_values=singular.tolist(),
            sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),seconds=time.monotonic()-start)
        model=VectorTable(vectors,a.context,max_concepts=a.concepts,judged_rows=a.judged_rows,
            refine_rows=a.refine_rows,chunk_rows=2048,covariance=False,workers=1)
        progress=[]
        def tell(row):
            if not progress or row['phase']!=progress[-1]['phase'] or row.get('rows',0)-progress[-1].get('rows',0)>=20000:
                progress.append(row);print(json.dumps(row),flush=True)
        model.model.report=tell;start=time.monotonic();model.fit(train,rows=a.rows)
        record=dict(kind='ConceptModel_vector_rows',train_rows=model.model.rows,seconds=time.monotonic()-start,
            concepts=model.model.concepts,levels=[len(level['statistics']['C']) for level in model.model.levels],
            sparse_pairs=model.counts.nnz,sparse_density=float(model.counts.nnz/np.prod(model.counts.shape)),
            target_judgment='numeric vector density in the unchanged ConceptModel; exact-token judgment is not yet used for selecting the merge path',
            training_merge='unchanged reader radius excludes target columns',
            progress=progress)
        report['runs'].append(record);save()
        scores=[]
        for kappa in KAPPA_GRID:
            model.kappa=kappa;score=evaluate(model,valid[:32],True)['nll'];scores.append((score,kappa))
        model.kappa=min(scores)[1]
        record.update(kappa=model.kappa,validation=evaluate(model,valid[32:],True),test=evaluate(model,test,True),
            kappa_search=scores,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report['status']='measured';save();print(json.dumps({k:value for k,value in record.items() if k!='progress'}),flush=True)

if __name__=='__main__':main()
