"""Explicit matched-data comparison of legacy and paper routing, not full-paper certification."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from acan.vis.datasets import clean_frame,read_arff,target_and_filters,split_for_training,check_budget
from acan.vis.training import train_native,graph_stats
from acan.vis.evaluation import evaluate


def compare(paths, output, seeds=(7,21,42,84,123), baselines=False):
    options=dict(max_depth=1,max_epochs=100,min_samples_split=12,similarity_threshold=.92,
                 lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300)
    results=[]
    with threadpool_limits(limits=1):
        for path in paths:
            path=Path(path)
            if path.suffix.casefold()=='.arff':frame=read_arff(path.read_bytes())
            elif path.suffix.casefold()=='.csv':frame=clean_frame(pd.read_csv(path))
            elif path.suffix.casefold()=='.xlsx':frame=clean_frame(pd.read_excel(path,engine='openpyxl'))
            else:raise ValueError('Formato non supportato: '+path.suffix)
            labels,filters=target_and_filters(frame)
            for seed in seeds:
                train,test,info=split_for_training(frame,labels,filters,seed=seed)
                size=check_budget(train)
                unseen={target:int((~test[target].isin(train[target])).sum()) for target in labels}
                for profile in ('legacy-v8','paper-routing-v1'):
                    started=time.perf_counter();graph=train_native(train,labels,dict(options,routing_profile=profile))
                    seconds=time.perf_counter()-started
                    # Modern baselines are run once, on seed 42, only with --baselines.
                    report=evaluate(graph.to_json(),test,train,baselines=baselines and seed==42 and profile=='paper-routing-v1')
                    result=dict(dataset=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=int(seed),profile=profile,
                                targets=labels,excluded_columns=filters,stratified=info['stratified'],train_rows=len(train),validation_rows=len(test),
                                unseen_target_rows=unseen,encoding=size,training_seconds=seconds,graph=graph_stats(graph),
                                scores=report['scores'],timings=report['timings'],diagnostics=report.get('diagnostics',[]))
                    result['majority']={}
                    for target in labels:
                        truth=test[target].astype(str);prediction=np.repeat(train[target].astype(str).mode().iloc[0],len(test))
                        result['majority'][target]=dict(Accuracy=accuracy_score(truth,prediction),Balanced_accuracy=balanced_accuracy_score(truth,prediction),F1_macro=f1_score(truth,prediction,average='macro',zero_division=0))
                    results.append(result)
                    output.parent.mkdir(parents=True,exist_ok=True)
                    output.write_text(json.dumps(dict(protocol=dict(options=options,seeds=list(seeds),
                        description='Same typed data, exclusions, split and optimizer budget. Seed 42 reproduces GUI split membership. No tuning or score-based model selection. Multiple holdouts overlap; these are exploratory repeats, not independent confidence intervals.',
                        interpretation='Only routing is corrected: fixed omega, legacy relevance and CP remain. LLM has a different task and is not evaluated on a fabricated ARFF-to-text corpus.',
                        baselines_requested=baselines,baseline_seed=42),results=results),indent=2)+'\n')
                    scores={r['Metrica']:round(r['Valore'],4) for r in report['scores'] if r['Modello']=='ACAN attuale'}
                    print(json.dumps(dict(dataset=path.name,seed=seed,profile=profile,scores=scores,groups=result['graph']['groups'])),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,21,42,84,123])
    parser.add_argument('--baselines',action='store_true')
    args=parser.parse_args()
    if not 1<=len(args.seeds)<=5 or any(not 0<=seed<2**31 for seed in args.seeds):
        parser.error('Specify one to five seeds between zero and 2^31-1.')
    compare(args.files,args.output,args.seeds,args.baselines)


if __name__=='__main__':main()
