"""Reproduce the joint model's Letter evaluation without publishing input records.

Usage: OPENBLAS_NUM_THREADS=1 python -m scripts.evaluate_joint_cp --letter letter.arff
"""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from acan import GraphClusterModel
from acan.vis.training import train_native, graph_stats
from acan.vis.inference import infer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--letter',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('reports/joint-letter.json'))
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--depth',type=int,default=1)
    args=parser.parse_args()
    raw,_=arff.loadarff(args.letter)
    data=pd.DataFrame(raw)
    for col in data:
        if data[col].dtype==object:
            data[col]=data[col].map(lambda v:v.decode() if isinstance(v,bytes) else v)
    report=dict(versions=dict(python=platform.python_version(),numpy=version('numpy'),pandas=version('pandas'),scipy=version('scipy'),sklearn=version('scikit-learn')),source_sha256=hashlib.sha256(args.letter.read_bytes()).hexdigest(),seed=42,
        selection='best full-training joint objective, never held-out accuracy',
        note='Single prescribed split, exploratory replication; no hyperparameter search. Not evidence about LLM quality.',runs=[])
    options=dict(lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300,max_depth=args.depth,
        max_epochs=args.epochs,min_samples_split=12,similarity_threshold=.92,routing_profile='cp-joint-v1')
    with threadpool_limits(limits=1):
        for sample in (None,10000):
            frame=data if sample is None else data.sample(sample,random_state=42)
            train,test=train_test_split(frame,test_size=.2,stratify=frame['class'],random_state=42)
            started=time.monotonic();graph=train_native(train,['class'],options)
            seconds=time.monotonic()-started
            payload=graph.to_json();root=graph.clusterModel[graph.rootId]
            predicted=infer(payload,test)['class']
            restored=GraphClusterModel.loadModel(payload)
            before=restored.to_json()
            probe=test.head(20)
            roundtrip_identical=np.array_equal(infer(payload,probe)['class'],infer(restored.to_json(),probe)['class'])
            record=dict(sample=sample,train_rows=len(train),held_out_rows=len(test),options=options,
                accuracy=accuracy_score(test['class'],predicted),balanced_accuracy=balanced_accuracy_score(test['class'],predicted),
                training_seconds=seconds,stats=graph_stats(graph),
                training_nll=root.routing_parameters['training_nll'],training_objective=root.routing_parameters['training_objective'],
                roundtrip_predictions_identical=roundtrip_identical,inference_does_not_mutate_model=before==restored.to_json(),
                history=root.history)
            report['runs'].append(record)
            print({key:value for key,value in record.items() if key not in ('options','history')},flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
