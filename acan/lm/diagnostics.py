"""Own-project debugging, with explicit fields and secret-redacted log tails."""
import json
import os
from pathlib import Path
import re
from importlib.metadata import version
from .checkpoints import contract
from .manager import snapshots


def redact(text,secret=''):
    if secret:
        text=text.replace(secret,'[redacted]')
    text=re.sub(r'(?i)(bearer\s+)[^\s\"\']+',r'\1[redacted]',text)
    text=re.sub(r'hf_[A-Za-z0-9]{12,}','[redacted]',text)
    text=re.sub(r'(?i)((?:token|secret|authorization|api_key)[=:\s]+)[^\s,&\"\']+',r'\1[redacted]',text)
    return text


def report(store,job,secret=''):
    directory=store.directory('jobs',job['id'])
    log=directory/'worker.log';tail=''
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-16000));tail=stream.read().decode('utf-8',errors='replace')
    memory={}
    for name in ('memory.current','memory.max','memory.events'):
        path=Path('/sys/fs/cgroup')/name
        if path.exists():memory[name]=path.read_text().strip()
    progress=job['progress']
    train_tokens=progress.get('dataset',{}).get('tokens',{}).get('train')
    explanation=[]
    best=[item for item in snapshots(store,job) if item.get('best')]
    if best and progress.get('validation_nll') is not None and progress['validation_nll']>best[0]['validation_nll']:
        explanation.append('La validazione attuale è peggiore del miglior checkpoint conservato. Non implica da sola un errore dell’architettura.')
    if train_tokens and progress.get('tokens_trained',0)>train_tokens:
        explanation.append('Il training ha già ripetuto materiale del corpus: controlla sovra-adattamento e diversità dei dati.')
    if not job['config'].get('protocol'):
        explanation.append('Progetto legacy: il minimo storico eliminato non è recuperabile; il migliore conservato è tra i salvataggi ancora disponibili.  split originale e nessun arresto anticipato automatico. Il nuovo protocollo si applica ai nuovi progetti.')
    return dict(status=job['status'],job=job['id'],config=job['config'],progress=progress,error=job['error'],
                core_contract=contract(),package_version=version('acan-clustering'),build_revision=os.environ.get('ACAN_BUILD_REVISION','development'),
                snapshots=snapshots(store,job),diagnosis=explanation,
                corpus_passes=progress.get('tokens_trained',0)/train_tokens if train_tokens else None,
                worker_cgroup_memory=memory,log_tail=redact(tail,secret),
                scope='Own project only; no raw corpus, workspace key, service token or environment dump.')
