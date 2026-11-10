"""Resumable document-level deduplication and disjoint train/validation/test tokens."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import numpy as np
from .checkpoints import fingerprint
from .data import PreparationPaused
from .store import atomic_json, digest_file

ROLES = ('train', 'validation', 'test')


def document(stream, jsonl):
    if jsonl:
        line = stream.readline(16*1024*1024+1)
        if not line:
            return None
        if len(line)>16*1024*1024:
            raise ValueError('Documento oltre 16 MiB.')
        if not line.strip():
            return '', None
        record = json.loads(line.decode('utf-8'))
        if not isinstance(record, dict) or not isinstance(record.get('text'), str):
            raise ValueError('Ogni record JSONL richiede un campo text stringa.')
        role = record.get('split')
        if role is not None and role not in ROLES:
            raise ValueError('Split JSONL non valido: train, validation o test.')
        return record['text'], role
    lines, size = [], 0
    while True:
        line = stream.readline(16*1024*1024+1)
        if not line:
            return ('\n'.join(lines), None) if lines else None
        size += len(line)
        if size>16*1024*1024:
            raise ValueError('Paragrafo oltre 16 MiB: suddividere il testo.')
        if not line.strip():
            if lines:
                return '\n'.join(lines), None
            continue
        lines.append(line.decode('utf-8').rstrip())


def prepare(source, name, tokenizer, directory, identity, should_pause=lambda:False):
    directory = Path(directory)
    metadata = directory/'document-tokens.json'
    marker = fingerprint(dict(identity=identity, protocol='document-splits-v1'))
    progress = json.loads(metadata.read_text()) if metadata.exists() else dict(
        identity=marker, source_offset=0, tokens={role:0 for role in ROLES},
        documents={role:0 for role in ROLES}, duplicates=0, complete=False)
    if progress['identity'] != marker:
        raise ValueError('Corpus o tokenizer diversi dalla preparazione salvata.')
    if progress['complete']:
        for role in ROLES:
            path=directory/(role+'.bin')
            if path.stat().st_size!=progress['tokens'][role]*4 or digest_file(path)!=progress['sha256'][role]:
                raise ValueError('Archivio token modificato: '+role)
        return progress
    seen = sqlite3.connect(directory/'document-index.sqlite')
    seen.execute('CREATE TABLE IF NOT EXISTS documents(hash TEXT PRIMARY KEY, source_offset INTEGER NOT NULL)')
    # Entries beyond the durable byte offset belong to a partial interrupted write.
    seen.execute('DELETE FROM documents WHERE source_offset>=?', (progress['source_offset'],));seen.commit()
    files = {role:(directory/(role+'.bin')).open('r+b' if (directory/(role+'.bin')).exists() else 'w+b') for role in ROLES}
    def commit():
        for role, stream in files.items():
            stream.flush();os.fsync(stream.fileno())
        seen.commit();atomic_json(metadata, progress)
    try:
        for role,stream in files.items():
            stream.truncate(progress['tokens'][role]*4);stream.seek(0,os.SEEK_END)
        with open(source,'rb') as stream:
            stream.seek(progress['source_offset']);count=0
            while True:
                if should_pause():
                    commit();raise PreparationPaused()
                offset=stream.tell();item=document(stream,name.lower().endswith('.jsonl'))
                if item is None:
                    break
                text,role=item
                normalized=' '.join(text.split())
                if normalized:
                    digest=hashlib.sha256(normalized.encode()).hexdigest()
                    inserted=seen.execute('INSERT OR IGNORE INTO documents VALUES(?,?)',(digest,offset)).rowcount
                    if inserted:
                        if role is None:
                            bucket=int(digest[:8],16)%100
                            role='train' if bucket<80 else 'validation' if bucket<90 else 'test'
                        ids=tokenizer.encode(text,add_special_tokens=False)
                        if tokenizer.eos_token_id is not None:
                            ids.append(tokenizer.eos_token_id)
                        files[role].write(np.asarray(ids,dtype='<u4').tobytes())
                        progress['tokens'][role]+=len(ids);progress['documents'][role]+=1
                    else:
                        progress['duplicates']+=1
                progress['source_offset']=stream.tell();count+=1
                if count%100==0:
                    commit()
        commit()
        progress['sha256']={role:digest_file(directory/(role+'.bin')) for role in ROLES}
        progress['complete']=True;atomic_json(metadata,progress)
        return progress
    finally:
        for stream in files.values():
            stream.close()
        seen.close()


class Blocks:
    def __init__(self,directory,length,seed=42,max_validation_batches=8):
        self.length,self.seed=length,seed
        self.tokens={}
        self.counts={}
        for role in ROLES:
            path=Path(directory)/(role+'.bin')
            if path.stat().st_size<(length+1)*4:
                raise ValueError(f'Split {role} troppo piccolo: servono più documenti o blocchi più corti.')
            self.tokens[role]=np.memmap(path,mode='r',dtype='<u4')
            self.counts[role]=len(self.tokens[role])//(length+1)
        self.train_count=self.counts['train'];self.cached_epoch=None
        self.max_validation_batches=max_validation_batches

    def _block(self,role,index):
        start=index*(self.length+1)
        return np.array(self.tokens[role][start:start+self.length+1],dtype=np.int64)

    def batch(self,batch_size,cursor):
        import torch
        items=[]
        for _ in range(batch_size):
            if cursor['offset']>=self.train_count:
                cursor['epoch']+=1;cursor['offset']=0
            if self.cached_epoch!=cursor['epoch']:
                generator=torch.Generator().manual_seed(self.seed+cursor['epoch'])
                self.order=torch.randperm(self.train_count,generator=generator).tolist();self.cached_epoch=cursor['epoch']
            items.append(self._block('train',self.order[cursor['offset']]))
            cursor['offset']+=1
        return torch.from_numpy(np.stack(items))

    def evaluation_indices(self,role,limit):
        # Spread the fixed evaluation sample across the corpus, not just its
        # opening article/table of contents. Same positions for both models.
        return np.linspace(0,self.counts[role]-1,num=min(limit,self.counts[role]),dtype=int).tolist()

    def evaluation(self,role,limit):
        import torch
        for index in self.evaluation_indices(role,limit):
            yield torch.from_numpy(self._block(role,index))[None]

    def validation(self):
        return self.evaluation('validation',self.max_validation_batches)
