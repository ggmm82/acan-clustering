"""Local UI fixture: no model, download, training or remote service is used."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from urllib.parse import urlparse,parse_qs
from flask import jsonify
from acan.vis.render import app
from acan.vis import generative

calls=[]
def view(number):
    return dict(id=f'job-{number}',status='paused',error=None,checkpoint=None,snapshots=[],
        config=dict(name=f'Project {number}',total_steps=2000,protocol='controlled-v1'),
        progress=dict(step=1500),history_view='full_overview',
        history=[dict(step=i,training_nll=number+i/1000) for i in range(1501)])

def backend(workspace,path,*args,**kwargs):
    calls.append(path);url=urlparse(path);query=parse_qs(url.query)
    if url.path=='/capabilities':return dict(cuda=False)
    if url.path=='/public-corpora':return {}
    if url.path=='/state':
        selected=query.get('selected',[''])[0];jobs=[]
        for n in range(1,4):
            job=view(n)
            if selected!=job['id']:job['history']=[]
            jobs.append(job)
        return dict(jobs=jobs,corpora=[],busy=False,detail_job=selected)
    if url.path.endswith('/view'):return view(int(url.path.split('/')[2].split('-')[1]))
    if url.path.endswith('/history'):
        number=int(url.path.split('/')[2].split('-')[1]);start=int(query.get('start',['0'])[0])
        return dict(rows=view(number)['history'][start:start+500])
    raise ValueError('Unexpected fixture request: '+path)

generative.backend=backend
@app.server.get('/fixture/calls')
def requested_paths():return jsonify(calls)

if __name__=='__main__':app.server.run(host='127.0.0.1',port=int(sys.argv[1]),threaded=True)
