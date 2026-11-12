"""The complete timeline remains reachable without returning it on every poll."""
import json
import tempfile
import unittest
from acan.lm.service import create_app
from acan.lm.store import owner_key


class HistoryNavigationTests(unittest.TestCase):
    def test_overview_and_exact_pages_reach_all_steps_and_enforce_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            app=create_app(directory,token='x'*32)
            store=app.store;owner=owner_key('a'*64)
            corpus=store.create_corpus(owner,'fixture.txt',6)
            store.append_corpus(corpus['id'],owner,0,b'hello ')
            store.finish_corpus(corpus['id'],owner)
            job=store.create_job(owner,dict(corpus_id=corpus['id']))
            with store.db() as db:
                db.executemany('INSERT INTO metrics VALUES(?,?,?)',[(job['id'],i,json.dumps(dict(step=i,training_nll=10/(i+1)))) for i in range(1507)])
            overview=store.history(job['id'],owner,overview=True)
            self.assertEqual(overview[0]['step'],0)
            self.assertEqual(overview[-1]['step'],1506)
            self.assertLessEqual(len(overview),1001)
            sparse=store.create_job(owner,dict(corpus_id=corpus['id']))
            with store.db() as db:
                db.executemany('INSERT INTO metrics VALUES(?,?,?)',[(sparse['id'],i*10000,json.dumps(dict(step=i*10000))) for i in range(17)])
            self.assertEqual([row['step'] for row in store.history(sparse['id'],owner,overview=True)],
                             [i*10000 for i in range(17)])
            client=app.test_client();headers={'Authorization':'Bearer '+'x'*32,'X-ACAN-Workspace':'a'*64}
            steps=[];start=0
            while True:
                reply=client.get(f"/jobs/{job['id']}/history?start={start}&limit=500",headers=headers)
                self.assertEqual(reply.status_code,200)
                data=reply.get_json()
                if not data['rows']:break
                steps.extend(row['step'] for row in data['rows']);start=data['next_step']
            self.assertEqual(steps,list(range(1507)))
            window=client.get(f"/jobs/{job['id']}/history?start=500&end=999&limit=500",headers=headers).get_json()
            self.assertEqual([r['step'] for r in window['rows']],list(range(500,1000)))
            other=dict(headers,**{'X-ACAN-Workspace':'b'*64})
            self.assertEqual(client.get(f"/jobs/{job['id']}/history",headers=other).status_code,404)
            for query in ['start=-1','start=x','limit=0','limit=3000','start=100&end=20']:
                self.assertEqual(client.get(f"/jobs/{job['id']}/history?{query}",headers=headers).status_code,400)
