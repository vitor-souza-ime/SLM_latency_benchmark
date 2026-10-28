import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark.run import normalize_mc, stream_ollama, stream_openrouter, load_tasks
from benchmark.catalog import matched_providers

class FakeResponse:
    def __init__(self,lines,headers=None):
        self.lines=lines
        self.status_code=200
        self.headers=headers or {}
        self.ok=True
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def raise_for_status(self):return None
    def iter_lines(self,**kwargs):yield from self.lines

class FakeSession:
    def __init__(self,response):
        self.response=response
        self.payload=None
    def post(self,url,**kwargs):
        self.payload=kwargs['json']
        return self.response

class FakeCatalogResponse:
    def __init__(self,endpoints):self.endpoints=endpoints
    def raise_for_status(self):return None
    def json(self):return {'data':{'endpoints':self.endpoints}}

class FakeCatalogSession:
    def __init__(self,endpoints):self.endpoints=endpoints;self.url=''
    def get(self,url,**kwargs):self.url=url;return FakeCatalogResponse(self.endpoints)

class TestProtocol(unittest.TestCase):
    def test_four_models(self):
        models=json.loads((ROOT/'models.json').read_text(encoding='utf-8'))
        self.assertEqual(len(models),4)
        self.assertEqual(set(m['name'].split()[0] for m in models),{'Llama','Ministral','Qwen3'})
        self.assertTrue(models[1]['openrouter'].endswith(':free'))
    def test_all_english_key_mapping(self):
        qa=json.loads((ROOT/'data/questions.json').read_text(encoding='utf-8'))
        gen=json.loads((ROOT/'data/generation_prompts.json').read_text(encoding='utf-8'))
        self.assertEqual(len(qa),60)
        self.assertEqual(len(gen),12)
        self.assertTrue(all(item['language']=='en' for item in qa+gen))
        for q in qa:
            self.assertEqual(q['answer_text'],q['options'][ord(q['expected'])-65])
            self.assertTrue(q['prompt'].startswith('Choose the correct answer.'))
        for g in gen:
            self.assertIn('Use English.',g['prompt'])
    def test_compact_profile(self):
        tasks=load_tasks(profile='compact')
        self.assertEqual(len(tasks),12)
        self.assertEqual(sum(t['type']=='accuracy' for t in tasks),6)
        self.assertEqual(sum(t['type']=='performance' for t in tasks),6)
        self.assertEqual(sum(t['language']=='en' for t in tasks),12)
        self.assertEqual({t['category'] for t in tasks if t['type']=='performance'}, {'short','medium','long'})
    def test_full_profile(self):self.assertEqual(len(load_tasks(profile='full')),72)
    def test_strict_answer(self):
        self.assertEqual(normalize_mc('a.'),'A')
        self.assertEqual(normalize_mc(' B) '),'B')
        self.assertEqual(normalize_mc('Answer: B'),'')
    def test_ollama_stream(self):
        frames=[{'message':{'content':'B','thinking':''},'done':False},
                {'message':{'content':''},'done':True,'eval_count':4,'prompt_eval_count':20,
                 'eval_duration':400000000,'total_duration':700000000}]
        sess=FakeSession(FakeResponse([json.dumps(x).encode() for x in frames]))
        result=stream_ollama(sess,'http://local','a:8b',[{'role':'user','content':'abc'}],24,11,False,30)
        self.assertEqual(result['output_text'],'B')
        self.assertAlmostEqual(result['ollama_eval_tps'],10)
        self.assertFalse(sess.payload['think'])
        self.assertEqual(result['output_tokens'],4)
    def test_openrouter_stream(self):
        frames=[{'id':'gen-test','choices':[{'delta':{'content':'B'}}]},
                {'id':'gen-test','choices':[{'delta':{'content':''},'finish_reason':'stop'}],
                 'usage':{'prompt_tokens':24,'completion_tokens':3}}]
        lines=[b': OPENROUTER PROCESSING',*[('data: '+json.dumps(f)).encode() for f in frames],b'data: [DONE]']
        sess=FakeSession(FakeResponse(lines,{'X-Generation-Id':'gen-test'}))
        res=stream_openrouter(sess,'TEST','a/b',[{'role':'user','content':'foo'}],24,11,False,'DeepInfra',30)
        self.assertEqual(res['generation_id'],'gen-test')
        self.assertEqual(res['output_text'],'B')
        self.assertEqual(res['output_tokens'],3)
        self.assertEqual(sess.payload['provider']['allow_fallbacks'],False)
        self.assertEqual(sess.payload['reasoning']['enabled'],False)
    def test_no_charge_catalog(self):
        session=FakeCatalogSession([{'provider_name':'Example'}])
        result=matched_providers('nvidia/nemotron:free',session=session)
        self.assertEqual(len(result),1)
        self.assertIn('nemotron%3Afree/endpoints',session.url)
        self.assertEqual(matched_providers('missing/model',session=FakeCatalogSession([])),[])

if __name__=='__main__':unittest.main()
