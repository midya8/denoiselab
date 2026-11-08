import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from denoiselab import experiment
from denoiselab.data import ROOT,load,partition,masks,hidden_mse
from denoiselab.io import read,digest,save
from denoiselab.network import Denoiser
from denoiselab.baselines import PCA

class ProtocolTests(unittest.TestCase):
    def test_original_hashes_and_no_duplicate_features(self):
        x,y=load('train');t,z=load('test');self.assertEqual(x.shape,(3823,64));self.assertEqual(t.shape,(1797,64))
        f,v=partition(x,y);self.assertEqual((len(f),len(v)),(3062,761));self.assertEqual(len(set(f)&set(v)),0)
        self.assertEqual(set(f)|set(v),set(range(len(x))))
        def keys(a):return {tuple(r) for r in a}
        self.assertFalse(keys(x[f])&keys(x[v]));self.assertFalse(keys(x)&keys(t))

    def test_full_refit_prohibits_test_loader(self):
        original=experiment.load
        def train_only(part):
            self.assertEqual(part,'train','test data must not enter training or selection')
            return original(part)
        with tempfile.TemporaryDirectory() as d,patch.object(experiment,'load',side_effect=train_only):
            sel=experiment.train(d);saved=read(ROOT/'artifacts/selection.json')
            self.assertEqual(sel['selected_epoch'],saved['selected_epoch'])
            self.assertEqual(sel['selected_pca_rank'],saved['selected_pca_rank'])
            self.assertEqual(sel['selected_pca_ridge'],saved['selected_pca_ridge'])
            for name in ['network','pca','pca16']:
                a=read(Path(d)/(name+'.json'));b=read(ROOT/'artifacts'/(name+'.json'))
                for field in (['params'] if name=='network' else ['basis','mean']):
                    if field=='params':
                        for p,q in zip(a[field],b[field]):np.testing.assert_allclose(p,q,rtol=1e-7,atol=1e-8)
                    else:np.testing.assert_allclose(a[field],b[field],rtol=1e-7,atol=1e-8)
            self.assertEqual(read(Path(d)/'protocol.json'),read(ROOT/'artifacts/protocol.json'))
            self.assertEqual(read(Path(d)/'evaluation-plan.json'),experiment.EVALUATION)

    def test_frozen_full_evaluation_reproduces(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ['network','pca','pca16','protocol','selection','evaluation-plan']:save(Path(d)/(n+'.json'),read(ROOT/'artifacts'/(n+'.json')))
            got=experiment.evaluate(d);expected=read(ROOT/'artifacts/report.json')
            self.assertEqual(got['rows'],1797)
            for a,b in zip(got['cases'],expected['cases']):
                self.assertEqual(a['mask_sha256'],b['mask_sha256'])
                for name in a['methods']:
                    np.testing.assert_allclose(a['methods'][name]['row_hidden_mse'],b['methods'][name]['row_hidden_mse'],rtol=1e-10,atol=1e-12)
                    self.assertEqual(a['methods'][name]['observed_max_absolute_error'],0)
                self.assertEqual([v['row_id'] for v in a['examples']],list(range(10)))

    def test_checkpoint_selection_not_test_score(self):
        s=read(ROOT/'artifacts/selection.json');cs=s['network_candidates']
        expected=min(cs,key=lambda a:(a['validation_hidden_mse'],a['epoch']))
        self.assertEqual(s['selected_epoch'],expected['epoch'])
        self.assertLess(expected['validation_hidden_mse'],s['initial_network_validation_hidden_mse'])
        best=min(s['pca_candidates'],key=lambda a:a['validation_hidden_mse'])
        self.assertEqual((s['selected_pca_rank'],s['selected_pca_ridge']),(best['rank'],best['ridge']))

    def test_keeps_stronger_baseline_and_shift_results(self):
        j=read(ROOT/'artifacts/report.json');cases={c['name']:c for c in j['cases']}
        self.assertLess(cases['random16']['methods']['nearest']['hidden_mse'],cases['random16']['methods']['network']['hidden_mse'])
        self.assertLess(cases['random16']['methods']['pca']['hidden_mse'],cases['random16']['methods']['network']['hidden_mse'])
        for c in j['cases']:
            for row in c['methods'].values():
                self.assertEqual(len(row['row_hidden_mse']),1797)
                self.assertAlmostEqual(sum(row['row_hidden_mse'])/1797,row['hidden_mse'],places=14)
                self.assertAlmostEqual(row['all_pixel_mse'],row['hidden_mse']*c['missing']/64,places=14)
        self.assertEqual(cases['block16']['missing'],cases['random16']['missing'])

    def test_artifact_tampering_refused_before_test_load(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ['network','pca','pca16','protocol','selection','evaluation-plan']:save(Path(d)/(n+'.json'),read(ROOT/'artifacts'/(n+'.json')))
            v=read(Path(d)/'network.json');v['params'][0][0][0]+=.1;save(Path(d)/'network.json',v)
            with patch.object(experiment,'load',side_effect=AssertionError('must reject before loading')):
                with self.assertRaisesRegex(ValueError,'hash'):experiment.evaluate(d)

    def test_browser_export_exact_and_covers_all_public_cases(self):
        from examples.export_browser import content
        text=content();self.assertEqual((ROOT.parent/'docs/data.js').read_text(),text)
        data=json.loads(text.split('const DENOISE_DATA = ',1)[1].rsplit(';',1)[0])
        self.assertEqual(len(data['fit']),3062);self.assertEqual(len(data['cases']),4)
        self.assertEqual(sum(len(c['examples']) for c in data['cases']),40)
        self.assertEqual(data['source_hashes']['network.json'],digest(read(ROOT/'artifacts/network.json')))

class CLITests(unittest.TestCase):
    def cli(self,*args):return subprocess.run([sys.executable,'-m','denoiselab',*args],cwd=ROOT.parent,text=True,capture_output=True)
    def test_demo_is_complete_saved_report(self):
        p=self.cli('demo');self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),read(ROOT/'artifacts/report.json'))
    def test_predict_matches_api_and_target_erasure(self):
        ex=read(ROOT/'artifacts/report.json')['cases'][0]['examples'][0]
        with tempfile.TemporaryDirectory() as d:
            save(Path(d)/'input.json',{'pixels':[ex['clean']],'mask':[ex['mask']]})
            p=self.cli('predict','--model',str(ROOT/'artifacts/network.json'),'--input',str(Path(d)/'input.json'))
            self.assertEqual(p.returncode,0,p.stderr);np.testing.assert_allclose(json.loads(p.stdout)['prediction'][0],ex['predictions']['network'],atol=1e-14)
    def test_bad_json_and_input_exit2(self):
        for text in ['{"pixels":NaN,"mask":[]}', '{"pixels":[],"pixels":[],"mask":[]}', '{"pixels":[],"mask":[],"extra":1}']:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'input.json';p.write_text(text)
                r=self.cli('predict','--model',str(ROOT/'artifacts/network.json'),'--input',str(p));self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'')
