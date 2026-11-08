import unittest,copy,random,json,subprocess,sys
from fractions import Fraction as F
import numpy as np
from denoiselab.edit_path import edit_path,query_path,verify_path,parameter,unpack
from denoiselab.io import read
from denoiselab.data import ROOT,load

class EditPathTests(unittest.TestCase):
 def test_scalar_switch_coincident_lines_and_complete_point_ties(self):
  fit=[[0,0],[1,1],[0,1]];w=edit_path(fit,[.5,.875],[0,1],0);self.assertEqual(w['segments'][0]['tied_indices'],[0,2]);self.assertEqual(w['points'][1]['tied_indices'],[0,1,2]);self.assertEqual(query_path(w,'1/2')['index'],0);self.assertEqual(query_path(w,'500001/1000000')['index'],1);self.assertTrue(verify_path(w,fit,[.5,.1],[0,1],0)['verified'])
 def test_pruned_middle_line_can_win_only_the_triple_tie_point(self):
  # At u=.5, all distances equal. Row0 is the middle slope and wins
  # only this point, not either adjacent open interval.
  fit=[[.5,.5],[0,0],[1,0]];w=edit_path(fit,[.5,0],[0,0],0);self.assertEqual([s['index'] for s in w['segments']],[1,2]);self.assertEqual(w['points'][1]['index'],0);self.assertEqual(w['points'][1]['tied_indices'],[0,1,2]);self.assertTrue(verify_path(w,fit,[.5,0],[0,0],0)['verified'])
 def test_domain_endpoints_and_one_row(self):
  fit=[[0,0],[.5,.5]];w=edit_path(fit,[0,.5],[0,0],0);self.assertEqual(w['points'][0]['tied_indices'],[0,1]);self.assertEqual(len(w['segments']),1);self.assertEqual(w['segments'][0]['index'],1)
  w=edit_path([[.25,.5]],[.5,.1],[0,1],0);self.assertEqual(len(w['segments']),1);self.assertEqual(query_path(w,'1')['index'],0)
 def test_all_pair_breakpoints_and_midpoints_independent_distance_oracle(self):
  rng=random.Random(12);checked=0
  for _ in range(80):
   d=4;n=12;fit=[[rng.randrange(17)/16 for j in range(d)] for i in range(n)];x=[rng.randrange(17)/16 for j in range(d)];mask=[0,1,rng.randrange(2),1];w=edit_path(fit,x,mask,0);self.assertTrue(verify_path(w,fit,x,mask,0)['verified']);breaks={F(0),F(1)}
   for a in range(n):
    for b in range(a):
     # Pair equation computed directly from rational squared source distances,
     # without row_lines, groups, hull starts or production coefficient helpers.
     const=lambda row:sum((F(x[j])-F(row[j]))**2 for j in range(1,d) if not mask[j])+F(row[0])**2
     slope=2*(F(fit[a][0])-F(fit[b][0]))
     if slope:
      q=(const(fit[a])-const(fit[b]))/slope
      if 0<q<1:breaks.add(q)
   ordered=sorted(breaks);points=ordered+[(a+b)/2 for a,b in zip(ordered,ordered[1:])]
   for q in points:
    ds=[sum(((q-F(row[j])) if j==0 else F(x[j])-F(row[j]))**2 for j in range(d) if not mask[j]) for row in fit];best=min(ds);ties=[i for i,v in enumerate(ds) if v==best];answer=query_path(w,str(q));self.assertEqual(answer['tied_indices'],ties);self.assertEqual(unpack(answer['squared_distance']),best)
    segments=[s for s in w['segments'] if unpack(s['left'])<q<unpack(s['right'])]
    if segments:self.assertEqual(segments[0]['index'],ties[0])
    checked+=1
  self.assertEqual(checked,5344)
 def test_hidden_erasure_and_no_input_mutation(self):
  fit=[[0,0],[1,1]];x=[.5,.1234];mask=[0,1];before=copy.deepcopy([fit,x,mask]);a=edit_path(fit,x,mask,0);b=edit_path(fit,[.5,.98765],mask,0);self.assertEqual(a,b);self.assertEqual([fit,x,mask],before);self.assertEqual(a['observed'],[.5,0]);self.assertFalse(a['hidden_targets_used'])
 def test_source_and_certificate_tampering_refused(self):
  fit=[[.5,.5],[0,0],[1,0]];x=[.5,0];mask=[0,0];w=edit_path(fit,x,mask,0)
  mutations=[lambda z:z['row_lines'][0].update(intercept=0),lambda z:z['segments'][0].update(right=['2','3']),lambda z:z['segments'][0].update(index=0),lambda z:z['segments'][0].update(minimum_endpoint_slack=['1','1']),lambda z:z['points'][1].update(tied_indices=[1,2]),lambda z:z['groups'][0].update(index=2),lambda z:z.update(hidden_targets_used=True),lambda z:z['segments'][0].update(left=['0','2']),lambda z:z.update(dominance_checks=0),lambda z:z['segments'].pop(),lambda z:z['points'][0].update(squared_distance=['1','1'])]
  for mutate in mutations:
   z=copy.deepcopy(w);mutate(z)
   with self.assertRaises(ValueError):verify_path(z,fit,x,mask,0)
  with self.assertRaises(ValueError):verify_path(w,[[.5,.5],[0,.5],[1,.5]],x,mask,0)
  # A different source can have a self-consistent proof; it is not this source.
  alternative=edit_path([[0,0],[1,1]],x,mask,0)
  with self.assertRaises(ValueError):verify_path(alternative,fit,x,mask,0)
 def test_parameter_and_invalid_scope(self):
  self.assertEqual(parameter('0.5'),F(1,2));self.assertEqual(parameter('2/4'),F(1,2))
  for v in [.5,'1/0','-1','1e-2','.5','nan','2','1/2junk','1'*161,'0.'+'1'*81]:
   with self.assertRaises(ValueError):parameter(v)
  for fit,x,mask,pixel in [([],[],[],0),([[0,0]],[.1,0],[0,1],0),([[.1,0]],[.5,0],[0,1],0),([[0,0]],[.5,0],[0,1],1),([[0,0]],[.5,0],[0,1],True)]:
   with self.assertRaises(ValueError):edit_path(fit,x,mask,pixel)
 def test_saved_40_paths_all_3062_rows_source_dominance_and_exact_queries(self):
  fit=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']];report=read(ROOT/'artifacts/report.json');checked=0
  self.assertTrue(np.all(fit*16==np.floor(fit*16)))
  for case in report['cases']:
   for e in case['examples']:
    pixel=min((i for i,v in enumerate(e['mask']) if not v),key=lambda i:abs(i-27));w=edit_path(fit,e['clean'],e['mask'],pixel);v=verify_path(w,fit,e['clean'],e['mask'],pixel);self.assertTrue(v['source_matched']);self.assertLessEqual(len(w['groups']),17)
    for value in ['0','1/3','1/2','1']:
     q=F(value);ints=(fit*16).astype(int);clean=np.array(e['clean']);visible=np.array(e['mask'])==0;other=visible.copy();other[pixel]=False;b=np.sum((ints[:,other]-(clean[other]*16).astype(int))**2,axis=1);ds=[(16*q-int(row[pixel]))**2+int(rest) for row,rest in zip(ints,b)];best=min(ds);ties=[i for i,s in enumerate(ds) if s==best];out=query_path(w,value);self.assertEqual(out['tied_indices'],ties);self.assertEqual(unpack(out['squared_distance']),best/256);checked+=1
  self.assertEqual(checked,160)

 def test_cli_build_source_verification_and_tamper_refusal(self):
  from pathlib import Path
  import tempfile
  root=Path(__file__).resolve().parents[1]
  run=lambda command,file:subprocess.run([sys.executable,'-m','denoiselab',command,'--input',str(file)],capture_output=True,text=True)
  built=run('edit-path',root/'examples/edit-path.json');self.assertEqual(built.returncode,0,built.stderr);out=json.loads(built.stdout);self.assertTrue(out['verification']['source_matched']);self.assertEqual(out['query']['parameter'],['1','3'])
  with tempfile.TemporaryDirectory() as tmp:
   v=json.loads((root/'examples/edit-path.json').read_text());del v['value'];v['witness']=out['path'];p=Path(tmp)/'verify.json';p.write_text(json.dumps(v));self.assertEqual(run('verify-path',p).returncode,0);v['witness']['points'][0]['index']=-1;p.write_text(json.dumps(v));self.assertEqual(run('verify-path',p).returncode,2)
