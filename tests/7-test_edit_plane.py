import unittest,copy,random
from fractions import Fraction as F
from itertools import combinations
import numpy as np
from denoiselab.edit_plane import edit_plane,verify_plane,query_plane
from denoiselab.data import ROOT,load
from denoiselab.io import read


def rational(v):return F(int(v[0]),int(v[1]))
def point(p):return tuple(rational(q) for q in p)
def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def hull(points):
 points=sorted(set(points))
 if len(points)<2:return points
 low=[];high=[]
 for sequence,out in [(points,low),(points[::-1],high)]:
  for p in sequence:
   while len(out)>=2 and cross(out[-2],out[-1],p)<=0:out.pop()
   out.append(p)
 return low[:-1]+high[:-1]

def distances(fit,x,mask,axes,u,v):
 edited=list(map(F,x));edited[axes[0]]=u;edited[axes[1]]=v
 return [sum((edited[j]-F(row[j]))**2 for j in range(len(x)) if not mask[j]) for row in fit]

def oracle_cell(fit,x,mask,axes,index):
 # Independent source-distance construction and every pair of boundary lines.
 # No production coefficients, grouping, clipping, Farkas or geometry helpers.
 base=distances(fit,x,mask,axes,F(0),F(0));dx=distances(fit,x,mask,axes,F(1),F(0));dy=distances(fit,x,mask,axes,F(0),F(1));cs=[(F(1),F(0),F(0)),(F(-1),F(0),F(1)),(F(0),F(1),F(0)),(F(0),F(-1),F(1))]
 for i in range(len(fit)):cs.append((dx[i]-base[i]-dx[index]+base[index],dy[i]-base[i]-dy[index]+base[index],base[i]-base[index]))
 vertices=[]
 for a,b in combinations(cs,2):
  det=a[0]*b[1]-a[1]*b[0]
  if not det:continue
  u=(a[1]*b[2]-a[2]*b[1])/det;v=(a[2]*b[0]-a[0]*b[2])/det
  if all(c[0]*u+c[1]*v+c[2]>=0 for c in cs):vertices.append((u,v))
 return hull(vertices)

class EditPlaneTests(unittest.TestCase):
 def test_segment_only_and_point_only_winners_are_retained(self):
  for fit,x,dimension in [([[.5,0,.5],[0,0,0],[1,0,0]],[0,0,0],1),([[.5,.5,.5,.5],[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0]],[0,0,0,0],0)]:
   mask=[0]*len(x);w=edit_plane(fit,x,mask,[0,1]);v=verify_plane(w,fit,x,mask,[0,1]);cell=next(c for c in w['cells'] if w['groups'][c['group']]['index']==0);self.assertEqual(cell['dimension'],dimension);self.assertEqual(query_plane(w,'1/2','1/2')['tied_indices'],list(range(len(fit))));self.assertEqual(query_plane(w,'1/2','1/2')['index'],0);self.assertEqual(v['area'],['1','1'])
 def test_coincident_lines_and_empty_regions(self):
  f=[[0,0,0],[0,0,1],[1,1,0],[.5,.5,1],[0,0,0]];x=[0,0,0];w=edit_plane(f,x,[0,0,0],[0,1]);v=verify_plane(w,f,x,[0,0,0],[0,1]);self.assertEqual(v['empty_cells'],1);self.assertEqual(query_plane(w,'0','0')['tied_indices'],[0,4]);self.assertEqual(query_plane(w,'1/2','1/2')['tied_indices'],[0,2,4]);self.assertEqual(len(w['groups']),3)
 def test_independent_boundary_pair_enumeration_and_direct_distance_queries(self):
  rng=random.Random(83);checked=0;cells=0
  for _ in range(30):
   fit=[[rng.randrange(17)/16 for _ in range(4)] for _ in range(8)];x=[rng.randrange(17)/16 for _ in range(4)];mask=[0,0,rng.randrange(2),0];axes=[0,1];w=edit_plane(fit,x,mask,axes);self.assertTrue(verify_plane(w,fit,x,mask,axes)['verified']);queries={(F(a,6),F(b,6)) for a in range(7) for b in range(7)}
   for cell in w['cells']:
    expected=oracle_cell(fit,x,mask,axes,w['groups'][cell['group']]['index']);actual=[point(p) for p in cell['vertices']];self.assertEqual(actual,expected);queries.update(expected);queries.update(((p[0]+q[0])/2,(p[1]+q[1])/2) for p,q in zip(expected,expected[1:]+expected[:1]));cells+=1
   for u,v in queries:
    ds=distances(fit,x,mask,axes,u,v);best=min(ds);ties=[i for i,d in enumerate(ds) if d==best];out=query_plane(w,str(u),str(v));self.assertEqual(out['tied_indices'],ties);self.assertEqual(rational(out['squared_distance']),best);checked+=1
  print('Independent plane oracle:',cells,'cells',checked,'queries');self.assertEqual((cells,checked),(238,2110))
 def test_hidden_erasure_no_mutation_and_one_row(self):
  f=[[0,0,0]];x=[.5,.5,.123];mask=[0,0,1];original=copy.deepcopy([f,x,mask]);w=edit_plane(f,x,mask,[0,1]);self.assertEqual(w,edit_plane(f,[.5,.5,.9876],mask,[0,1]));self.assertEqual([f,x,mask],original);self.assertEqual(w['cells'][0]['dimension'],2);self.assertTrue(verify_plane(w,f,x,mask,[0,1])['verified'])
 def test_invalid_scope(self):
  f=[[0,0,0]]
  for x,m,axes in [([0,0,0],[0,0,1],[0,0]),([0,0,0],[0,0,1],[0,2]),([0,0,0],[0,0,1],[True,1]),([.1,0,0],[0,0,1],[0,1]),([0,0,0],[0,0,1],[0])]:
   with self.assertRaises(ValueError):edit_plane(f,x,m,axes)
  with self.assertRaises(ValueError):edit_plane([[.1,0,0]],[0,0,0],[0,0,1],[0,1])
 def test_source_geometry_and_infeasibility_tampering(self):
  f=[[0,0,0],[1,1,0],[.5,.5,1]];x=[0,0,0];mask=[0,0,0];w=edit_plane(f,x,mask,[0,1]);area=next(i for i,c in enumerate(w['cells']) if c['dimension']==2);empty=next(i for i,c in enumerate(w['cells']) if c['dimension']==-1)
  mutations=[lambda z:z['row_lines'][0].__setitem__(2,1),lambda z:z['cells'].pop(),lambda z:z.update(area=['0','1']),lambda z:z['cells'][area].update(area=['0','1']),lambda z:z['cells'][area]['vertices'].pop(),lambda z:z['cells'][area]['vertices'].reverse(),lambda z:z['cells'][empty]['empty_proof'].update(weights=['0']*len(z['cells'][empty]['empty_proof']['weights'])),lambda z:z['cells'][empty].update(dimension=0),lambda z:z.update(refitted=True),lambda z:z['groups'][0].update(index=99)]
  for change in mutations:
   z=copy.deepcopy(w);change(z)
   with self.assertRaises(ValueError):verify_plane(z,f,x,mask,[0,1])
  alternative=edit_plane([[0,0,0],[1,0,0]],x,mask,[0,1])
  with self.assertRaises(ValueError):verify_plane(alternative,f,x,mask,[0,1])
  # A strict feasible subpolygon is not the maximal cell: its invented edges
  # have no supporting source/domain constraint, even if its area is updated.
  z=copy.deepcopy(w);z['cells'][area]['vertices']=[[['0','1'],['0','1']],[['1','4'],['0','1']],[['0','1'],['1','4']]];z['cells'][area]['area']=['1','32']
  with self.assertRaises(ValueError):verify_plane(z,f,x,mask,[0,1])
 def test_40_saved_planes_all_3062_rows_and_independent_queries(self):
  fit=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']];count=0;queries=0;dims=[0,0,0,0]
  for case in read(ROOT/'artifacts/report.json')['cases']:
   for e in case['examples']:
    visible=[i for i,m in enumerate(e['mask']) if not m];a=min(visible,key=lambda i:abs(i-27));b=min((i for i in visible if i!=a),key=lambda i:abs(i-28));axes=[a,b];w=edit_plane(fit,e['clean'],e['mask'],axes);v=verify_plane(w,fit,e['clean'],e['mask'],axes);self.assertTrue(v['source_matched']);self.assertLessEqual(v['groups'],289);count+=1
    for c in w['cells']:dims[c['dimension']+1]+=1
    # Independent squared count distances with the full original source.
    ints=(fit*16).astype(int);visible=np.array(e['mask'])==0;other=visible.copy();other[axes]=False;base=np.sum((ints[:,other]-(np.array(e['clean'])[other]*16).astype(int))**2,axis=1)
    for u,vv in [('0','0'),('1/3','2/3'),('1/2','1/2'),('1','1')]:
     u0,v0=F(u),F(vv);ds=[((16*u0-int(row[a]))**2+(16*v0-int(row[b]))**2+int(rest))/256 for row,rest in zip(ints,base)];best=min(ds);ties=[i for i,d in enumerate(ds) if d==best];answer=query_plane(w,u,vv);self.assertEqual(answer['tied_indices'],ties);self.assertEqual(rational(answer['squared_distance']),best);queries+=1
  self.assertEqual((count,queries),(40,160));print('Stored plane dimensions empty/point/segment/area',dims)

 def test_cli_build_source_verify_tamper_and_parameter_refusal(self):
  import json,subprocess,sys,tempfile
  from pathlib import Path
  root=Path(__file__).resolve().parents[1]
  run=lambda cmd,p:subprocess.run([sys.executable,'-m','denoiselab',cmd,'--input',str(p)],capture_output=True,text=True)
  result=run('edit-plane',root/'examples/edit-plane.json');self.assertEqual(result.returncode,0,result.stderr);out=json.loads(result.stdout);self.assertTrue(out['verification']['source_matched']);self.assertEqual(out['query']['parameters'],[['1','3'],['2','3']])
  with tempfile.TemporaryDirectory() as tmp:
   v=json.loads((root/'examples/edit-plane.json').read_text());v.pop('values');v['witness']=out['plane'];p=Path(tmp)/'verify.json';p.write_text(json.dumps(v));self.assertEqual(run('verify-plane',p).returncode,0);v['witness']['area']=['0','1'];p.write_text(json.dumps(v));self.assertEqual(run('verify-plane',p).returncode,2)
   v=json.loads((root/'examples/edit-plane.json').read_text());v['values']=[.5,'0'];p.write_text(json.dumps(v));self.assertEqual(run('edit-plane',p).returncode,2)
