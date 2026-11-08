'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),R=require('../docs/edit-plane.js');
const ctx={};vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../docs/data.js'),'utf8')+'\nthis.data=DENOISE_DATA;',ctx);const D=JSON.parse(JSON.stringify(ctx.data));
test('40 complete Python/Fraction and BigInt cell/empty certificates, including every source row and 160 queries',{timeout:120000},()=>{
const native=JSON.parse(execFileSync('python',['-c',`import json
from denoiselab.edit_plane import edit_plane,verify_plane,query_plane
from denoiselab.data import ROOT,load
from denoiselab.io import read
fit=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']];out=[]
for case in read(ROOT/'artifacts/report.json')['cases']:
 for e in case['examples']:
  visible=[i for i,m in enumerate(e['mask']) if not m];a=min(visible,key=lambda i:abs(i-27));b=min((i for i in visible if i!=a),key=lambda i:abs(i-28));axes=[a,b];w=edit_plane(fit,e['clean'],e['mask'],axes);out.append(dict(witness=w,verification=verify_plane(w,fit,e['clean'],e['mask'],axes),queries=[query_plane(w,u,v) for u,v in [('0','0'),('1/3','2/3'),('1/2','1/2'),('1','1')]]))
print(json.dumps(out,allow_nan=False))`],{cwd:path.resolve(__dirname,'..'),encoding:'utf8',maxBuffer:32e6,env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));let i=0;
for(const c of D.cases)for(const e of c.examples){const n=native[i++],axes=n.witness.input_pixels,w=R.editPlane(D.fit,e.clean,e.mask,axes);assert.deepEqual(w,n.witness);assert.deepEqual(R.verifyPlane(w,D.fit,e.clean,e.mask,axes),n.verification);assert.deepEqual([['0','0'],['1/3','2/3'],['1/2','1/2'],['1','1']].map(p=>R.queryPlane(w,...p)),n.queries);}assert.equal(i,40);
});
test('point-only and segment-only original row winner, coincident sources, geometry/empty proof tampering',()=>{
for(const [fit,x,dimension]of [[[[.5,0,.5],[0,0,0],[1,0,0]],[0,0,0],1],[[[.5,.5,.5,.5],[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0]],[0,0,0,0],0]]){const mask=x.map(()=>0),w=R.editPlane(fit,x,mask,[0,1]);assert.ok(R.verifyPlane(w,fit,x,mask,[0,1]).verified);assert.equal(w.cells.find(c=>w.groups[c.group].index===0).dimension,dimension);assert.deepEqual(R.queryPlane(w,'1/2','1/2').tied_indices,fit.map((r,i)=>i));assert.equal(R.queryPlane(w,'1/2','1/2').index,0);}
const f=[[0,0,0],[1,1,0],[.5,.5,1]],x=[0,0,0],mask=[0,0,0],w=R.editPlane(f,x,mask,[0,1]),area=w.cells.findIndex(c=>c.dimension===2),empty=w.cells.findIndex(c=>c.dimension===-1);assert.equal(R.verifyPlane(w,f,x,mask,[0,1]).empty_cells,1);
for(const mutate of [z=>z.row_lines[0][2]=1,z=>z.cells.pop(),z=>z.area=['0','1'],z=>z.cells[area].area=['0','1'],z=>z.cells[area].vertices.pop(),z=>z.cells[area].vertices.reverse(),z=>z.cells[empty].empty_proof.weights=z.cells[empty].empty_proof.weights.map(()=> '0'),z=>z.cells[empty].dimension=0,z=>z.refitted=true,z=>z.groups[0].index=99]){const z=structuredClone(w);mutate(z);assert.throws(()=>R.verifyPlane(z,f,x,mask,[0,1]));}
assert.throws(()=>R.verifyPlane(R.editPlane([[0,0,0],[1,0,0]],x,mask,[0,1]),f,x,mask,[0,1]));for(const axes of [[0,0],[0,2],[true,1],[0]])assert.throws(()=>R.editPlane(f,x,[0,0,1],axes));const a=R.editPlane([[0,0,0]],[.5,.5,.1234],[0,0,1],[0,1]);assert.deepEqual(a,R.editPlane([[0,0,0]],[.5,.5,.9876],[0,0,1],[0,1]));assert.throws(()=>R.editPlane([[0,0,0]],[.1,0,0],[0,0,1],[0,1]));
});
