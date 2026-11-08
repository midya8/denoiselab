'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),R=require('../docs/edit-path.js');
const ctx={};vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../docs/data.js'),'utf8')+'\nthis.data=DENOISE_DATA;',ctx);const D=JSON.parse(JSON.stringify(ctx.data));
test('40 complete Python/Fraction and scalar BigInt path proofs, every coefficient/tie/interval, 160 queries',()=>{
 const native=JSON.parse(execFileSync('python',['-c',`import json
from denoiselab.edit_path import edit_path,query_path,verify_path
from denoiselab.data import ROOT,load
from denoiselab.io import read
f=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']];out=[]
for case in read(ROOT/'artifacts/report.json')['cases']:
 for e in case['examples']:
  pixel=min((i for i,v in enumerate(e['mask']) if not v),key=lambda i:abs(i-27));w=edit_path(f,e['clean'],e['mask'],pixel);out.append(dict(witness=w,verification=verify_path(w,f,e['clean'],e['mask'],pixel),queries=[query_path(w,v) for v in ['0','1/3','1/2','1']]))
print(json.dumps(out,allow_nan=False))`],{cwd:path.resolve(__dirname,'..'),encoding:'utf8',maxBuffer:32e6,env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));let i=0;
 for(const c of D.cases)for(const e of c.examples){const pixel=e.mask.map((v,i)=>v?-1:i).filter(i=>i>=0).reduce((a,b)=>Math.abs(b-27)<Math.abs(a-27)?b:a),w=R.editPath(D.fit,e.clean,e.mask,pixel),n=native[i++];assert.deepEqual(w,n.witness);assert.deepEqual(R.verifyPath(w,D.fit,e.clean,e.mask,pixel),n.verification);assert.deepEqual(['0','1/3','1/2','1'].map(v=>R.queryPath(w,v)),n.queries);}assert.equal(i,40);
});
test('BigInt full point ties including a pruned boundary-only winner; source/tamper/parse refusals',()=>{
 const fit=[[.5,.5],[0,0],[1,0]],x=[.5,0],mask=[0,0],w=R.editPath(fit,x,mask,0);assert.deepEqual(w.segments.map(s=>s.index),[1,2]);assert.deepEqual(w.points[1].tied_indices,[0,1,2]);assert.equal(w.points[1].index,0);assert.equal(R.queryPath(w,'1/2').index,0);assert.ok(R.verifyPath(w,fit,x,mask,0).verified);
 for(const mutate of [z=>z.row_lines[0].intercept=0,z=>z.segments[0].right=['2','3'],z=>z.segments[0].index=0,z=>z.points[1].tied_indices=[1,2],z=>z.segments[0].minimum_endpoint_slack=['1','1'],z=>z.groups[0].index=2,z=>z.hidden_targets_used=true,z=>z.segments[0].left=['0','2'],z=>z.dominance_checks=0,z=>z.segments.pop(),z=>z.points[0].squared_distance=['1','1']]){const z=structuredClone(w);mutate(z);assert.throws(()=>R.verifyPath(z,fit,x,mask,0));}
 assert.throws(()=>R.verifyPath(w,[[0,.5],[0,0],[1,0]],x,mask,0));for(const v of [.5,'1/0','-1','1e-2','.5','nan','2','1/2junk','1'.repeat(161),'0.'+'1'.repeat(81)])assert.throws(()=>R.parameter(v));assert.deepEqual(R.parameter('2/4'),['1','2']);assert.deepEqual(R.parameter('0.1'),['1','10']);const coincident=R.editPath([[0,0],[1,1],[0,1]],[.5,.123],[0,1],0);assert.deepEqual(coincident.segments[0].tied_indices,[0,2]);assert.deepEqual(coincident.points[1].tied_indices,[0,1,2]);assert.deepEqual(coincident,R.editPath([[0,0],[1,1],[0,1]],[.5,.9876],[0,1],0));assert.throws(()=>R.editPath([[0,0]],[.1,0],[0,1],0));
});
