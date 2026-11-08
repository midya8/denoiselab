'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path'),R=require('../docs/response.js');
const context={};vm.createContext(context);vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../docs/data.js'),'utf8')+'\nthis.data=DENOISE_DATA;',context);const D=JSON.parse(JSON.stringify(context.data));
function close(a,b,tol=2e-9){if(a===null||b===null)return assert.equal(a,b);if(Array.isArray(a)){assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tol));}else if(typeof a==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort());for(const k of Object.keys(a))close(a[k],b[k],tol);}else if(typeof a==='number')assert.ok(Math.abs(a-b)<=tol*Math.max(1,Math.abs(a),Math.abs(b)),`${a} vs ${b}`);else assert.equal(a,b);}
test('120 full Python/JS responses, all 64x64 Jacobians, finite edits and neighbor indices',()=>{
const out=JSON.parse(execFileSync('python',['-c',`import json,numpy as np
from denoiselab.io import read
from denoiselab.data import ROOT,load
from denoiselab.network import Denoiser
from denoiselab.baselines import PCA
from denoiselab.response import response
n=Denoiser.from_dict(read(ROOT/'artifacts/network.json'));p=PCA.from_dict(read(ROOT/'artifacts/pca.json'));f=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']];out=[]
for ci,c in enumerate(read(ROOT/'artifacts/report.json')['cases']):
 for e in c['examples']:
  pixel=e['mask'].index(0)
  for value in [0,.5,1]:
   edited=e['clean'][:];edited[pixel]=value;out.append([ci,e['row_id'],pixel,value,response(n,p,f,e['clean'],e['mask'],edited)])
print(json.dumps(out,allow_nan=False))`],{cwd:path.resolve(__dirname,'..'),encoding:'utf8',maxBuffer:128e6,env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));assert.equal(out.length,120);for(const [ci,row,pixel,value,w]of out){const e=D.cases[ci].examples.find(e=>e.row_id===row),edited=e.clean.slice();edited[pixel]=value;const a=R.response(D.network,D.pca,D.fit,e.clean,e.mask,edited);close(a,w);}
});
test('independent scalar network chain and ridge finite differences, all observed columns',()=>{
const I=require('../docs/inference.js'),e=D.cases[1].examples[0],x=e.clean.map((v,i)=>e.mask[i]?0:Math.max(.1,Math.min(.9,v))),eps=1e-5,n=R.networkJacobian(D.network,x,e.mask),p=R.pcaJacobian(D.pca,x,e.mask);for(let col=0;col<64;col++){const plus=x.slice(),minus=x.slice();plus[col]+=eps;minus[col]-=eps;if(e.mask[col]){for(let row=0;row<64;row++){assert.equal(n.jacobian[row][col],0);if(p.jacobian[row][col]!==null)assert.equal(p.jacobian[row][col],0);}continue;}for(const [model,j,fn]of [[D.network,n,I.network],[D.pca,p,I.pca]]){const a=fn(plus,e.mask,model),b=fn(minus,e.mask,model);for(let row=0;row<64;row++)if(j.jacobian[row][col]!==null)close(j.jacobian[row][col],(a[row]-b[row])/(2*eps),2e-7);}}
});
test('clipping boundary, saturation and nearest switch scalar controls',()=>{
const p={mean:[0,0],basis:[[1,1]],ridge:.1};const r=R.pcaJacobian(p,[0,0],[0,1]);assert.deepEqual(r.undefined_output_rows,[1]);assert.deepEqual(r.jacobian[1],[null,null]);const constant=R.pcaJacobian({...p,basis:[[0,0]]},[.5,0],[0,1]);assert.deepEqual(constant.undefined_output_rows,[]);assert.deepEqual(constant.jacobian[1],[0,0]);const saturated=R.pcaJacobian({...p,mean:[0,-2]},[.5,0],[0,1]);assert.deepEqual(saturated.clipped_output_rows,[1]);assert.deepEqual(saturated.jacobian[1],[0,0]);const I=require('../docs/inference.js');assert.equal(I.nearest([.5,0],[0,1],[[0,0],[1,1]]).index,0);assert.equal(I.nearest([.500001,0],[0,1],[[0,0],[1,1]]).index,1);assert.throws(()=>R.networkJacobian(D.network,Array(64).fill(2),Array(64).fill(0)));
});
