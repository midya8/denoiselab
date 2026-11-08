'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const I=require('../docs/inference.js'),ctx={};vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../docs/data.js'),'utf8')+'\nthis.data=DENOISE_DATA;',ctx);const d=ctx.data;
function close(a,b,tol=2e-12){assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)assert.ok(Math.abs(a[i]-b[i])<tol,`${i}: ${a[i]} vs ${b[i]}`);}
test('independent JS inference matches Python on every frozen example and method',()=>{
 for(const c of d.cases)for(const e of c.examples){
  close(I.network(e.clean,e.mask,d.network),e.predictions.network);close(I.pca(e.clean,e.mask,d.pca),e.predictions.pca);
  let nn=I.nearest(e.clean,e.mask,d.fit);close(nn.prediction,e.predictions.nearest);assert.equal(d.fit_indices[nn.index],e.nearest_fit_original_row);
  close(I.mean(e.clean,e.mask,d.pca.mean),e.predictions.pixel_mean);
 }
});
test('hidden ground truth cannot influence any predictor; observed values survive',()=>{
 const e=d.cases[1].examples[3],other=e.clean.map((v,i)=>e.mask[i]?1-v:v);
 for(const predictor of [(x,m)=>I.network(x,m,d.network),(x,m)=>I.pca(x,m,d.pca),(x,m)=>I.nearest(x,m,d.fit).prediction,(x,m)=>I.mean(x,m,d.pca.mean)]){
  close(predictor(e.clean,e.mask),predictor(other,e.mask));close(predictor(e.clean,Array(64).fill(0)),e.clean);
  const a=predictor(e.clean,Array(64).fill(1)),b=predictor(other,Array(64).fill(1));close(a,b);
 }
});
test('mask extremes, PCA known tiny ridge solve, NN tie and invalid inputs',()=>{
 assert.equal(I.score([.2],[.8],[0]),null);assert.ok(Math.abs(I.score([.2],[.8],[1])-.36)<1e-14);
 const p={mean:[.5,.5],basis:[[Math.SQRT1_2,Math.SQRT1_2]],ridge:.5};close(I.pca([1,.9],[0,1],p),[1,.75]);
 assert.equal(I.nearest([.5,.1],[1,1],[[0,1],[1,0]]).index,0);
 assert.throws(()=>I.network([1],[1],d.network));assert.throws(()=>I.mean([NaN],[1],[.5]));assert.throws(()=>I.pca([.1,.2],[.5,0],p));
});
