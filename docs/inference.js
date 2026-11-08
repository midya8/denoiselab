"use strict";
// Independent scalar JavaScript inference, not a framework or remote model.
const DenoiseInference = (() => {
  function validate(pixels, mask, d=64) {
    if (!Array.isArray(pixels)||pixels.length!==d||!Array.isArray(mask)||mask.length!==d||pixels.some(v=>typeof v!=="number"||!Number.isFinite(v)||v<0||v>1)||mask.some(v=>v!==0&&v!==1&&v!==false&&v!==true)) throw Error("Pixels/mask must be finite [0,1] values and binary missing flags of matching width.");
  }
  function network(pixels, mask, model) {
    validate(pixels,mask,model.features);
    let a=pixels.map((v,i)=>mask[i]?0:v).concat(mask.map(Number));
    for(let layer=0;layer<4;layer++) {
      const w=model.params[2*layer],b=model.params[2*layer+1];
      a=b.map((bias,j)=>{let z=bias;for(let i=0;i<a.length;i++)z+=a[i]*w[i][j];return layer<3?Math.tanh(z):z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));});
    }
    return a.map((v,i)=>mask[i]?v:pixels[i]);
  }
  function pca(pixels, mask, model) {
    validate(pixels,mask,model.mean.length);
    const rank=model.basis.length;
    const a=Array.from({length:rank},(_,i)=>Array.from({length:rank},(_,j)=>i===j?model.ridge:0));
    const b=Array(rank).fill(0);
    for(let k=0;k<pixels.length;k++)if(!mask[k])for(let i=0;i<rank;i++){
      b[i]+=model.basis[i][k]*(pixels[k]-model.mean[k]);
      for(let j=0;j<rank;j++)a[i][j]+=model.basis[i][k]*model.basis[j][k];
    }
    const l=Array.from({length:rank},()=>Array(rank).fill(0));
    for(let i=0;i<rank;i++)for(let j=0;j<=i;j++){
      let s=a[i][j];for(let k=0;k<j;k++)s-=l[i][k]*l[j][k];
      l[i][j]=i===j?Math.sqrt(s):s/l[j][j];
    }
    const y=Array(rank).fill(0),z=Array(rank).fill(0);
    for(let i=0;i<rank;i++){let s=b[i];for(let j=0;j<i;j++)s-=l[i][j]*y[j];y[i]=s/l[i][i];}
    for(let i=rank-1;i>=0;i--){let s=y[i];for(let j=i+1;j<rank;j++)s-=l[j][i]*z[j];z[i]=s/l[i][i];}
    return pixels.map((v,k)=>{if(!mask[k])return v;let p=model.mean[k];for(let j=0;j<rank;j++)p+=z[j]*model.basis[j][k];return Math.max(0,Math.min(1,p));});
  }
  function nearest(pixels,mask,fit){
    validate(pixels,mask,fit[0].length);
    let best=Infinity,index=0;
    for(let i=0;i<fit.length;i++){let distance=0;for(let j=0;j<pixels.length;j++)if(!mask[j])distance+=(fit[i][j]-pixels[j])**2;if(distance<best){best=distance;index=i;}}
    return {prediction:pixels.map((v,j)=>mask[j]?fit[index][j]:v),index};
  }
  function mean(pixels,mask,means){validate(pixels,mask,means.length);return pixels.map((v,i)=>mask[i]?means[i]:v);}
  function score(pixels,pred,mask){const n=mask.filter(Boolean).length;return n?pixels.reduce((s,v,i)=>s+(mask[i]?(v-pred[i])**2:0),0)/n:null;}
  return {validate,network,pca,nearest,mean,score};
})();
if(typeof module!=="undefined")module.exports=DenoiseInference;
