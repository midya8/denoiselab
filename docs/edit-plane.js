/* Exact two-observed-coordinate power cells. Established geometry, fixed source. */
'use strict';
(()=>{
const P=typeof module!=='undefined'?require('./edit-path.js'):DenoiseEditPath;
const scope='Exact rational two-observed-coordinate slice for fixed quantized inputs and mask. Not a general robustness radius or a binary64 tie guarantee.';
const DOMAIN=[[1,0,0],[-1,0,1],[0,1,0],[0,-1,1]];
const abs=x=>x<0n?-x:x;function gcd(a,b){a=abs(a);b=abs(b);while(b){const c=a%b;a=b;b=c;}return a;}
function rat(n,d=1n){n=BigInt(n);d=BigInt(d);if(!d)throw Error('zero denominator');if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return [n/g,d/g];}
const ZERO=rat(0),ONE=rat(1),cmp=(a,b)=>a[0]*b[1]<b[0]*a[1]?-1:a[0]*b[1]>b[0]*a[1]?1:0;
const add=(a,b)=>rat(a[0]*b[1]+b[0]*a[1],a[1]*b[1]),neg=a=>[-a[0],a[1]],sub=(a,b)=>add(a,neg(b)),mul=(a,b)=>rat(a[0]*b[0],a[1]*b[1]),div=(a,b)=>rat(a[0]*b[1],a[1]*b[0]),encode=q=>q.map(String);
function equal(a,b){if(a===null||b===null)return a===b;if(Array.isArray(a)&&Array.isArray(b))return a.length===b.length&&a.every((v,i)=>equal(v,b[i]));if(typeof a==='object'&&typeof b==='object'){const ka=Object.keys(a).sort(),kb=Object.keys(b).sort();return equal(ka,kb)&&ka.every(k=>equal(a[k],b[k]));}return a===b;}
function decode(v){if(!Array.isArray(v)||v.length!==2||v.some(s=>typeof s!=='string'||s.length>160||!(/^-?(?:0|[1-9]\d*)$/).test(s)))throw Error('rational encoding');const q=rat(v[0],v[1]);if(!equal(encode(q),v))throw Error('noncanonical rational');return q;}
const parameter=s=>decode(P.parameter(s));
function counts(row){if(!Array.isArray(row)||row.some(v=>!Number.isFinite(v)||v<0||v>1||!Number.isInteger(v*16)))throw Error('source must have exact multiples of 1/16');return row.map(v=>v*16);}
function source(fit,pixels,mask,axes){
 if(!Array.isArray(fit)||!fit.length||fit.length>10000||!Array.isArray(fit[0])||fit[0].length<2||fit[0].length>64)throw Error('fit dimensions');const d=fit[0].length;
 if(!Array.isArray(pixels)||pixels.length!==d||pixels.some(v=>!Number.isFinite(v)||v<0||v>1)||!Array.isArray(mask)||mask.length!==d||mask.some(v=>![0,1,false,true].includes(v)))throw Error('input/mask');
 if(!Array.isArray(axes)||axes.length!==2||axes[0]===axes[1]||axes.some(i=>!Number.isInteger(i)||i<0||i>=d||mask[i]))throw Error('two distinct observed coordinates required');
 const clean=pixels.map((v,j)=>mask[j]?0:v),q=counts(clean),fc=fit.map(row=>{if(row.length!==d)throw Error('fit width');return counts(row);});const [a,b]=axes;
 const lines=fc.map(row=>{let c=row[a]**2+row[b]**2;for(let j=0;j<d;j++)if(!mask[j]&&j!==a&&j!==b)c+=(row[j]-q[j])**2;return [-32*row[a]||0,-32*row[b]||0,c];});const byKey=new Map();
 lines.forEach((line,i)=>{const key=line.slice(0,2).join(),old=byKey.get(key);if(!old||line[2]<old.line[2])byKey.set(key,{line,index:i,tied_indices:[i]});else if(line[2]===old.line[2])old.tied_indices.push(i);});
 return {schema:'denoiselab-edit-plane-v1',input_pixels:axes.slice(),mask:mask.map(Number),observed:clean,fit_count:fit.length,row_lines:lines,groups:[...byKey.values()].sort((g,h)=>h.line[0]-g.line[0]||h.line[1]-g.line[1]),hidden_targets_used:false,refitted:false,scope};
}
const evaluate=(c,p)=>add(add(mul(rat(c[0]),p[0]),mul(rat(c[1]),p[1])),rat(c[2]));
const cross=(a,b)=>sub(mul(a[0],b[1]),mul(a[1],b[0])),delta=(a,b)=>[sub(a[0],b[0]),sub(a[1],b[1])],pointCmp=(a,b)=>cmp(a[0],b[0])||cmp(a[1],b[1]);
function area(poly){let sum=ZERO;poly.forEach((p,i)=>{sum=add(sum,cross(p,poly[(i+1)%poly.length]));});return mul(sum,rat(1,2));}
function normalize(poly){let out=[];for(const p of poly)if(!out.length||pointCmp(p,out.at(-1)))out.push(p);if(out.length>1&&!pointCmp(out[0],out.at(-1)))out.pop();if(out.length<2)return out;
 if(!cmp(area(out),ZERO)){out.sort(pointCmp);return pointCmp(out[0],out.at(-1))?[out[0],out.at(-1)]:[out[0]];}
 while(out.length>2){const keep=out.filter((p,i)=>cmp(cross(delta(p,out[(i+out.length-1)%out.length]),delta(out[(i+1)%out.length],p)),ZERO));if(keep.length===out.length)break;out=keep;}
 if(cmp(area(out),ZERO)<0)out.reverse();let k=0;out.forEach((p,i)=>{if(pointCmp(p,out[k])<0)k=i;});return out.slice(k).concat(out.slice(0,k));
}
function clip(poly,c){if(!poly.length)return [];const out=[];let s=poly.at(-1),sv=evaluate(c,s);for(const e of poly){const ev=evaluate(c,e),a=cmp(sv,ZERO),b=cmp(ev,ZERO);if((a<0&&b>=0)||(b<0&&a>=0)){const t=div(sv,sub(sv,ev));out.push([add(s[0],mul(t,sub(e[0],s[0]))),add(s[1],mul(t,sub(e[1],s[1])))]);}if(b>=0)out.push(e);s=e;sv=ev;}return normalize(out);}
const constraints=(groups,k)=>DOMAIN.concat(groups.map(g=>g.line.map((v,j)=>v-groups[k].line[j])));
const det=(a,b)=>a[0]*b[1]-a[1]*b[0];
function* combinations(ids,size,start=0,prefix=[]){if(!size){yield prefix;return;}for(let i=start;i<=ids.length-size;i++)yield* combinations(ids,size-1,i+1,prefix.concat(ids[i]));}
function emptyCertificate(cs,previous,cut){const ids=[];for(let i=0;i<cut;i++)if(previous.some(p=>!cmp(evaluate(cs[i],p),ZERO)))ids.push(i);ids.push(cut);
 for(const size of [1,2,3])for(const chosen of combinations(ids,size)){if(!chosen.includes(cut))continue;const lines=chosen.map(i=>cs[i]);let weights;
  if(size===1)weights=[1];else if(size===2){const [p,q]=lines;if(det(p,q))continue;const axis=p[0]||q[0]?0:1;weights=[Math.abs(q[axis]),Math.abs(p[axis])];}else{const [p,q,r]=lines;weights=[det(q,r),det(r,p),det(p,q)];if(weights.every(w=>w<=0))weights=weights.map(w=>-w);}
  if(!weights.some(Boolean)||weights.some(w=>w<0))continue;const sums=[0,1,2].map(j=>weights.reduce((s,w,i)=>s+w*lines[i][j],0));if(sums[0]===0&&sums[1]===0&&sums[2]<0){const g=weights.map(BigInt).reduce(gcd);return {constraints:chosen,weights:weights.map(w=>(BigInt(w)/g).toString())};}
 }throw Error('no infeasibility combination recovered');
}
function editPlane(fit,pixels,mask,axes){const out=source(fit,pixels,mask,axes),square=[[ZERO,ZERO],[ONE,ZERO],[ONE,ONE],[ZERO,ONE]],cells=[];
 out.groups.forEach((g,k)=>{const cs=constraints(out.groups,k);let poly=square.slice(),proof=null;for(let i=0;i<cs.length;i++){const previous=poly;poly=clip(poly,cs[i]);if(!poly.length){proof=emptyCertificate(cs,previous,i);break;}}const dimension=!poly.length?-1:poly.length===1?0:poly.length===2?1:2;cells.push({group:k,dimension,vertices:poly.map(p=>p.map(encode)),area:encode(area(poly)),empty_proof:proof});});
 out.cells=cells;out.area=encode(cells.reduce((s,c)=>add(s,decode(c.area)),ZERO));if(!equal(out.area,['1','1']))throw Error('area coverage');return out;
}
function queryPlane(w,u,v){const x=parameter(u),y=parameter(v),scores=w.row_lines.map(c=>evaluate(c,[x,y]));let best=scores[0];for(const s of scores)if(cmp(s,best)<0)best=s;const ties=[];scores.forEach((s,i)=>{if(!cmp(s,best))ties.push(i);});return {parameters:[encode(x),encode(y)],index:ties[0],tied_indices:ties,reduced_distance:encode(best),squared_distance:encode(mul(add(mul(rat(256),add(mul(x,x),mul(y,y))),best),rat(1,256)))};}
function verifyPlane(w,fit,pixels,mask,axes){const expected=source(fit,pixels,mask,axes);for(const key in expected)if(!equal(w[key],expected[key]))throw Error('source mismatch: '+key);const groups=expected.groups;if(!Array.isArray(w.cells)||w.cells.length!==groups.length)throw Error('missing cells');let total=ZERO,checks=0;const dims=[0,0,0,0];
 w.cells.forEach((cell,k)=>{if(cell.group!==k||!Number.isInteger(cell.dimension)||cell.dimension< -1||cell.dimension>2)throw Error('cell index/dimension');const dimension=cell.dimension;dims[dimension+1]++;const cs=constraints(groups,k);
  if(!Array.isArray(cell.vertices)||cell.vertices.length>cs.length)throw Error('vertex size');const poly=cell.vertices.map(p=>{if(!Array.isArray(p)||p.length!==2)throw Error('vertex');return p.map(decode);});if(new Set(cell.vertices.map(p=>JSON.stringify(p))).size!==poly.length)throw Error('duplicate vertices');const a=decode(cell.area);if(cmp(a,area(poly)))throw Error('false area');total=add(total,a);
  if(dimension===-1){if(poly.length||cmp(a,ZERO))throw Error('empty geometry');const proof=cell.empty_proof,ids=proof.constraints;if(!Array.isArray(ids)||ids.length<1||ids.length>3||new Set(ids).size!==ids.length||ids.some(i=>!Number.isInteger(i)||i<0||i>=cs.length)||!Array.isArray(proof.weights)||ids.length!==proof.weights.length)throw Error('empty proof');const weights=proof.weights.map(s=>{if(typeof s!=='string'||s.length>160||!(/^(?:0|[1-9]\d*)$/).test(s))throw Error('weight');return BigInt(s);});if(!weights.some(Boolean))throw Error('zero combination');const sums=[0,1,2].map(j=>weights.reduce((s,v,i)=>s+v*BigInt(cs[ids[i]][j]),0n));if(sums[0]||sums[1]||sums[2]>=0n)throw Error('false infeasibility combination');return;}
  if(cell.empty_proof!==null||(dimension<2&&poly.length!==dimension+1)||(dimension===2&&poly.length<3))throw Error('dimension encoding');if(poly.some(p=>cs.some(c=>cmp(evaluate(c,p),ZERO)<0)))throw Error('infeasible vertex');checks+=poly.length*cs.length;
  if(dimension===2){if(cmp(a,ZERO)<=0)throw Error('orientation/area');poly.forEach((p,i)=>{const q=poly[(i+1)%poly.length],edge=delta(q,p);if(poly.some(r=>cmp(cross(edge,delta(r,p)),ZERO)<0)||cmp(cross(edge,delta(poly[(i+2)%poly.length],q)),ZERO)<=0)throw Error('nonconvex/noncanonical');if(!cs.some(c=>(c[0]||c[1])&&!cmp(evaluate(c,p),ZERO)&&!cmp(evaluate(c,q),ZERO)))throw Error('unsupported edge');});}
  else if(dimension===1){const [p,q]=poly,d=delta(q,p);if(pointCmp(p,q)>=0)throw Error('segment ordering');const active=cs.filter(c=>(c[0]||c[1])&&!cmp(evaluate(c,p),ZERO)&&!cmp(evaluate(c,q),ZERO));if(!active.some(a=>active.some(b=>!det(a,b)&&a[0]*b[0]+a[1]*b[1]<0)))throw Error('segment not forced to line');const dot=c=>add(mul(rat(c[0]),d[0]),mul(rat(c[1]),d[1]));if(!cs.some(c=>!cmp(evaluate(c,p),ZERO)&&cmp(dot(c),ZERO)>0)||!cs.some(c=>!cmp(evaluate(c,q),ZERO)&&cmp(dot(c),ZERO)<0))throw Error('nonmaximal segment');}
  else{const p=poly[0],active=cs.filter(c=>(c[0]||c[1])&&!cmp(evaluate(c,p),ZERO)),rays=active.flatMap(c=>[[-c[1],c[0]],[c[1],-c[0]]]);if(!active.length||rays.some(d=>active.every(c=>c[0]*d[0]+c[1]*d[1]>=0)))throw Error('point has feasible direction');}
 });if(cmp(total,ONE)||cmp(decode(w.area),ONE))throw Error('area coverage');return {verified:true,source_matched:true,groups:groups.length,area_cells:dims[3],segment_cells:dims[2],point_cells:dims[1],empty_cells:dims[0],vertex_constraint_checks:checks,area:['1','1'],scope};
}
const api={editPlane,queryPlane,verifyPlane};globalThis.DenoiseEditPlane=api;if(typeof module!=='undefined')module.exports=api;
})();
