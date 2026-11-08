"use strict";
(() => {
try {
const $=id=>document.getElementById(id),data=DENOISE_DATA,I=DenoiseInference;
let mask=[],current,selection=0,lastResult;
const methods=["network","pca","nearest","pixel_mean"];
for(let i=0;i<10;i++)$("row").add(new Option("Row "+i+" / digit "+data.cases[0].examples[i].label,i));
for(let i=0;i<64;i++)$("pixel").add(new Option("Row "+(1+Math.floor(i/8))+", column "+(1+i%8),i));
function draw(id,values,missing=null,selected=null){const c=$(id),ctx=c.getContext("2d"),cell=32;ctx.clearRect(0,0,256,256);for(let i=0;i<64;i++){const v=Math.round(Math.max(0,Math.min(1,values[i]))*255);ctx.fillStyle=missing&&missing[i]?"#36647e":`rgb(${v},${v},${v})`;ctx.fillRect(i%8*cell,Math.floor(i/8)*cell,cell,cell);ctx.strokeStyle="#8898a344";ctx.lineWidth=1;ctx.strokeRect(i%8*cell,Math.floor(i/8)*cell,cell,cell);}if(selected!==null){ctx.strokeStyle="#ffcc45";ctx.lineWidth=3;ctx.strokeRect(selected%8*cell+2,Math.floor(selected/8)*cell+2,cell-4,cell-4);}}
function update(){
  const pixels=current.clean,nn=I.nearest(pixels,mask,data.fit);
  const predictions={network:I.network(pixels,mask,data.network),pca:I.pca(pixels,mask,data.pca),nearest:nn.prediction,pixel_mean:I.mean(pixels,mask,data.pca.mean)};
  const scores={};for(const name of methods){scores[name]=I.score(pixels,predictions[name],mask);draw(name,predictions[name]);$("score-"+name).textContent=scores[name]===null?"No hidden pixels to score.":"Hidden-pixel MSE: "+scores[name].toFixed(7);}
  draw("clean",pixels);draw("masked",pixels.map((v,i)=>mask[i]?0:v),mask,selection);
  $("status").textContent=mask.filter(Boolean).length+" of 64 pixels missing. Selected row "+(1+Math.floor(selection/8))+", column "+(1+selection%8)+" is "+(mask[selection]?"hidden.":"visible.");
  $("neighbor").textContent="Nearest fitting example: original training row "+data.fit_indices[nn.index]+". Lowest fitting index breaks a distance tie. No labels enter that search.";
  lastResult={schema:1,row_id:current.row_id,mask:mask.slice(),observed:pixels.map((v,i)=>mask[i]?0:v),predictions,hidden_pixel_mse:scores,nearest_fit_original_row:data.fit_indices[nn.index]};
  $("witness").textContent=JSON.stringify(lastResult,null,2);$("pixel").value=String(selection);$("copy-status").textContent="";
}
function restore(){const cfg=data.cases.find(c=>c.name===$("pattern").value);current=cfg.examples[Number($("row").value)];mask=current.mask.slice();update();}
$("row").onchange=restore;$("pattern").onchange=restore;$("reset").onclick=restore;
$("pixel").onchange=()=>{selection=Number($("pixel").value);update();};
$("toggle").onclick=()=>{mask[selection]=mask[selection]?0:1;update();};
$("hide").onclick=()=>{mask.fill(1);update();};$("reveal").onclick=()=>{mask.fill(0);update();};
$("masked").onclick=e=>{const r=e.target.getBoundingClientRect();selection=Math.max(0,Math.min(63,Math.floor((e.clientY-r.top)/r.height*8)*8+Math.floor((e.clientX-r.left)/r.width*8)));mask[selection]=mask[selection]?0:1;update();};
$("masked").onkeydown=e=>{const d={ArrowLeft:-1,ArrowRight:1,ArrowUp:-8,ArrowDown:8};if(e.key in d){e.preventDefault();selection=Math.max(0,Math.min(63,selection+d[e.key]));update();}else if(e.key===" "){e.preventDefault();mask[selection]=mask[selection]?0:1;update();}};
$("copy").onclick=async()=>{try{await navigator.clipboard.writeText(JSON.stringify(lastResult,null,2));$("copy-status").textContent="Copied current result JSON.";}catch{$("copy-status").textContent="Copy is unavailable. Use Download result JSON instead.";}};
$("download").onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(lastResult,null,2)+"\n"],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download="denoiselab-row"+current.row_id+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
for(const c of data.cases){const tr=document.createElement("tr");for(const [i,text] of [c.name,...methods.map(m=>c.methods[m].hidden_mse.toFixed(7))].entries()){const td=document.createElement(i===0?"th":"td");if(i===0)td.scope="row";else td.dataset.label=["Pattern","Neural","PCA","Neighbor","Mean"][i];td.textContent=text;tr.append(td);}$("audit-body").append(tr);}
$("hashes").textContent=JSON.stringify(data.source_hashes,null,2);restore();
}catch(e){document.getElementById("error").hidden=false;document.getElementById("error").textContent="Inspector could not load: "+e.message;}
})();
