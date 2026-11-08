"""Exact two-coordinate nearest-neighbor slices by closed halfplane clipping.

Power-diagram geometry, not a new search method or general robustness bound.
The verifier checks supporting halfplanes / infeasibility combinations rather
than rerunning the clipping algorithm.
"""
from fractions import Fraction as F
from itertools import combinations
from functools import reduce
from math import gcd
import numpy as np
from .network import inputs
from .edit_path import _counts, _same, pair, parameter

SCOPE='Exact rational two-observed-coordinate slice for fixed quantized inputs and mask. Not a general robustness radius or a binary64 tie guarantee.'
DOMAIN=[(1,0,0),(-1,0,1),(0,1,0),(0,-1,1)]


def _source(fit,pixels,mask,input_pixels,scalar=False):
    f=np.asarray(fit,dtype=float)
    if f.ndim!=2 or not 1<=len(f)<=10000 or not 2<=f.shape[1]<=64:raise ValueError('fit must have 1..10000 rows and 2..64 columns')
    x,m,_=inputs([pixels],[mask],f.shape[1]);obs=~m[0]
    if not isinstance(input_pixels,list) or len(input_pixels)!=2 or any(type(i)!=int or not 0<=i<f.shape[1] or not obs[i] for i in input_pixels) or len(set(input_pixels))!=2:raise ValueError('two distinct observed coordinates required')
    clean=np.where(obs,x[0],0);qc=_counts(clean);fc=_counts(f);a,b=input_pixels
    if scalar:
        lines=[]
        for row in fc:
            c=int(row[a])**2+int(row[b])**2
            for j in range(f.shape[1]):
                if obs[j] and j not in input_pixels:c+=(int(row[j])-int(qc[j]))**2
            lines.append([-32*int(row[a]),-32*int(row[b]),c])
    else:
        other=obs.copy();other[input_pixels]=False
        intercept=np.sum((fc[:,other]-qc[other])**2,axis=1)+fc[:,a]**2+fc[:,b]**2
        lines=[[-32*int(row[a]),-32*int(row[b]),int(c)] for row,c in zip(fc,intercept)]
    groups={}
    for i,line in enumerate(lines):
        key=tuple(line[:2]);old=groups.get(key)
        if old is None or line[2]<old['line'][2]:groups[key]=dict(line=line,index=i,tied_indices=[i])
        elif line[2]==old['line'][2]:old['tied_indices'].append(i)
    return dict(schema='denoiselab-edit-plane-v1',input_pixels=input_pixels.copy(),mask=m[0].astype(int).tolist(),observed=clean.tolist(),fit_count=len(f),row_lines=lines,groups=[groups[k] for k in sorted(groups,reverse=True)],hidden_targets_used=False,refitted=False,scope=SCOPE)


def _eval(c,p):return c[0]*p[0]+c[1]*p[1]+c[2]
def _cross(a,b):return a[0]*b[1]-a[1]*b[0]
def _sub(a,b):return (a[0]-b[0],a[1]-b[1])
def _area(poly):return sum((_cross(p,q) for p,q in zip(poly,poly[1:]+poly[:1])),F(0))/2


def _normalize(poly):
    out=[]
    for p in poly:
        if not out or p!=out[-1]:out.append(p)
    if len(out)>1 and out[0]==out[-1]:out.pop()
    if len(out)<2:return out
    if _area(out)==0:return sorted({min(out),max(out)})
    while len(out)>2:
        keep=[p for i,p in enumerate(out) if _cross(_sub(p,out[i-1]),_sub(out[(i+1)%len(out)],p))!=0]
        if len(keep)==len(out):break
        out=keep
    if _area(out)<0:out.reverse()
    k=out.index(min(out));return out[k:]+out[:k]


def _clip(poly,c):
    if not poly:return []
    out=[];s=poly[-1];sv=_eval(c,s)
    for e in poly:
        ev=_eval(c,e)
        if (sv<0<=ev) or (ev<0<=sv):
            t=sv/(sv-ev);out.append((s[0]+t*(e[0]-s[0]),s[1]+t*(e[1]-s[1])))
        if ev>=0:out.append(e)
        s,sv=e,ev
    return _normalize(out)


def _constraints(groups,k):
    own=groups[k]['line']
    return DOMAIN+[tuple(a-b for a,b in zip(g['line'],own)) for g in groups]


def _empty_certificate(constraints,previous,cut):
    # Helly/Farkas in dimension two. Search the new cut and constraints active
    # on the previous bounded cell, not all triples from all source rows.
    ids=[i for i in range(cut) if any(_eval(constraints[i],p)==0 for p in previous)]+[cut]
    for size in [1,2,3]:
        for chosen in combinations(ids,size):
            if cut not in chosen:continue
            cs=[constraints[i] for i in chosen]
            if size==1:
                weights=[1]
            elif size==2:
                p,q=cs
                if _cross(p,q)!=0:continue
                axis=0 if p[0] or q[0] else 1
                weights=[abs(q[axis]),abs(p[axis])]
            else:
                p,q,r=cs;weights=[_cross(q,r),_cross(r,p),_cross(p,q)]
                if all(w<=0 for w in weights):weights=[-w for w in weights]
            if not any(weights) or any(w<0 for w in weights):continue
            total=[sum(w*c[j] for w,c in zip(weights,cs)) for j in range(3)]
            if total[:2]==[0,0] and total[2]<0:
                div=reduce(gcd,weights);return dict(constraints=list(chosen),weights=[str(w//div) for w in weights])
    raise ArithmeticError('empty cell has no recovered infeasibility combination')


def edit_plane(fit,pixels,mask,input_pixels):
    out=_source(fit,pixels,mask,input_pixels);cells=[]
    square=[(F(0),F(0)),(F(1),F(0)),(F(1),F(1)),(F(0),F(1))]
    for k,g in enumerate(out['groups']):
        constraints=_constraints(out['groups'],k);poly=square.copy();proof=None
        for i,c in enumerate(constraints):
            previous=poly;poly=_clip(poly,c)
            if not poly:
                proof=_empty_certificate(constraints,previous,i);break
        dimension=-1 if not poly else (0 if len(poly)==1 else (1 if len(poly)==2 else 2))
        cells.append(dict(group=k,dimension=dimension,vertices=[[pair(x),pair(y)] for x,y in poly],area=pair(_area(poly)),empty_proof=proof))
    out['cells']=cells;out['area']=pair(sum((F(*map(int,c['area'])) for c in cells),F(0)))
    if out['area']!=['1','1']:raise ArithmeticError('cell area does not cover square')
    return out


def query_plane(witness,u,v):
    x,y=parameter(u),parameter(v);scores=[_eval(c,(x,y)) for c in witness['row_lines']];best=min(scores);tied=[i for i,s in enumerate(scores) if s==best]
    return dict(parameters=[pair(x),pair(y)],index=tied[0],tied_indices=tied,reduced_distance=pair(best),squared_distance=pair((256*(x*x+y*y)+best)/256))


def verify_plane(witness,fit,pixels,mask,input_pixels):
    """Check intended source and maximal cells without rerunning clipping.

    Mathematical/source consistency only, not authentication or a parser
    security boundary. Independently supply the intended source.
    """
    try:
        expected=_source(fit,pixels,mask,input_pixels,scalar=True)
        for key,value in expected.items():
            if not _same(witness[key],value):raise ValueError('source mismatch: '+key)
        cells=witness['cells'];groups=expected['groups']
        if type(cells)!=list or len(cells)!=len(groups):raise ValueError('missing cells')
        def rational(v):
            if type(v)!=list or len(v)!=2 or any(type(s)!=str or len(s)>160 for s in v):raise ValueError('rational encoding')
            q=F(int(v[0]),int(v[1]))
            if pair(q)!=v:raise ValueError('noncanonical rational')
            return q
        total=F(0);vertex_checks=0;dims={-1:0,0:0,1:0,2:0}
        for k,cell in enumerate(cells):
            if type(cell['group'])!=int or cell['group']!=k or type(cell['dimension'])!=int or cell['dimension'] not in dims:raise ValueError('cell index/dimension')
            dimension=cell['dimension'];dims[dimension]+=1;cs=_constraints(groups,k)
            if type(cell['vertices'])!=list or len(cell['vertices'])>len(cs):raise ValueError('vertex size')
            poly=[tuple(rational(q) for q in p) for p in cell['vertices']]
            if any(len(p)!=2 for p in poly) or len(set(poly))!=len(poly):raise ValueError('vertices')
            area=rational(cell['area'])
            if area!=_area(poly):raise ValueError('false area')
            total+=area
            if dimension==-1:
                if poly or area:raise ValueError('empty geometry')
                proof=cell['empty_proof'];ids=proof['constraints'];weights=[int(w) for w in proof['weights']]
                if not 1<=len(ids)<=3 or len(ids)!=len(weights) or len(set(ids))!=len(ids) or any(type(i)!=int or not 0<=i<len(cs) for i in ids) or not any(weights) or any(w<0 for w in weights):raise ValueError('empty proof encoding')
                sums=[sum(w*cs[i][j] for i,w in zip(ids,weights)) for j in range(3)]
                if sums[:2]!=[0,0] or sums[2]>=0:raise ValueError('false infeasibility combination')
                continue
            if cell['empty_proof'] is not None or len(poly)!=(dimension+1 if dimension<2 else len(poly)) or (dimension==2 and len(poly)<3):raise ValueError('dimension encoding')
            if any(_eval(c,p)<0 for p in poly for c in cs):raise ValueError('infeasible vertex')
            vertex_checks+=len(poly)*len(cs)
            if dimension==2:
                if area<=0:raise ValueError('orientation/area')
                for i,p in enumerate(poly):
                    q=poly[(i+1)%len(poly)];edge=_sub(q,p)
                    if any(_cross(edge,_sub(r,p))<0 for r in poly) or _cross(edge,_sub(poly[(i+2)%len(poly)],q))<=0:raise ValueError('nonconvex/noncanonical polygon')
                    if not any((c[0] or c[1]) and _eval(c,p)==0 and _eval(c,q)==0 for c in cs):raise ValueError('unsupported polygon edge')
            elif dimension==1:
                p,q=poly;d=_sub(q,p)
                if p>=q:raise ValueError('segment ordering')
                active=[c for c in cs if (c[0] or c[1]) and _eval(c,p)==0 and _eval(c,q)==0]
                if not any(_cross(a,b)==0 and a[0]*b[0]+a[1]*b[1]<0 for a in active for b in active):raise ValueError('segment not forced to a line')
                if not any(_eval(c,p)==0 and c[0]*d[0]+c[1]*d[1]>0 for c in cs) or not any(_eval(c,q)==0 and c[0]*d[0]+c[1]*d[1]<0 for c in cs):raise ValueError('segment not maximal')
            else:
                p=poly[0];active=[c for c in cs if (c[0] or c[1]) and _eval(c,p)==0]
                rays=[(sign*-c[1],sign*c[0]) for c in active for sign in [1,-1]]
                if not active or any(all(c[0]*d[0]+c[1]*d[1]>=0 for c in active) for d in rays):raise ValueError('point has a feasible direction')
        if total!=1 or rational(witness['area'])!=1:raise ValueError('area coverage')
    except (KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError) as e:raise ValueError('invalid plane witness') from e
    return dict(verified=True,source_matched=True,groups=len(groups),area_cells=dims[2],segment_cells=dims[1],point_cells=dims[0],empty_cells=dims[-1],vertex_constraint_checks=vertex_checks,area=['1','1'],scope=SCOPE)
