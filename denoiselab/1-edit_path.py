"""Exact scalar nearest-neighbor edit paths for quantized block-count inputs.

The common quadratic cancels. Rational lower envelopes use established
squared-distance transform algebra, not a new nearest-neighbor algorithm.
"""
from fractions import Fraction
import re
import numpy as np
from .network import inputs


def parameter(value):
    if not isinstance(value,str) or len(value)>160 or not re.fullmatch(r'(?:[0-9]+(?:\.[0-9]{1,80})?|[0-9]+/[1-9][0-9]*)',value):
        raise ValueError('parameter must be a bounded unsigned decimal or numerator/denominator string')
    q=Fraction(value)
    if not 0<=q<=1:raise ValueError('parameter must be within [0,1]')
    return q


def pair(q):return [str(q.numerator),str(q.denominator)]
def unpack(v):return Fraction(int(v[0]),int(v[1]))


def _counts(values):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all() or np.any(x<0) or np.any(x>1) or np.any(x*16!=np.floor(x*16)):
        raise ValueError('supported source values must be exact multiples of 1/16 within [0,1]')
    return (x*16).astype(np.int64)


def _winner(lines,q):
    scores=[int(l['slope'])*q+int(l['intercept']) for l in lines];best=min(scores);tied=[i for i,v in enumerate(scores) if v==best]
    return dict(index=tied[0],tied_indices=tied,reduced_distance=pair(best),squared_distance=pair((256*q*q+best)/256))


def edit_path(fit,pixels,mask,input_pixel):
    fit=np.asarray(fit,dtype=float)
    if fit.ndim!=2 or not 1<=len(fit)<=10000 or not 1<=fit.shape[1]<=64:raise ValueError('fit must have 1..10000 rows and 1..64 columns')
    d=fit.shape[1];x,m,_=inputs([pixels],[mask],d);observed=~m[0]
    if isinstance(input_pixel,bool) or not isinstance(input_pixel,int) or not 0<=input_pixel<d or not observed[input_pixel]:raise ValueError('input_pixel must identify an observed coordinate')
    # Erasure before quantization: hidden supplied values never constrain this path.
    erased=np.where(observed,x[0],0);qcounts=_counts(erased);fcounts=_counts(fit);other=observed.copy();other[input_pixel]=False
    intercepts=np.sum((fcounts[:,other]-qcounts[other])**2,axis=1)+fcounts[:,input_pixel]**2
    lines=[dict(slope=int(-32*row[input_pixel]),intercept=int(b)) for row,b in zip(fcounts,intercepts)]
    groups=[]
    for count in sorted(set(fcounts[:,input_pixel].tolist())):
        indices=np.flatnonzero(fcounts[:,input_pixel]==count);best=int(np.min(intercepts[indices]));tied=[int(i) for i in indices if intercepts[i]==best];groups.append(dict(count=count,slope=-32*count,intercept=best,index=tied[0],tied_indices=tied))
    hull=[];starts=[]
    for line in groups:
        while hull:
            crossing=Fraction(line['intercept']-hull[-1]['intercept'],hull[-1]['slope']-line['slope'])
            if len(hull)==1 or crossing>starts[-1]:break
            hull.pop();starts.pop()
        starts.append(None if not hull else crossing);hull.append(line)
    bounds=sorted({Fraction(0),Fraction(1),*(s for s in starts[1:] if 0<s<1)})
    segments=[]
    for lo,hi in zip(bounds,bounds[1:]):
        middle=(lo+hi)/2;winner=_winner(lines,middle);index=winner['index'];chosen=lines[index]
        minimum=None
        for candidate in lines:
            for endpoint in [lo,hi]:
                slack=(candidate['slope']-chosen['slope'])*endpoint+candidate['intercept']-chosen['intercept']
                if slack<0:raise ArithmeticError('hull dominance failed')
                minimum=slack if minimum is None else min(minimum,slack)
        segments.append(dict(left=pair(lo),right=pair(hi),index=index,tied_indices=winner['tied_indices'],slope=chosen['slope'],intercept=chosen['intercept'],minimum_endpoint_slack=pair(minimum)))
    # Evaluate all rows at every endpoint, including lines pruned by the hull:
    # a line may win only a point tie and no nonempty interval.
    points=[dict(parameter=pair(q),**_winner(lines,q)) for q in bounds]
    return dict(schema='denoiselab-edit-path-v1',input_pixel=input_pixel,mask=m[0].astype(int).tolist(),observed=erased.tolist(),fit_count=len(fit),groups=groups,row_lines=lines,segments=segments,points=points,dominance_checks=2*len(lines)*len(segments),hidden_targets_used=False,refitted=False,scope='Exact rational single-observed-coordinate path for fixed quantized inputs and mask. Not a multicoordinate radius or a binary64 tie guarantee.')


def query_path(witness,value):
    q=parameter(value);out=_winner(witness['row_lines'],q)
    return dict(parameter=pair(q),**out)


def _same(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(_same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(_same(v,w) for v,w in zip(a,b))
    return a==b


def verify_path(witness,fit,pixels,mask,input_pixel):
    """Check source and interval proof without recomputing the lower hull.

    A self-consistent edited source is not the original source. Callers must
    supply their intended fitting rows and fixed query, not take them on trust
    from a certificate. Raises ValueError on invalid/tampered witnesses.
    """
    try:
        f=np.asarray(fit,dtype=float)
        if f.ndim!=2 or not 1<=len(f)<=10000 or not 1<=f.shape[1]<=64:raise ValueError('source dimensions')
        x,m,_=inputs([pixels],[mask],f.shape[1]);obs=~m[0]
        if type(input_pixel)!=int or not 0<=input_pixel<f.shape[1] or not obs[input_pixel]:raise ValueError('observed input required')
        clean=np.where(obs,x[0],0);qc=_counts(clean);fc=_counts(f);lines=[]
        for row in fc:
            b=int(row[input_pixel])**2
            for j in range(f.shape[1]):
                if obs[j] and j!=input_pixel:b+=(int(row[j])-int(qc[j]))**2
            lines.append(dict(slope=-32*int(row[input_pixel]),intercept=b))
        expected=dict(schema='denoiselab-edit-path-v1',input_pixel=input_pixel,mask=m[0].astype(int).tolist(),observed=clean.tolist(),fit_count=len(f),row_lines=lines,hidden_targets_used=False,refitted=False,scope='Exact rational single-observed-coordinate path for fixed quantized inputs and mask. Not a multicoordinate radius or a binary64 tie guarantee.')
        for key,v in expected.items():
            if not _same(witness[key],v):raise ValueError('source mismatch: '+key)
        # Independently rebuild slope groups from supplied source coefficients.
        group_map={}
        for i,line in enumerate(lines):
            count=-line['slope']//32;b=line['intercept']
            if count not in group_map or b<group_map[count]['intercept']:group_map[count]=dict(count=count,**line,index=i,tied_indices=[i])
            elif b==group_map[count]['intercept']:group_map[count]['tied_indices'].append(i)
        if not _same(witness['groups'],[group_map[k] for k in sorted(group_map)]):raise ValueError('group mismatch')
        segments=witness['segments'];points=witness['points']
        if not isinstance(segments,list) or not 1<=len(segments)<=17 or len(points)!=len(segments)+1:raise ValueError('partition size')
        def rational(v):
            if not isinstance(v,list) or len(v)!=2 or any(type(s)!=str or len(s)>160 or not re.fullmatch(r'-?(?:0|[1-9][0-9]*)',s) for s in v):raise ValueError('rational encoding')
            q=unpack(v)
            if pair(q)!=v:raise ValueError('noncanonical rational')
            return q
        previous=Fraction(0)
        for k,segment in enumerate(segments):
            lo=rational(segment['left']);hi=rational(segment['right'])
            if lo!=previous or not 0<=lo<hi<=1:raise ValueError('partition gap or order')
            previous=hi;mid=(lo+hi)/2;winner=_winner(lines,mid);index=winner['index'];chosen=lines[index]
            if type(segment['index'])!=int or segment['index']!=index or segment['tied_indices']!=winner['tied_indices']:raise ValueError('interval tie mismatch')
            if segment['slope']!=chosen['slope'] or segment['intercept']!=chosen['intercept']:raise ValueError('chosen coefficient mismatch')
            slack=min((c['slope']-chosen['slope'])*q+c['intercept']-chosen['intercept'] for c in lines for q in [lo,hi])
            if slack<0 or rational(segment['minimum_endpoint_slack'])!=slack:raise ValueError('negative or false dominance slack')
            for pos,q in [(k,lo),(k+1,hi)]:
                if not _same(points[pos],dict(parameter=pair(q),**_winner(lines,q))):raise ValueError('point tie mismatch')
            if k and segments[k-1]['index']==index:raise ValueError('nonmaximal adjacent intervals')
        if previous!=1 or witness['dominance_checks']!=2*len(f)*len(segments):raise ValueError('coverage or check count')
    except (KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError) as e:raise ValueError('invalid path witness') from e
    return dict(verified=True,source_matched=True,intervals=len(segments),points=len(points),dominance_checks=2*len(f)*len(segments),scope=expected['scope'])
