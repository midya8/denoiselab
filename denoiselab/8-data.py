"""Original UCI numeric split, content hashes and deterministic training-only split."""
from pathlib import Path
import hashlib
import numpy as np
from .io import read
ROOT=Path(__file__).parent


def load(part='train'):
    if part not in ('train','test'):raise ValueError('partition must be train or test')
    path=ROOT/'data'/('optdigits.tra' if part=='train' else 'optdigits.tes')
    manifest=read(ROOT/'data/manifest.json')
    raw=path.read_bytes()
    expected=manifest[path.name]['sha256']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('original numeric data hash mismatch')
    a=np.loadtxt(path,delimiter=',',dtype=int)
    n=3823 if part=='train' else 1797
    if a.shape!=(n,65) or (a[:,:64]<0).any() or (a[:,:64]>16).any() or (a[:,64]<0).any() or (a[:,64]>9).any():raise ValueError('invalid original data')
    return a[:,:64]/16.,a[:,64]


def partition(x,y):
    fit=[];val=[]
    for label in range(10):
        ids=np.flatnonzero(y==label).tolist()
        ids.sort(key=lambda i:(hashlib.sha256((x[i]*16).astype('uint8').tobytes()).hexdigest(),i))
        val+=ids[:len(ids)//5];fit+=ids[len(ids)//5:]
    return np.array(sorted(fit)),np.array(sorted(val))


def masks(n,missing=32,seed=101,kind='random'):
    if type(n) is not int or n<1 or type(missing) is not int or not 1<=missing<=64 or kind not in ('random','block'):
        raise ValueError('invalid mask config')
    rng=np.random.default_rng(seed)
    out=np.zeros((n,64),dtype=bool)
    if kind=='random':
        for row in out:row[rng.choice(64,missing,replace=False)]=True
    else:
        if missing!=16:raise ValueError('block uses exactly 16 pixels')
        for row in out:
            a,b=rng.integers(0,5,size=2);row.reshape(8,8)[a:a+4,b:b+4]=True
    return out


def hidden_mse(x,pred,mask):
    if x.shape!=pred.shape or x.shape!=mask.shape or not np.isfinite(pred).all() or not mask.any():raise ValueError('invalid scored arrays')
    count=mask.sum(axis=1)
    if (count==0).any():raise ValueError('every scored row needs hidden pixels')
    return np.sum((pred-x)**2*mask,axis=1)/count
