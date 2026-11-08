"""Fit-only controls. All methods see only observed test pixels; output keeps them."""
import numpy as np
from .network import inputs, matrix


def pixel_mean(x, mask, mean):
    x,m,_ = inputs(x,mask)
    mean = np.asarray(mean, dtype=float)
    if mean.shape != (x.shape[1],) or not np.isfinite(mean).all():
        raise ValueError('invalid pixel mean')
    return np.where(m,np.clip(mean,0,1),x)


class PCA:
    def __init__(self, mean, basis, ridge):
        self.mean = np.asarray(mean,dtype=float).copy()
        self.basis = matrix(basis).copy()
        if self.mean.shape != (self.basis.shape[1],) or not np.isfinite(self.mean).all() or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError('invalid PCA parameters or ridge')
        self.ridge = float(ridge)

    @classmethod
    def fit(cls,x,rank=16,ridge=.1):
        x=matrix(x)
        if type(rank) is not int or not 1 <= rank <= min(x.shape):
            raise ValueError('invalid PCA rank')
        mean=x.mean(axis=0)
        _,_,vt=np.linalg.svd(x-mean,full_matrices=False)
        # Sign convention fixes ordinary SVD sign ambiguity for portable artifacts.
        basis=vt[:rank].copy()
        for v in basis:
            if v[np.argmax(np.abs(v))] < 0:v *= -1
        return cls(mean,basis,ridge)

    def predict(self,x,mask):
        x,m,_=inputs(x,mask,len(self.mean))
        result=np.empty_like(x)
        for i,(row,hidden) in enumerate(zip(x,m)):
            b=self.basis[:,~hidden].T
            a=np.linalg.solve(b.T@b+self.ridge*np.eye(len(self.basis)), b.T@(row[~hidden]-self.mean[~hidden]))
            result[i]=np.where(hidden,np.clip(self.mean+a@self.basis,0,1),row)
        return result

    def to_dict(self):
        return dict(schema=1,mean=self.mean.tolist(),basis=self.basis.tolist(),ridge=self.ridge)

    @classmethod
    def from_dict(cls,v):
        if not isinstance(v,dict) or set(v)!={'schema','mean','basis','ridge'} or v['schema']!=1:
            raise ValueError('invalid PCA schema')
        return cls(v['mean'],v['basis'],v['ridge'])


def nearest(x,mask,fit,return_indices=False):
    fit=matrix(fit)
    x,m,_=inputs(x,mask,fit.shape[1])
    if np.any(fit<0) or np.any(fit>1):raise ValueError('invalid fitting pixels')
    # Independent per-query direct differences avoids cancellation-driven tie errors.
    indices=np.array([int(np.argmin(np.sum((fit[:,~h]-r[~h])**2,axis=1))) for r,h in zip(x,m)])
    out=np.where(m,fit[indices],x)
    return (out,indices) if return_indices else out
