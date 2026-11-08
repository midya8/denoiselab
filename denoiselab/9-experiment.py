"""Training-only checkpoint selection, then separately frozen test evaluation."""
from pathlib import Path
import hashlib
import numpy as np
from .data import ROOT, load, partition, masks, hidden_mse
from .network import Denoiser, Adam
from .baselines import PCA, nearest, pixel_mean
from .io import read,save,digest

PROTOCOL=dict(schema=1,scale=16,features=64,hidden=64,code=16,initial_seed=23,
              shuffle_seed=41,corruption_seed=51,batch_size=128,epochs=60,
              candidate_epochs=[10,30,60],learning_rate=.003,l2=.00001,
              training_missing=32,validation_mask_seed=101,validation_missing=32,
              pca_ranks=[8,16,32],pca_ridges=[.001,.01,.1,1.],
              selection='minimum mean validation hidden-pixel MSE; earliest listed on ties')
EVALUATION=dict(schema=1,cases=[dict(name='random16',kind='random',missing=16,seed=201),
                dict(name='random32',kind='random',missing=32,seed=202),
                dict(name='random48',kind='random',missing=48,seed=203),
                dict(name='block16',kind='block',missing=16,seed=204)],
                examples='first 10 official test rows, no success filtering',
                metric='per-image hidden-pixel MSE then mean over 1797 images',
                compare='same masks and observed pixels for all methods; observed pixels copied unchanged')


def train(out):
    out=Path(out)
    x,y=load('train');fitids,valids=partition(x,y);fit=x[fitids];val=x[valids]
    protocol=dict(PROTOCOL,fit_indices=fitids.tolist(),validation_indices=valids.tolist(),
                  official_train_sha256=read(ROOT/'data/manifest.json')['optdigits.tra']['sha256'])
    save(out/'protocol.json',protocol)
    save(out/'evaluation-plan.json',EVALUATION)
    vm=masks(len(val),32,101)
    model=Denoiser(seed=23);opt=Adam(model.params,.003)
    shuffle=np.random.default_rng(41);corrupt=np.random.default_rng(51)
    history=[];best=None;bestscore=float('inf')
    initial=float(hidden_mse(val,model.predict(val,vm),vm).mean())
    for epoch in range(1,61):
        order=shuffle.permutation(len(fit))
        losses=[]
        for start in range(0,len(fit),128):
            xb=fit[order[start:start+128]]
            mask=np.zeros_like(xb,dtype=bool)
            for row in mask:row[corrupt.choice(64,32,replace=False)]=True
            loss,g=model.loss_grad(xb,mask,1e-5);opt.step(model.params,g);losses.append(loss)
        if epoch in (10,30,60):
            score=float(hidden_mse(val,model.predict(val,vm),vm).mean())
            history.append(dict(epoch=epoch,validation_hidden_mse=score,last_epoch_sampled_train_objective=float(np.mean(losses))))
            if score<bestscore:bestscore=score;best=model.to_dict();bestepoch=epoch
    pcas=[];bestp=None;bestpscore=float('inf');matched=None;matchedscore=float('inf')
    for rank in (8,16,32):
        p=PCA.fit(fit,rank,.001)
        for ridge in (.001,.01,.1,1.):
            p.ridge=ridge
            score=float(hidden_mse(val,p.predict(val,vm),vm).mean())
            pcas.append(dict(rank=rank,ridge=ridge,validation_hidden_mse=score))
            if score<bestpscore:bestpscore=score;bestp=p.to_dict()
            if rank==16 and score<matchedscore:matchedscore=score;matched=p.to_dict()
    selection=dict(schema=1,protocol_sha256=digest(protocol),initial_network_validation_hidden_mse=initial,
                   network_candidates=history,selected_epoch=bestepoch,pca_candidates=pcas,
                   selected_pca_rank=len(bestp['basis']),selected_pca_ridge=bestp['ridge'],
                   matched_rank=16,matched_ridge=matched['ridge'],
                   mean_validation_hidden_mse=float(hidden_mse(val,pixel_mean(val,vm,fit.mean(axis=0)),vm).mean()),
                   nearest_validation_hidden_mse=float(hidden_mse(val,nearest(val,vm,fit),vm).mean()))
    save(out/'network.json',best);save(out/'pca.json',bestp);save(out/'pca16.json',matched)
    selection['model_sha256']={n:digest(read(out/(n+'.json'))) for n in ('network','pca','pca16')}
    save(out/'selection.json',selection)
    return selection


def evaluate(directory):
    directory=Path(directory)
    protocol=read(directory/'protocol.json');selection=read(directory/'selection.json')
    plan=read(directory/'evaluation-plan.json')
    if plan!=EVALUATION or selection['protocol_sha256']!=digest(protocol):raise ValueError('protocol/plan mismatch')
    for name,expected in selection['model_sha256'].items():
        if digest(read(directory/(name+'.json')))!=expected:raise ValueError('selected model hash mismatch')
    # Test is only loaded here, after immutable selected artifacts have been checked.
    trainx,_=load('train');fit=trainx[protocol['fit_indices']]
    test,labels=load('test')
    trainset={hashlib.sha256((r*16).astype('uint8').tobytes()).hexdigest() for r in trainx}
    assert not any(hashlib.sha256((r*16).astype('uint8').tobytes()).hexdigest() in trainset for r in test)
    model=Denoiser.from_dict(read(directory/'network.json'))
    pca=PCA.from_dict(read(directory/'pca.json'));pca16=PCA.from_dict(read(directory/'pca16.json'))
    result=dict(schema=1,protocol_sha256=digest(protocol),selection_sha256=digest(selection),plan_sha256=digest(plan),
                rows=len(test),fit_rows=len(fit),validation_rows=len(protocol['validation_indices']),
                exact_train_test_feature_duplicates=0,cases=[])
    for cfg in plan['cases']:
        m=masks(len(test),cfg['missing'],cfg['seed'],cfg['kind'])
        nn,indices=nearest(test,m,fit,True)
        predictions=dict(network=model.predict(test,m),pca=pca.predict(test,m),pca16=pca16.predict(test,m),
                         pixel_mean=pixel_mean(test,m,fit.mean(axis=0)),nearest=nn)
        scores={}
        for name,pred in predictions.items():
            if not np.array_equal(pred[~m],test[~m]):raise ValueError('observed pixel changed')
            row=hidden_mse(test,pred,m)
            scores[name]=dict(hidden_mse=float(row.mean()),row_hidden_mse=row.tolist(),
                             hidden_rmse=float(np.sqrt(row.mean())),observed_max_absolute_error=0.,
                             all_pixel_mse=float(np.mean((pred-test)**2)))
        examples=[]
        for i in range(10):
            examples.append(dict(row_id=i,label=int(labels[i]),clean=test[i].tolist(),mask=m[i].astype(int).tolist(),
                observed=np.where(m[i],0,test[i]).tolist(),predictions={name:pred[i].tolist() for name,pred in predictions.items()},
                nearest_fit_original_row=int(protocol['fit_indices'][indices[i]])))
        result['cases'].append(dict(**cfg,methods=scores,examples=examples,mask_sha256=hashlib.sha256(m.astype('uint8').tobytes()).hexdigest()))
    save(directory/'report.json',result)
    return result
