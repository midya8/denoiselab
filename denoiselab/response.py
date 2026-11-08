"""Fixed-mask input response. Local derivatives are not finite-edit guarantees."""
import numpy as np
from .network import inputs,Denoiser
from .baselines import PCA,nearest,pixel_mean

def network_jacobian(model,pixels,mask):
    x,m,enc=inputs([pixels],[mask],model.features);acts=model._forward(enc);d=model.features
    jac=np.concatenate([np.diag((~m[0]).astype(float)),np.zeros((d,d))])
    for layer in range(4):
        out=acts[layer+1][0];derivative=out*(1-out) if layer==3 else 1-out*out
        jac=derivative[:,None]*(model.params[2*layer].T@jac)
    jac[~m[0]]=np.eye(d)[~m[0]]
    return dict(prediction=model.predict(x,m)[0].tolist(),jacobian=jac.tolist(),undefined_output_rows=[],scope='Local derivative of copied-visible / predicted-hidden outputs at fixed mask.')

def pca_jacobian(model,pixels,mask):
    x,m,_=inputs([pixels],[mask],len(model.mean));hidden=m[0];observed=~hidden;d=len(model.mean);basis=model.basis[:,observed].T
    gram=basis.T@basis+model.ridge*np.eye(len(model.basis));rhs=basis.T@(x[0,observed]-model.mean[observed]);coef=np.linalg.solve(gram,rhs);raw=model.mean+coef@model.basis
    response=np.linalg.solve(gram,model.basis[:,observed]);jac=np.eye(d);jac[hidden]=0
    if observed.any():jac[np.ix_(hidden,observed)]=(model.basis[:,hidden].T@response)
    outside=hidden&((raw<0)|(raw>1));boundary=hidden&((raw==0)|(raw==1));jac[outside]=0
    rows=jac.tolist()
    for j in np.flatnonzero(boundary):
        if np.any(jac[j]!=0):rows[j]=[None]*d
        else:boundary[j]=False
    return dict(prediction=np.where(hidden,np.clip(raw,0,1),x[0]).tolist(),jacobian=rows,raw_prediction=raw.tolist(),coefficients=coef.tolist(),clipped_output_rows=np.flatnonzero(outside).tolist(),undefined_output_rows=np.flatnonzero(boundary).tolist(),solve_residual=float(np.linalg.norm(gram@response-model.basis[:,observed])),scope='Observed-only ridge solve, followed by clipping. Derivative undefined on a clipping boundary.')

def response(network,pca,fit,pixels,mask,edited):
    if not isinstance(pixels,list) or not isinstance(edited,list) or not isinstance(mask,list):raise ValueError('pixels, mask and edited must be lists')
    x,m,_=inputs([pixels],[mask],network.features);edit,em,_=inputs([edited],[mask],network.features)
    if len(pca.mean)!=network.features:raise ValueError('model widths differ')
    # Hidden supplied values are never inputs, outputs or evaluation targets here.
    observed=np.where(m[0],0.,x[0]);changed=np.where(m[0],0.,edit[0]);delta=changed-observed
    n=network_jacobian(network,observed.tolist(),mask);p=pca_jacobian(pca,observed.tolist(),mask)
    old_nn,oi=nearest([observed],m,fit,True);new_nn,ni=nearest([changed],m,fit,True)
    before={'network':n['prediction'],'pca':p['prediction'],'nearest':old_nn[0].tolist(),'pixel_mean':pixel_mean([observed],m,pca.mean)[0].tolist()}
    after={'network':network.predict([changed],m)[0].tolist(),'pca':pca.predict([changed],m)[0].tolist(),'nearest':new_nn[0].tolist(),'pixel_mean':pixel_mean([changed],m,pca.mean)[0].tolist()}
    methods={}
    for name,local in [('network',n),('pca',p)]:
        tangent=[None if row[0] is None else before[name][j]+float(np.dot(row,delta)) for j,row in enumerate(local['jacobian'])];residual=[None if v is None else after[name][j]-v for j,v in enumerate(tangent)]
        methods[name]=dict(local=local,edited_prediction=after[name],tangent_prediction=tangent,finite_minus_tangent=residual,hidden_residual_l2=None if any(residual[j] is None for j in np.flatnonzero(m[0])) else float(np.linalg.norm(np.array([residual[j] for j in np.flatnonzero(m[0])]))))
    for name in ['nearest','pixel_mean']:
        # Not a claimed NN Jacobian: this explicitly freezes the original retrieval index.
        frozen=np.where(m[0],before[name],changed).tolist();methods[name]=dict(prediction=before[name],edited_prediction=after[name],frozen_hidden_prediction=frozen,finite_minus_frozen=(np.array(after[name])-frozen).tolist())
    fit=np.asarray(fit,dtype=float);distances=np.sum((fit[:,~m[0]]-observed[~m[0]])**2,axis=1);ordered=np.argsort(distances,kind='stable')
    return dict(schema='denoiselab-response-v1',mask=m[0].astype(int).tolist(),observed=observed.tolist(),edited_observed=changed.tolist(),delta=delta.tolist(),methods=methods,nearest_before_index=int(oi[0]),nearest_after_index=int(ni[0]),nearest_switched=bool(oi[0]!=ni[0]),nearest_before_distance=float(distances[oi[0]]),nearest_runner_up_gap=None if len(fit)<2 else float(distances[ordered[1]]-distances[ordered[0]]),hidden_targets_used=False,refitted=False,scope='Fixed mask, fixed models, observed values only. Derivatives are local, not causal or robustness certificates.')
