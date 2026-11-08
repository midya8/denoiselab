import unittest
import numpy as np
from denoiselab.network import Denoiser,Adam,inputs
from denoiselab.baselines import PCA,pixel_mean,nearest
from denoiselab.data import masks,hidden_mse

class GradientTests(unittest.TestCase):
    def test_all_parameters_independent_objective(self):
        probes=0
        for seed in [1,7]:
            for l2 in [0.,.07]:
                m=Denoiser(features=3,hidden=3,code=2,seed=seed)
                x=np.random.default_rng(seed+2).random((2,3));mask=np.array([[1,0,1],[0,1,0]],dtype=bool)
                # Separate expression for objective, not loss_grad/_forward.
                def reference():
                    h=np.concatenate([np.where(mask,0,x),mask],axis=1)
                    for i in range(0,8,2):
                        a=h@m.params[i]+m.params[i+1]
                        h=np.tanh(a) if i<6 else 1/(1+np.exp(-a))
                    return np.sum((h-x)**2*mask)/mask.sum()+l2/2*sum(np.sum(p*p) for p in m.params[::2])
                loss,g=m.loss_grad(x,mask,l2)
                self.assertAlmostEqual(loss,reference(),places=14)
                for p,grad in zip(m.params,g):
                    for idx in np.ndindex(p.shape):
                        old=p[idx];p[idx]=old+1e-6;a=reference();p[idx]=old-1e-6;b=reference();p[idx]=old
                        self.assertAlmostEqual(grad[idx],(a-b)/2e-6,places=8);probes+=1
        self.assertEqual(probes,200)

    def test_no_mutation_and_first_adam_update(self):
        m=Denoiser(2,3,2);x=np.array([[.3,.9]]);mask=np.array([[True,False]])
        snapshot=[p.copy() for p in m.params];xc=x.copy();mc=mask.copy()
        loss,g=m.loss_grad(x,mask);m.predict(x,mask)
        for a,b in zip(snapshot,m.params):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(x,xc);np.testing.assert_array_equal(mask,mc)
        Adam(m.params,.002).step(m.params,g)
        for before,after,grad in zip(snapshot,m.params,g):np.testing.assert_allclose(after,before-.002*grad/(np.abs(grad)+1e-8),rtol=1e-12,atol=1e-12)
        self.assertLess(m.loss_grad(x,mask)[0],loss)

    def test_missing_input_is_not_seen(self):
        a=np.array([[.1,.7,.9]]);b=np.array([[.1,.2,.2]]);mask=np.array([[0,1,1]])
        m=Denoiser(3,4,2)
        np.testing.assert_array_equal(m.predict(a,mask),m.predict(b,mask))
        self.assertEqual(m.predict(a,mask)[0,0],.1)

    def test_stable_sigmoid_extreme(self):
        m=Denoiser(2,2,1);m.params[-1][:]=[-1e4,1e4]
        np.testing.assert_array_equal(m.raw([[.3,.7]],[[1,1]]),[[0,1]])

    def test_model_roundtrip_is_detached(self):
        m=Denoiser(3,4,2);v=m.to_dict();r=Denoiser.from_dict(v)
        np.testing.assert_array_equal(m.predict([[.2,.5,.8]],[[1,0,1]]),r.predict([[.2,.5,.8]],[[1,0,1]]))
        v['params'][0][0][0]=999;self.assertNotEqual(r.params[0][0,0],999)

    def test_zero_missing_loss_refused_but_predict_identity(self):
        m=Denoiser(2,3,1);x=np.array([[.1,.8]])
        np.testing.assert_array_equal(m.predict(x,[[0,0]]),x)
        with self.assertRaises(ValueError):m.loss_grad(x,[[0,0]])

    def test_invalid_inputs(self):
        m=Denoiser(2,3,1)
        for x,mask in [([],[]),([[np.nan,.1]],[[1,0]]),([[1.1,.1]],[[1,0]]),([[.2,.1]],[[.5,0]]),([[.2,.1]],[[1]])]:
            with self.assertRaises(ValueError):m.predict(x,mask)
        for args in [(True,3,1),(0,3,1),(2,3.5,1),(2,3,513)]:
            with self.assertRaises(ValueError):Denoiser(*args)

    def test_bad_model_and_grad_rejected(self):
        m=Denoiser(2,2,1)
        for kind in ['schema','params','finite']:
            v=m.to_dict()
            if kind=='schema':v['schema']=2
            elif kind=='params':v['params'][0]=[[1]]
            else:v['params'][0][0][0]=float('nan')
            with self.assertRaises(ValueError):Denoiser.from_dict(v)
        ps=[p.copy() for p in m.params];g=[np.ones_like(p) for p in ps];g[-1][0]=float('nan')
        with self.assertRaises(ValueError):Adam(ps).step(ps,g)
        for a,b in zip(ps,m.params):np.testing.assert_array_equal(a,b)

class BaselineTests(unittest.TestCase):
    def test_pca_observed_lstsq_augmented_reference(self):
        rng=np.random.default_rng(9);fit=rng.random((12,5));x=rng.random((3,5));mask=np.array([[1,0,1,0,0],[0,0,1,1,0],[1,1,1,1,1]],dtype=bool)
        p=PCA.fit(fit,3,.15);got=p.predict(x,mask)
        for i in range(3):
            b=p.basis[:,~mask[i]].T
            a=np.linalg.lstsq(np.vstack([b,np.sqrt(.15)*np.eye(3)]),np.r_[x[i,~mask[i]]-p.mean[~mask[i]],np.zeros(3)],rcond=None)[0]
            expected=np.where(mask[i],np.clip(p.mean+a@p.basis,0,1),x[i]);np.testing.assert_allclose(got[i],expected,atol=1e-13)
        # Hidden targets cannot change latent least-squares solve.
        altered=x.copy();altered[mask]=1-altered[mask]
        np.testing.assert_array_equal(got,p.predict(altered,mask))

    def test_pca_projection_and_orthonormality(self):
        x=np.array([[0,0,1],[.2,.4,.5],[1,1,0],[.5,.1,.3]])
        p=PCA.fit(x,2,.1);np.testing.assert_allclose(p.basis@p.basis.T,np.eye(2),atol=1e-14)
        np.testing.assert_array_equal(p.predict(x,np.zeros_like(x)),x)
        restored=PCA.from_dict(p.to_dict());np.testing.assert_array_equal(restored.basis,p.basis)

    def test_pixel_mean_and_nearest_direct_reference(self):
        fit=np.array([[.1,.7,.2],[.9,.3,.8],[.1,.5,.6]])
        x=np.array([[.1,.8,.9],[.9,.3,.4]]);mask=np.array([[0,1,1],[1,0,1]],dtype=bool)
        pred,ids=nearest(x,mask,fit,True)
        self.assertEqual(ids.tolist(),[0,1])
        np.testing.assert_array_equal(pred,[[.1,.7,.2],[.9,.3,.8]])
        np.testing.assert_allclose(pixel_mean(x,mask,fit.mean(0)),np.where(mask,fit.mean(0),x))
        # No observations: deterministic lowest fitting index tie, not target-aware.
        np.testing.assert_array_equal(nearest(x,np.ones_like(x),fit),np.broadcast_to(fit[0],x.shape))

    def test_mask_cardinality_and_locality(self):
        for k in [1,16,32,48,64]:
            a=masks(20,k);np.testing.assert_array_equal(a.sum(1),np.full(20,k));np.testing.assert_array_equal(a,masks(20,k))
        b=masks(20,16,11,'block')
        for row in b:
            coords=np.argwhere(row.reshape(8,8));self.assertEqual(len(coords),16);np.testing.assert_array_equal(coords.max(0)-coords.min(0),[3,3])

    def test_metric_denominator_only_hidden(self):
        x=np.array([[.2,.3],[.4,.5]]);p=np.array([[1,.8],[.9,.5]]);m=np.array([[0,1],[1,0]],dtype=bool)
        np.testing.assert_allclose(hidden_mse(x,p,m),[.25,.25])
        with self.assertRaises(ValueError):hidden_mse(x,p,np.array([[0,0],[1,1]]))

    def test_invalid_control_config(self):
        for rank in [0,4,True,1.5]:
            with self.assertRaises(ValueError):PCA.fit([[.1,.3],[.2,.4]],rank,.1)
        for ridge in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):PCA.fit([[.1,.3],[.2,.4]],1,ridge)
        for args in [(0,32,1,'random'),(1,0,1,'random'),(1,32,1,'block'),(1,16,1,'bad')]:
            with self.assertRaises(ValueError):masks(*args)
