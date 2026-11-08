import unittest,numpy as np,json,subprocess,sys
from denoiselab.response import network_jacobian,pca_jacobian,response
from denoiselab.network import Denoiser
from denoiselab.baselines import PCA
from denoiselab.io import read
from denoiselab.data import ROOT,load

class ResponseTests(unittest.TestCase):
 def test_full_network_input_finite_differences(self):
  rng=np.random.default_rng(73);net=Denoiser(6,5,3);x=rng.uniform(.2,.8,6);mask=[0,1,0,1,0,1];j=np.array(network_jacobian(net,x.tolist(),mask)['jacobian']);eps=1e-5
  for c in range(6):
   a=x.copy();b=x.copy();a[c]+=eps;b[c]-=eps;fd=(net.predict([a],[mask])[0]-net.predict([b],[mask])[0])/(2*eps);np.testing.assert_allclose(j[:,c],fd,atol=1e-10,rtol=1e-7)
  np.testing.assert_array_equal(j[:,[1,3,5]],np.zeros((6,3)))
 def test_pca_independent_augmented_least_squares_and_derivatives(self):
  rng=np.random.default_rng(11);p=PCA([.5]*6,rng.normal(0,.1,(2,6)),.4);x=rng.uniform(.2,.8,6);m=np.array([0,1,0,1,0,1],bool);out=pca_jacobian(p,x.tolist(),m.tolist());b=p.basis[:,~m].T;aug=np.concatenate([b,np.sqrt(p.ridge)*np.eye(2)]);rhs=np.concatenate([x[~m]-p.mean[~m],np.zeros(2)]);coef=np.linalg.lstsq(aug,rhs,rcond=None)[0];np.testing.assert_allclose(out['coefficients'],coef,atol=1e-14)
  for c in range(6):
   a=x.copy();b=x.copy();a[c]+=1e-5;b[c]-=1e-5;fd=(p.predict([a],[m])[0]-p.predict([b],[m])[0])/2e-5;np.testing.assert_allclose(np.array(out['jacobian'])[:,c],fd,atol=1e-10)
 def test_clipping_boundaries_saturation_and_undefined_tangent(self):
  p=PCA([0,0],[[1,1]],.1);out=pca_jacobian(p,[0,0],[0,1]);self.assertEqual(out['undefined_output_rows'],[1]);self.assertEqual(out['jacobian'][1],[None,None])
  constant=pca_jacobian(PCA([0,0],[[0,0]],.1),[.5,0],[0,1]);self.assertEqual(constant['undefined_output_rows'],[]);self.assertEqual(constant['jacobian'][1],[0,0])
  p=PCA([0,-2],[[1,1]],.1);out=pca_jacobian(p,[.5,0],[0,1]);self.assertEqual(out['clipped_output_rows'],[1]);self.assertEqual(out['jacobian'][1],[0,0])
  # Means outside fitting-pixel range are accepted by the original PCA class; this is only a scalar clipping control.
  p=PCA([0,0],[[1,1]],.1);w=response(Denoiser(2,3,2),p,[[0,0],[1,1]],[0,.9],[0,1],[.2,.2]);self.assertIsNone(w['methods']['pca']['hidden_residual_l2']);self.assertIsNone(w['methods']['pca']['tangent_prediction'][1])
 def test_hidden_values_erased_visible_identity_and_no_mutation(self):
  net=Denoiser(4,3,2);p=PCA([.5]*4,[[.2]*4],.1);fit=[[.1]*4,[.9]*4];x=[.2,.3,.6,.7];edit=[.4,.5,.8,.9];m=[0,1,0,1];old=json.dumps([x,edit,m,fit]);a=response(net,p,fit,x,m,edit);b=response(net,p,fit,[.2,.99,.6,.01],m,[.4,.01,.8,.99]);self.assertEqual(a,b);self.assertEqual(json.dumps([x,edit,m,fit]),old);self.assertFalse(a['hidden_targets_used'])
  for method,w in a['methods'].items():
   self.assertEqual(w['edited_prediction'][0],.4);self.assertEqual(w['edited_prediction'][2],.8)
 def test_nearest_discontinuity_and_all_hidden_all_visible(self):
  net=Denoiser(2,3,2);p=PCA([.5,.5],[[.1,.1]],.1);fit=[[0,0],[1,1]];w=response(net,p,fit,[.5,0],[0,1],[.500001,0]);self.assertEqual(w['nearest_before_index'],0);self.assertTrue(w['nearest_switched']);self.assertEqual(w['methods']['nearest']['finite_minus_frozen'][1],1);self.assertEqual(w['nearest_runner_up_gap'],0)
  for m in [[0,0],[1,1]]:
   w=response(net,p,fit,[.3,.4],m,[.8,.9]);json.dumps(w,allow_nan=False)
   if all(m):self.assertEqual(w['delta'],[0,0]);self.assertEqual(w['methods']['network']['hidden_residual_l2'],0)
   else:
    np.testing.assert_array_equal(w['methods']['network']['local']['jacobian'],np.eye(2));self.assertEqual(w['methods']['network']['hidden_residual_l2'],0)
 def test_saved_64_by_64_all_observed_fd_columns(self):
  net=Denoiser.from_dict(read(ROOT/'artifacts/network.json'));p=PCA.from_dict(read(ROOT/'artifacts/pca.json'));report=read(ROOT/'artifacts/report.json');e=report['cases'][1]['examples'][0];x=np.array(e['clean']);m=e['mask'];x=np.where(m,0,np.clip(x,.1,.9));a=network_jacobian(net,x.tolist(),m);b=pca_jacobian(p,x.tolist(),m);eps=1e-5
  for c in np.flatnonzero(np.array(m)==0):
   plus=x.copy();minus=x.copy();plus[c]+=eps;minus[c]-=eps
   for model,j in [(net,a),(p,b)]:
    fd=(model.predict([plus],[m])[0]-model.predict([minus],[m])[0])/(2*eps);valid=[i for i in range(64) if i not in j['undefined_output_rows']];np.testing.assert_allclose(np.array(j['jacobian'],dtype=float)[valid,c],fd[valid],atol=2e-8,rtol=2e-6)
 def test_invalid_inputs(self):
  net=Denoiser(2,3,2);p=PCA([.5]*2,[[.2]*2],.1)
  for x,m,e in [([0,2],[0,1],[0,1]),([0,1],[0,2],[0,1]),([0],[0,1],[0,1]),([0,1],[0,1],[float('nan'),0]),([0,1],[0,1],[0])]:
   with self.assertRaises(ValueError):response(net,p,[[0,0]],x,m,e)

 def test_cli(self):
  from pathlib import Path
  root=Path(__file__).resolve().parents[1]
  out=subprocess.run([sys.executable,'-m','denoiselab','response','--input',str(root/'examples/response.json')],capture_output=True,text=True)
  self.assertEqual(out.returncode,0,out.stderr);r=json.loads(out.stdout);self.assertEqual(len(r['methods']['network']['local']['jacobian']),64);self.assertFalse(r['hidden_targets_used'])
