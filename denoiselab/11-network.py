"""Tanh bottleneck denoiser with explicit missing-mask inputs and manual derivatives."""
import numpy as np


def matrix(x, width=None):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1] or (width is not None and a.shape[1] != width) or not np.isfinite(a).all():
        raise ValueError('expected nonempty finite matrix with matching feature width')
    return a


def inputs(x, mask, width=None):
    x = matrix(x, width)
    m = np.asarray(mask)
    if m.shape != x.shape or not np.isin(m, [0, 1]).all():
        raise ValueError('mask must match input and contain only 0/1; 1 is missing')
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError('pixels must be in [0,1]')
    m = m.astype(bool)
    # Hidden values are erased before inference, including if caller supplies them.
    return x, m, np.concatenate([np.where(m, 0., x), m.astype(float)], axis=1)


class Denoiser:
    def __init__(self, features=64, hidden=64, code=16, seed=23):
        if any(type(v) is not int or not 1 <= v <= 512 for v in (features, hidden, code)):
            raise ValueError('dimensions must be integers in 1..512')
        self.features, self.hidden, self.code = features, hidden, code
        rng = np.random.default_rng(seed)
        dims = [2 * features, hidden, code, hidden, features]
        self.params = []
        for a, b in zip(dims[:-1], dims[1:]):
            self.params += [rng.normal(0, np.sqrt(1 / a), (a, b)), np.zeros(b)]

    def _forward(self, encoded):
        acts = [encoded]
        for i in range(0, 8, 2):
            z = acts[-1] @ self.params[i] + self.params[i+1]
            if i == 6:
                # Stable sigmoid, without overflow at extreme finite logits.
                y = np.empty_like(z)
                positive = z >= 0
                y[positive] = 1 / (1 + np.exp(-z[positive]))
                ez = np.exp(z[~positive])
                y[~positive] = ez / (1 + ez)
            else:
                y = np.tanh(z)
            acts.append(y)
        return acts

    def raw(self, x, mask):
        _, _, enc = inputs(x, mask, self.features)
        return self._forward(enc)[-1]

    def predict(self, x, mask):
        x, m, enc = inputs(x, mask, self.features)
        return np.where(m, self._forward(enc)[-1], x)

    def loss_grad(self, x, mask, l2=1e-5):
        x, m, enc = inputs(x, mask, self.features)
        if type(l2) not in (int, float) or not np.isfinite(l2) or l2 < 0 or not m.any():
            raise ValueError('loss needs positive missing count and finite nonnegative l2')
        acts = self._forward(enc)
        delta = (acts[-1] - x) * m
        loss = float(np.sum(delta**2) / m.sum() + l2 / 2 * sum(np.sum(w*w) for w in self.params[::2]))
        dz = 2 * delta / m.sum() * acts[-1] * (1 - acts[-1])
        gradients = [None] * 8
        for i in range(6, -1, -2):
            gradients[i] = acts[i//2].T @ dz + l2 * self.params[i]
            gradients[i+1] = dz.sum(axis=0)
            if i:
                dz = (dz @ self.params[i].T) * (1 - acts[i//2]**2)
        return loss, gradients

    def to_dict(self):
        return dict(schema=1, features=self.features, hidden=self.hidden, code=self.code,
                    params=[p.tolist() for p in self.params])

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'schema','features','hidden','code','params'} or value['schema'] != 1:
            raise ValueError('invalid network schema')
        out = cls(value['features'], value['hidden'], value['code'])
        try:
            ps = [np.asarray(p, dtype=float) for p in value['params']]
            if len(ps) != 8 or any(a.shape != b.shape or not np.isfinite(a).all() for a,b in zip(ps,out.params)):
                raise ValueError('invalid parameter shape or value')
        except (TypeError, OverflowError) as e:
            raise ValueError('invalid parameters') from e
        out.params = [p.copy() for p in ps]
        return out


class Adam:
    def __init__(self, params, rate=.003):
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('positive finite learning rate required')
        self.rate, self.t = rate, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        if len(params) != len(grads) or len(params) != len(self.m) or any(g.shape != p.shape or not np.isfinite(g).all() for p,g in zip(params,grads)):
            raise ValueError('gradient shape/finite mismatch')
        self.t += 1
        for i,(p,g) in enumerate(zip(params,grads)):
            self.m[i] = .9*self.m[i] + .1*g
            self.v[i] = .999*self.v[i] + .001*g*g
            p -= self.rate*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)
