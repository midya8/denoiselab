"""Export selected public models and first ten test rows; no fitting or evaluation."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from denoiselab.data import ROOT,load
from denoiselab.io import read,dumps,digest

def content():
    report=read(ROOT/'artifacts/report.json');protocol=read(ROOT/'artifacts/protocol.json')
    train,_=load('train');fit=train[protocol['fit_indices']]
    cases=[dict(name=c['name'],kind=c['kind'],missing=c['missing'],seed=c['seed'],
                mask_sha256=c['mask_sha256'],methods={n:{k:v for k,v in d.items() if k!='row_hidden_mse'} for n,d in c['methods'].items()},examples=c['examples']) for c in report['cases']]
    value=dict(schema=1,network=read(ROOT/'artifacts/network.json'),pca=read(ROOT/'artifacts/pca.json'),
            fit=fit.tolist(),fit_indices=protocol['fit_indices'],cases=cases,
            source_hashes={n:digest(read(ROOT/'artifacts'/n)) for n in ['network.json','pca.json','protocol.json','selection.json','evaluation-plan.json','report.json']})
    return '"use strict";\nconst DENOISE_DATA = '+dumps(value)+';\n'
if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'docs/data.js';expected=content()
    if '--check' in sys.argv:
        assert path.read_text()==expected,'browser data differs; run exporter'
    else:path.write_text(expected)
