#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$1"
source_root=$2
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cd "$out"
python - <<'PY'
import denoiselab.edit_plane,denoiselab.edit_path,denoiselab.response,denoiselab,denoiselab.__main__,denoiselab.io,denoiselab.network,denoiselab.baselines,denoiselab.data,denoiselab.experiment
print('ten installed modules imported')
PY
python -m denoiselab demo >demo.json
cmp demo.json "$source_root/denoiselab/artifacts/report.json"
python -m denoiselab predict --model "$PYTHONPATH/denoiselab/artifacts/network.json" --input "$source_root/examples/predict.json" >predict.json
python -m denoiselab response --input "$source_root/examples/response.json" >response.json
python -m denoiselab edit-path --input "$source_root/examples/edit-path.json" >path.json
python - "$source_root" <<'PY2'
import json,sys
from pathlib import Path
v=json.loads((Path(sys.argv[1])/'examples/edit-path.json').read_text());v['witness']=json.load(open('path.json'))['path'];del v['value'];Path('verify-path.json').write_text(json.dumps(v))
PY2
python -m denoiselab verify-path --input verify-path.json >path-verification.json
python -m denoiselab edit-plane --input "$source_root/examples/edit-plane.json" >plane.json
python - "$source_root" <<'PY3'
import json,sys
from pathlib import Path
v=json.loads((Path(sys.argv[1])/'examples/edit-plane.json').read_text());v['witness']=json.load(open('plane.json'))['plane'];del v['values'];Path('verify-plane.json').write_text(json.dumps(v))
PY3
python -m denoiselab verify-plane --input verify-plane.json >plane-verification.json
python -m denoiselab train --output refit >selection.json
python -m denoiselab evaluate --output refit >evaluation.json
python - "$source_root" <<'PY'
import json,sys,numpy as np
from pathlib import Path
source=Path(sys.argv[1])/'denoiselab/artifacts'
a=json.load(open('evaluation.json'));b=json.load(open(source/'report.json'))
assert len(a['cases'])==4 and a['rows']==1797
plane=json.load(open('plane-verification.json'));assert plane['source_matched'] and plane['verified'] and plane['area']==['1','1']
path=json.load(open('path-verification.json'));assert path['source_matched'] and path['verified']
r=json.load(open('response.json'));assert r['hidden_targets_used'] is False and len(r['methods']['network']['local']['jacobian'])==64
for c,d in zip(a['cases'],b['cases']):
 assert c['mask_sha256']==d['mask_sha256']
 for k in c['methods']:np.testing.assert_allclose(c['methods'][k]['row_hidden_mse'],d['methods'][k]['row_hidden_mse'],rtol=1e-6,atol=1e-8)
s=json.load(open('selection.json'));t=json.load(open(source/'selection.json'))
assert (s['selected_epoch'],s['selected_pca_rank'],s['selected_pca_ridge'])==(t['selected_epoch'],t['selected_pca_rank'],t['selected_pca_ridge'])
print('installed wheel: all nine CLIs, full refit and held-out evaluation pass')
PY
