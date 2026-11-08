#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$1"
source_root=$2
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cd "$out"
python - <<'PY'
import denoiselab,denoiselab.__main__,denoiselab.io,denoiselab.network,denoiselab.baselines,denoiselab.data,denoiselab.experiment
print('seven installed modules imported')
PY
python -m denoiselab demo >demo.json
cmp demo.json "$source_root/denoiselab/artifacts/report.json"
python -m denoiselab predict --model "$PYTHONPATH/denoiselab/artifacts/network.json" --input "$source_root/examples/predict.json" >predict.json
python -m denoiselab train --output refit >selection.json
python -m denoiselab evaluate --output refit >evaluation.json
python - "$source_root" <<'PY'
import json,sys,numpy as np
from pathlib import Path
source=Path(sys.argv[1])/'denoiselab/artifacts'
a=json.load(open('evaluation.json'));b=json.load(open(source/'report.json'))
assert len(a['cases'])==4 and a['rows']==1797
for c,d in zip(a['cases'],b['cases']):
 assert c['mask_sha256']==d['mask_sha256']
 for k in c['methods']:np.testing.assert_allclose(c['methods'][k]['row_hidden_mse'],d['methods'][k]['row_hidden_mse'],rtol=1e-6,atol=1e-8)
s=json.load(open('selection.json'));t=json.load(open(source/'selection.json'))
assert (s['selected_epoch'],s['selected_pca_rank'],s['selected_pca_ridge'])==(t['selected_epoch'],t['selected_pca_rank'],t['selected_pca_ridge'])
print('installed wheel: all four CLIs, full refit and held-out evaluation pass')
PY
