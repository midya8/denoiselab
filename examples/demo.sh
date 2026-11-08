#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m denoiselab demo >/dev/null
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
python -m denoiselab train --output "$out" >/dev/null
python -m denoiselab evaluate --output "$out" >/dev/null
python -m denoiselab predict --model denoiselab/artifacts/network.json --input examples/predict.json >/dev/null
python examples/export_browser.py --check
python examples/render_audit.py --check

python -m denoiselab response --input examples/response.json > /tmp/denoiselab-response.json
