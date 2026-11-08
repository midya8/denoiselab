"""Finite JSON and content hashes; never pickle."""
import hashlib
import json
from pathlib import Path

def dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'

def digest(obj):
    return hashlib.sha256(dumps(obj).encode()).hexdigest()

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj))

def read(path):
    def reject(x):
        raise ValueError('nonfinite JSON: ' + x)
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError('duplicate JSON key: ' + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)
