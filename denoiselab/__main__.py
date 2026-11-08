import argparse
from pathlib import Path
import sys
from .io import read,dumps
from .data import ROOT

def main():
    p=argparse.ArgumentParser(description='Manual masked-digit reconstruction. All commands run offline.')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('demo')
    for name in ('train','evaluate'):
        s=sub.add_parser(name);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('predict');s.add_argument('--model',type=Path,required=True);s.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.command=='demo':out=read(ROOT/'artifacts/report.json')
        elif args.command=='train':
            from .experiment import train
            out=train(args.output)
        elif args.command=='evaluate':
            from .experiment import evaluate
            out=evaluate(args.output)
        else:
            from .network import Denoiser
            v=read(args.input)
            if not isinstance(v,dict) or set(v)!={'pixels','mask'}:raise ValueError('input requires pixels and mask')
            out={'prediction':Denoiser.from_dict(read(args.model)).predict(v['pixels'],v['mask']).tolist()}
        sys.stdout.write(dumps(out));return 0
    except (ValueError,TypeError,KeyError,OSError) as e:
        print('denoiselab: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
