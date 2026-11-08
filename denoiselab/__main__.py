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
    for name in ('response','edit-path','verify-path','edit-plane','verify-plane'):
        s=sub.add_parser(name);s.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    try:
        if args.command=='demo':out=read(ROOT/'artifacts/report.json')
        elif args.command=='train':
            from .experiment import train
            out=train(args.output)
        elif args.command=='evaluate':
            from .experiment import evaluate
            out=evaluate(args.output)
        elif args.command in ('edit-path','verify-path'):
            from .edit_path import edit_path,query_path,verify_path
            from .data import load
            v=read(args.input);extra='value' if args.command=='edit-path' else 'witness'
            if not isinstance(v,dict) or set(v)!={'pixels','mask','input_pixel',extra}:raise ValueError('path input requires pixels, mask, input_pixel and '+extra)
            fit=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']]
            w=edit_path(fit,v['pixels'],v['mask'],v['input_pixel']) if args.command=='edit-path' else v['witness']
            check=verify_path(w,fit,v['pixels'],v['mask'],v['input_pixel'])
            out={'path':w,'verification':check,'query':query_path(w,v['value'])} if args.command=='edit-path' else check
        elif args.command in ('edit-plane','verify-plane'):
            from .edit_plane import edit_plane,query_plane,verify_plane
            from .data import load
            v=read(args.input);extra='values' if args.command=='edit-plane' else 'witness'
            if not isinstance(v,dict) or set(v)!={'pixels','mask','input_pixels',extra}:raise ValueError('plane input requires pixels, mask, input_pixels and '+extra)
            fit=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']]
            w=edit_plane(fit,v['pixels'],v['mask'],v['input_pixels']) if args.command=='edit-plane' else v['witness']
            check=verify_plane(w,fit,v['pixels'],v['mask'],v['input_pixels'])
            if args.command=='edit-plane':
                if not isinstance(v['values'],list) or len(v['values'])!=2:raise ValueError('values requires two rational strings')
                out={'plane':w,'verification':check,'query':query_plane(w,*v['values'])}
            else:out=check
        elif args.command=='response':
            from .network import Denoiser
            from .baselines import PCA
            from .response import response
            from .data import load
            v=read(args.input)
            if not isinstance(v,dict) or set(v)!={'pixels','mask','edited'}:raise ValueError('response input requires pixels, mask and edited')
            fit=load('train')[0][read(ROOT/'artifacts/protocol.json')['fit_indices']]
            out=response(Denoiser.from_dict(read(ROOT/'artifacts/network.json')),PCA.from_dict(read(ROOT/'artifacts/pca.json')),fit,v['pixels'],v['mask'],v['edited'])
        else:
            from .network import Denoiser
            v=read(args.input)
            if not isinstance(v,dict) or set(v)!={'pixels','mask'}:raise ValueError('input requires pixels and mask')
            out={'prediction':Denoiser.from_dict(read(args.model)).predict(v['pixels'],v['mask']).tolist()}
        sys.stdout.write(dumps(out));return 0
    except (ValueError,TypeError,KeyError,OSError) as e:
        print('denoiselab: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
