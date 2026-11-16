"""Check the installed tests and full CLI outputs outside the checkout.

Use --section tests or --section commands to run either half on its own.
The default runs both. These are same-runtime packaging comparisons, not
independent numerical or statistical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['denoiselab', 'denoiselab.__main__', 'denoiselab.baselines',
           'denoiselab.data', 'denoiselab.edit_path', 'denoiselab.edit_plane',
           'denoiselab.experiment', 'denoiselab.io', 'denoiselab.network',
           'denoiselab.response']


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--section', choices=['all', 'tests', 'commands'], default='all')
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels), str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'assets', 'docs']:
        shutil.copytree(ROOT / name, outside / name)
    # Existing figure/export tests find static artifacts beside the package.
    # Copy static files only, never any library Python source.
    package = Path(call([py, '-I', '-c', 'import denoiselab;print(denoiselab.__path__[0])'], work).strip())
    for name in ['docs', 'assets']:
        shutil.copytree(ROOT / name, package.parent / name)
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
if sys.argv[3]=='tests':
    tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.testsRun>0 and result.wasSuccessful()
    print(json.dumps(dict(tests=result.testsRun)))
check()
print(json.dumps(dict(installed_origins=len(names),checked_before_and_after=True)))
'''
    if args.section in ['all', 'tests']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'tests'], outside))
    if args.section in ['all', 'commands']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        reference = work / 'reference'
        reference.mkdir()
        source_env = ENV.copy()
        source_env['PYTHONPATH'] = str(ROOT)

        def run_all(entry, folder, env):
            shutil.copytree(ROOT / 'examples', folder / 'inputs')
            shutil.copyfile(ROOT / 'denoiselab/artifacts/network.json', folder / 'network.json')
            reports = {}

            def command(name, *options):
                reports[name] = json.loads(call(entry + [name] + list(options), folder, env))
                return reports[name]

            command('demo')
            command('train', '--output', 'refit')
            command('evaluate', '--output', 'refit')
            command('predict', '--model', 'network.json', '--input', 'inputs/predict.json')
            command('response', '--input', 'inputs/response.json')
            for shape, field in [('path', 'value'), ('plane', 'values')]:
                result = command('edit-' + shape, '--input', 'inputs/edit-' + shape + '.json')
                value = json.loads((folder / ('inputs/edit-' + shape + '.json')).read_text())
                del value[field]
                value['witness'] = result[shape]
                file = 'verify-' + shape + '.json'
                (folder / file).write_text(json.dumps(value))
                command('verify-' + shape, '--input', file)
            files = {p.name: json.loads(p.read_text()) for p in (folder / 'refit').glob('*.json')}
            assert len(files) == 7, sorted(files)
            return reports, files

        expected = run_all([sys.executable, '-m', 'denoiselab'], reference, source_env)
        for number, entry in enumerate([[py, '-I', '-m', 'denoiselab'], [str(work / 'env/bin/denoiselab')]]):
            folder = outside / ('commands-' + str(number))
            folder.mkdir()
            actual = run_all(entry, folder, ENV)
            assert actual == expected, entry
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        print(json.dumps(dict(command_kinds=9, complete_reports=9, complete_refit_files=7,
                              fresh_models=3, entrypoints=2)))
