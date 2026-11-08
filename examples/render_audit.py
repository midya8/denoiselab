"""Deterministic vector preview from all stored audit values, with report metadata."""
import base64,gzip,html,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from denoiselab.io import read,dumps,digest
from denoiselab.data import ROOT

def render(report=None):
    r=report or read(ROOT/'artifacts/report.json')
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="820" viewBox="0 0 1120 820" role="img" aria-labelledby="title desc">',
        '<title id="title">DenoiseLab: real held-out reconstruction, including stronger controls</title>',
        '<desc id="desc">First official test row with the fixed 32-pixel mask. Four 1797-row audits compare hidden-pixel MSE; nearest neighbor and PCA beat the neural net at 16 missing pixels.</desc>',
        '<rect width="1120" height="820" fill="#fafbf9"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#24272b} .muted{fill:#5b626b}</style>']
    def text(x,y,s,size=17,extra=''):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" {extra}>{html.escape(str(s))}</text>')
    text(36,42,'DenoiseLab / manual gradients, learned weights, strong controls',16,'class="muted"')
    text(36,94,'Reconstruction is a guess.',38)
    text(36,130,'Same observed pixels. No model sees the erased targets. First test row, fixed 32-pixel mask.',17)
    e=r['cases'][1]['examples'][0]
    panels=[('Clean reference',e['clean']),('Masked input',e['observed']),('Neural',e['predictions']['network']),('PCA / ridge',e['predictions']['pca']),('Nearest neighbor',e['predictions']['nearest']),('Pixel mean',e['predictions']['pixel_mean'])]
    for pi,(label,values) in enumerate(panels):
        x=36+pi*178; text(x,178,label,16)
        for j,v in enumerate(values):
            color='#36647e' if pi==1 and e['mask'][j] else '#'+format(round(v*255),'02x')*3
            parts.append(f'<rect data-panel="{pi}" data-pixel="{j}" x="{x+j%8*19}" y="{196+j//8*19}" width="19" height="19" fill="{color}"/>')
    text(36,390,'Official test: mean hidden-pixel squared error / 1,797 digits per pattern',21)
    names=['Pattern','Neural','PCA','Neighbor','Pixel mean'];xs=[36,280,480,680,880]
    for x,n in zip(xs,names):text(x,436,n,18,'font-weight="600"')
    for i,c in enumerate(r['cases']):
        y=484+i*52
        for x,v in zip(xs,[c['name']]+[f"{c['methods'][m]['hidden_mse']:.7f}" for m in ['network','pca','nearest','pixel_mean']]):text(x,y,v,18)
        parts.append(f'<path d="M36 {y+15}H1084" stroke="#dce1e5"/>')
    text(36,700,'At 16 missing pixels, nearest neighbor and PCA beat this network. No universal winner.',18)
    text(36,733,'Scattered-32 training only. Fixed block masks are a disclosed distribution shift, not a causal test.',16,'class="muted"')
    text(36,766,'Alpaydin & Kaynak (1998), UCI Optical Digits, CC BY 4.0. Code: MIT. No endorsement.',15,'class="muted"')
    text(36,796,'No calibrated uncertainty, image-file OCR, generation, semantic-preservation or real-world guarantee.',15,'class="muted"')
    payload=base64.b64encode(gzip.compress(dumps(r).encode(),mtime=0)).decode()
    parts.append(f'<metadata id="report" data-sha256="{digest(r)}" data-format="base64-gzip-json">{payload}</metadata></svg>\n')
    return '\n'.join(parts)
if __name__=='__main__':
    p=ROOT.parent/'assets/denoise-audit.svg';s=render()
    if '--check' in sys.argv:assert p.read_text()==s,'figure differs'
    else:p.write_text(s)
