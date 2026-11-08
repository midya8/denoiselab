import base64,gzip,json,unittest,xml.etree.ElementTree as ET
from denoiselab.data import ROOT
from denoiselab.io import read
from examples.render_audit import render
class FigureTests(unittest.TestCase):
    def test_exact_svg_and_full_report_metadata(self):
        s=render();self.assertEqual((ROOT.parent/'assets/denoise-audit.svg').read_text(),s)
        t=ET.fromstring(s);ns={'s':'http://www.w3.org/2000/svg'}
        m=t.find('s:metadata',ns)
        self.assertEqual(json.loads(gzip.decompress(base64.b64decode(m.text))),read(ROOT/'artifacts/report.json'))
    def test_every_shown_pixel_and_metric(self):
        t=ET.fromstring(render());ns={'s':'http://www.w3.org/2000/svg'};r=read(ROOT/'artifacts/report.json');e=r['cases'][1]['examples'][0]
        panels=[e['clean'],e['observed']]+[e['predictions'][n] for n in ['network','pca','nearest','pixel_mean']]
        pixels=t.findall('.//s:rect[@data-pixel]',ns);self.assertEqual(len(pixels),384)
        for p in pixels:
            pi,j=int(p.attrib['data-panel']),int(p.attrib['data-pixel'])
            color='#36647e' if pi==1 and e['mask'][j] else '#'+format(round(panels[pi][j]*255),'02x')*3
            self.assertEqual(p.attrib['fill'],color)
        texts=[n.text for n in t.findall('s:text',ns)]
        for c in r['cases']:
            for n in ['network','pca','nearest','pixel_mean']:self.assertIn(f"{c['methods'][n]['hidden_mse']:.7f}",texts)
    def test_hidden_record_change_changes_metadata(self):
        r=read(ROOT/'artifacts/report.json');s=render(r);r['cases'][0]['methods']['network']['row_hidden_mse'][-1]+=.01
        self.assertNotEqual(s,render(r))
