import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import evaluate as ev
from data import prepare


class LocalTest(unittest.TestCase):
    def test_dual_encoder_paths_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'model';high=root/'high';low=root/'low'
            for p in (source,high,low):p.mkdir()
            original=json.dumps({'architectures':['LlavaLlamaForCausalLM'],
                                 'mm_highres_vision_tower':'old_high','mm_lowres_vision_tower':'old_low'})
            (source/'config.json').write_text(original);(source/'weights.bin').write_bytes(b'fake')
            for p in (high,low):(p/'config.json').write_text('{}')
            (low/'preprocessor_config.json').write_text('{}')
            cfg={'model_path':str(source),'highres_vision_path':str(high),'lowres_vision_path':str(low)}
            with patch.object(ev,'ROOT',root),patch.object(Path,'symlink_to'):
                view=Path(ev.model_view(cfg))
            d=json.loads((view/'config.json').read_text())
            self.assertEqual(d['mm_highres_vision_tower'],str(high.resolve()))
            self.assertEqual(d['mm_lowres_vision_tower'],str(low.resolve()))
            self.assertEqual((source/'config.json').read_text(),original)
            (source/'adapter_config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'LoRA adapter'):ev.model_view(cfg)

    def test_real_fused_length_not_fixed_clip_estimate(self):
        prepared=(None,SimpleNamespace(ndim=2,shape=(1,10)),None,None,
                  SimpleNamespace(ndim=3,shape=(1,2304,4096)),None)
        self.assertEqual(ev.expanded_prompt_length(prepared),2304)
        with self.assertRaisesRegex(ValueError,'Cannot identify'):ev.expanded_prompt_length((None,))

    def test_dotted_id_preserved_and_no_reference_in_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir()
            (root/'images'/'TabFact_x.html.csv.jpg').write_bytes(b'image')
            (root/'ann.json').write_text(json.dumps([{'item_id':'x','image_id':'TabFact_x.html.csv',
                'input':'question','task_type':'TFV','dataset_name':'TabFact','output':'SECRET'}]))
            rows,_=prepare({'data_root':str(root),'annotation':'ann.json','seed':42},'smoke',4)
            self.assertEqual(len(rows),1);self.assertNotIn('SECRET',json.dumps(rows))


if __name__=='__main__':unittest.main()
