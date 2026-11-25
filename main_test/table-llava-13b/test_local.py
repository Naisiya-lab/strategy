import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import evaluate as ev
from data import prepare


class LocalTest(unittest.TestCase):
    def test_original_model_view_preserves_source_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';clip=root/'clip';model.mkdir();clip.mkdir()
            config={'architectures':['LlavaLlamaForCausalLM'],'mm_vision_tower':'openai/clip-vit-large-patch14-336'}
            original=json.dumps(config);(model/'config.json').write_text(original)
            (model/'weights.bin').write_bytes(b'weights')
            (clip/'config.json').write_text('{}');(clip/'preprocessor_config.json').write_text('{}')
            with patch.object(ev,'ROOT',root),patch.object(Path,'symlink_to') as link:
                view=Path(ev.model_view({'model_path':str(model),'clip_path':str(clip)}))
                link.assert_called_once()
            self.assertEqual((model/'config.json').read_text(),original)
            self.assertEqual(json.loads((view/'config.json').read_text())['mm_vision_tower'],str(clip.resolve()))

    def test_dotted_image_id_and_gold_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir()
            (root/'images'/'TabFact_x.html.csv.jpg').write_bytes(b'image')
            row={'item_id':'test','image_id':'TabFact_x.html.csv','input':'question',
                 'task_type':'TFV','dataset_name':'TabFact','output':'SECRET'}
            (root/'ann.json').write_text(json.dumps([row]))
            cfg={'data_root':str(root),'annotation':'ann.json','seed':42}
            prepared,_=prepare(cfg,'smoke',4)
            self.assertEqual(len(prepared),1)
            self.assertNotIn('SECRET',json.dumps(prepared))


if __name__=='__main__':unittest.main()
