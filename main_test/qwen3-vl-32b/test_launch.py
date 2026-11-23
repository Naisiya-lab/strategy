import importlib.util
import json
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('launcher',Path(__file__).with_name('launch.py'))
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path(__file__).with_name('config.json').read_text())

    def test_two_disjoint_tp2_replicas(self):
        launch.validate(self.cfg)
        for i,port in enumerate(('8200','8201')):
            cmd=launch.command(self.cfg,i)
            self.assertEqual(cmd[cmd.index('--tensor-parallel-size')+1],'2')
            self.assertEqual(cmd[cmd.index('--port')+1],port)
            self.assertEqual(cmd[cmd.index('--max-model-len')+1],'32768')
            self.assertIn('Qwen3-VL-32B-Instruct',cmd[cmd.index('--model')+1])

    def test_reject_overlapping_gpus(self):
        self.cfg['gpu_groups']=['0,1','1,2']
        with self.assertRaisesRegex(ValueError,'overlap'):launch.validate(self.cfg)

    def test_reject_wrong_tp_gpu_count(self):
        self.cfg['gpu_groups']=['0','2,3']
        with self.assertRaisesRegex(ValueError,'GPU count'):launch.validate(self.cfg)


if __name__=='__main__':unittest.main()
