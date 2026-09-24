import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('setup_texture',Path(__file__).parents[1]/'scripts/setup_texture.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SetupTests(unittest.TestCase):
    def test_disables_inherited_optional_downloader(self):
        with patch.dict(os.environ,{'HF_HUB_ENABLE_HF_TRANSFER':'1'}),patch.object(m.subprocess,'run') as run:
            m.run(['python','-c','pass'])
            self.assertEqual(run.call_args.kwargs['env']['HF_HUB_ENABLE_HF_TRANSFER'],'0')
            self.assertEqual(os.environ['HF_HUB_ENABLE_HF_TRANSFER'],'1')
            self.assertTrue(run.call_args.kwargs['check'])
