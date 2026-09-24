import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('backend',Path(__file__).parents[1]/'texture/backend.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

class BackendTests(unittest.TestCase):
    def test_existing_backend_skips_setup(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);config={k:d for k in ('python','source','models')}
            (p/'texture_backend.json').write_text(json.dumps(config))
            with patch.object(b.subprocess,'Popen',side_effect=AssertionError('must not install')):
                self.assertEqual(b.ensure_backend(p),config)
    def test_stale_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'texture_backend.json').write_text('{bad json')
            self.assertIsNone(b.read_ready(p))
    def test_first_run_and_reuse(self):
        with tempfile.TemporaryDirectory() as d,patch.object(b.sys,'platform','linux'):
            p=Path(d);(p/'scripts').mkdir()
            (p/'scripts/setup_texture.py').write_text('from pathlib import Path\nimport json\np=Path(__file__).resolve().parents[1]\n(p/"texture_backend.json").write_text(json.dumps({k:str(p) for k in ("python","source","models")}))\nprint("fixture setup completed")\n')
            config=b.ensure_backend(p)
            self.assertTrue((p/'texture_setup.log').exists())
            with patch.object(b.subprocess,'Popen',side_effect=AssertionError('must reuse')):self.assertEqual(b.ensure_backend(p),config)
    def test_failed_setup_has_log(self):
        with tempfile.TemporaryDirectory() as d,patch.object(b.sys,'platform','linux'):
            p=Path(d);(p/'scripts').mkdir();(p/'scripts/setup_texture.py').write_text('raise RuntimeError("fixture compiler failure")')
            with self.assertRaisesRegex(RuntimeError,'fixture compiler failure'):b.ensure_backend(p)
    def test_cancel_before_launch(self):
        with tempfile.TemporaryDirectory() as d,patch.object(b.sys,'platform','linux'):
            def cancel():raise InterruptedError('cancelled')
            with self.assertRaises(InterruptedError):b.ensure_backend(Path(d),cancel)

if __name__=='__main__':unittest.main()
