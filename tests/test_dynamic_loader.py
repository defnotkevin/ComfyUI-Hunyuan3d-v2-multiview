import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
spec=importlib.util.spec_from_file_location('dynamic_loader',Path(__file__).parents[1]/'texture/dynamic_loader.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DynamicLoaderTests(unittest.TestCase):
    def test_new_module_after_parent_was_cached(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture_dynamic';p.mkdir();(p/'__init__.py').write_text('')
            sys.path.insert(0,d)
            try:
                importlib.import_module('fixture_dynamic')
                with self.assertRaises(ModuleNotFoundError):importlib.import_module('fixture_dynamic.modules')
                stamp=p.stat()
                (p/'modules.py').write_text('class Model: pass')
                # Reproduce a directory timestamp/cache race deterministically.
                os.utime(p,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
                original=lambda name,path:getattr(importlib.import_module(path),name)
                loader=types.SimpleNamespace(get_class_in_module=original)
                with m.fresh_dynamic_imports(loader):
                    self.assertEqual(loader.get_class_in_module('Model','fixture_dynamic.modules').__name__,'Model')
                self.assertIs(loader.get_class_in_module,original)
            finally:
                sys.path.remove(d)
                for key in list(sys.modules):
                    if key.startswith('fixture_dynamic'):sys.modules.pop(key)
    def test_real_missing_dependency_not_hidden(self):
        def fail(*args):raise ModuleNotFoundError('missing_dependency')
        loader=types.SimpleNamespace(get_class_in_module=fail)
        with self.assertRaisesRegex(ModuleNotFoundError,'missing_dependency'):
            with m.fresh_dynamic_imports(loader):loader.get_class_in_module('X','Y')
        self.assertIs(loader.get_class_in_module,fail)
