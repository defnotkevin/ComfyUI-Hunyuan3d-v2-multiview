"""Worker-local workaround for freshly copied Diffusers custom modules."""
from contextlib import contextmanager
import importlib


@contextmanager
def fresh_dynamic_imports(loader):
    # Diffusers 0.32.2 copies pipeline/component files into its Python cache,
    # then immediately imports them. Refresh FileFinder caches before imports,
    # including when pipeline loading has already imported the parent package.
    original = loader.get_class_in_module
    def load(class_name, module_path):
        importlib.invalidate_caches()
        return original(class_name, module_path)
    loader.get_class_in_module = load
    try:
        yield
    finally:
        loader.get_class_in_module = original


def create_painter(pipeline_class, config):
    from diffusers.utils import dynamic_modules_utils
    with fresh_dynamic_imports(dynamic_modules_utils):
        return pipeline_class(config)
