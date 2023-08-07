from mmcv import Config
from mmcv.utils.misc import import_modules_from_strings
from pathlib import Path


class ConfigWrapper(Config):

    def get_or_default(self, key, default):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default

    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        if isinstance(filename, Path):
            filename = str(filename)
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        return ConfigWrapper(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except:
            raise AttributeError(
                f"Config '{self.filename}' has no attribute '{name}'")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            raise AttributeError(
                f"Config '{self.filename}' has no attribute '{name}'")
        
    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except:
            raise KeyboardInterrupt(
                f"Config '{self.filename}' has no attribute '{name}'")
