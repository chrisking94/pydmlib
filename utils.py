import matplotlib.pyplot as ppt
from configparser import ConfigParser
import os


class RSPlotManager(object):
    def __init__(self):
        self.__dict__.update(ppt.__dict__)
        self.__dict__.update(RSPlotManager.__dict__)

    def show(*args, **kwargs):
        from control import RSControl
        RSControl.thread.pause()
        ppt.show(*args, **kwargs)
        RSControl.thread.resume()


plt = RSPlotManager()


def printf(s, *args, **kwargs):
    from control import RSControl
    RSControl.thread.pause()
    if isinstance(s, str):
        try:
            print(s % args, **kwargs)
        except Exception as e:
            print(s, *args, **kwargs)
    else:
        print(str(s), **kwargs)
    RSControl.thread.resume()


class PydmConfig(ConfigParser):
    def __init__(self, file_path, *args, **kwargs):
        ConfigParser.__init__(self, *args, **kwargs)
        self.file_path = file_path
        if os.path.exists(file_path):
            self.read(file_path)

    def write(self, space_around_delimiters=True, _do_not_use=None):
        with open(self.file_path, 'w') as fp:
            ConfigParser.write(self, fp, space_around_delimiters)


def get_option_configurator(file_path, *args, **kwargs):
    """
    a pydmlib configurator factory
    :param file_path:
    :param args:
    :param kwargs:
    :return: Configurator object
    """
    from typing import GenericMeta
    from pandas.core.config import _global_config

    class MetaGlobalOption(GenericMeta):
        def __new__(cls, name, bases, attrs):
            new_attrs = MetaGlobalOption._wrap_pandas_set_option()
            new_attrs.update(attrs)
            return GenericMeta.__new__(cls, name, bases, attrs)

        @staticmethod
        def _wrap_pandas_set_option():
            ret_dict = {}
            for k in _global_config.keys():
                def fset(self, value):
                    _global_config[k] = value

                def fget(self):
                    return _global_config[k]
                ret_dict[k] = property(fget, fset)
            return ret_dict

    class GlobalOption(PydmConfig, metaclass=MetaGlobalOption):
        def __init__(self):
            PydmConfig.__init__(self, file_path, *args, **kwargs)

    return GlobalOption()


