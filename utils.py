import matplotlib.pyplot as ppt
from configparser import ConfigParser
from pandas.core.config import _global_config
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


class MetaGlobalOption(type):
    def __new__(mcs, name, bases, attrs):
        new_attrs = MetaGlobalOption._wrap_pandas_set_option(attrs)
        new_attrs.update(attrs)
        return type(name, bases, attrs)

    @staticmethod
    def _wrap_pandas_set_option(option_dict):
        ret_dict = {}
        for k in option_dict.keys():
            def fset(self, value):
                _global_config[k] = value

            def fget(self):
                return _global_config[k]
            ret_dict[k] = property(fget, fset)
        return ret_dict


class GlobalOption(PydmConfig, metaclass=MetaGlobalOption):
    def __init__(self, *args, **kwargs):
        PydmConfig.__init__(*args, **kwargs)


