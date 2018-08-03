import matplotlib.pyplot as ppt
from configparser import ConfigParser
import os
import pandas as pd
import numpy as np


class RSPlotManager(object):
    def __init__(self):
        self.__dict__.update(ppt.__dict__)
        self.__dict__.update(RSPlotManager.__dict__)

    def show(*args, **kwargs):
        from control import RSControl
        if RSControl.thread is None:
            ppt.show(*args, **kwargs)
        else:
            RSControl.thread.pause()
            ppt.show(*args, **kwargs)
            RSControl.thread.resume()


plt = RSPlotManager()


def printf(s, *args, **kwargs):
    from control import RSControl
    if RSControl.thread is not None:
        RSControl.thread.pause()
    if isinstance(s, str):
        try:
            print(s % args, **kwargs)
        except Exception as e:
            print(s, *args, **kwargs)
    else:
        print(str(s), **kwargs)
    if RSControl.thread is not None:
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


class GlobalOption(PydmConfig):
    def __init__(self, file_path, *args, **kwargs):
        PydmConfig.__init__(self, file_path, *args, **kwargs)
        self.pd_config = pd.core.config._global_config

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]
        else:
            pd_config = self.pd_config
            return pd_config[item]


def test():
    pass


