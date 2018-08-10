import matplotlib.pyplot as plt
from configparser import ConfigParser
import os
import pandas as pd
import numpy as np

###############################################################
# rewrite some methods of plt
plt_show = plt.show


def show(*args, **kwargs):
    from control import RSControl
    if RSControl.thread is None:
        plt_show(*args, **kwargs)
    else:
        RSControl.thread.pause()
        plt_show(*args, **kwargs)
        RSControl.thread.resume()


plt.show = plt_show


###############################################################
# create internal functions
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


###############################################################
# configs
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

    def translate_object_name(self, name):
        """
        if exists section [Object] then name will be translated refers to config in [Object]
        :param name:
        :return: translated name
        """
        if self.has_option('Object', name):
            t_name = self.get('Object', name)
        else:
            t_name = ''
        if t_name == '':
            t_name = name
        return t_name


cfg = GlobalOption('./pydmlib.cfg')


def test():
    pass


