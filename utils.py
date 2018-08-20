import matplotlib.pyplot as plt
from configparser import ConfigParser
import os
import pandas as pd
import numpy as np
from xml.dom.minidom import parse
import xml.dom.minidom as minidom
from .error import *

###############################################################
# rewrite some methods of plt
plt_show = plt.show


def show(*args, **kwargs):
    from .control import RSControl
    if RSControl.thread is None:
        plt_show(*args, **kwargs)
    else:
        RSControl.thread.pause()
        plt_show(*args, **kwargs)
        RSControl.thread.resume()


plt.show = show


###############################################################
# create internal functions
def printf(s, *args, **kwargs):
    from .control import RSControl
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


class XMLUniqueNode(minidom.Element):
    def __init__(self, name, value=''):
        """
        node should be the unique one if its siblings
        :param name:
        :param value:
        """
        minidom.Element.__init__(self, name)
        self.value = value

    def _get_node(self, name):
        """
        create if not exists
        :param name:
        :return:
        """
        eles = self.getElementsByTagName(name)
        if len(eles) == 0:
            node = XMLUniqueNode(name, '')
            self.appendChild(node)
            return node
        elif len(eles) == 1:
            return eles[0]
        else:
            raise NotUniqueError('found several elements named [%s]:\n%s' % (name, str(eles)))

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return super(XMLUniqueNode, self).__getattr__(item)
        else:
            return self._get_node(item)

    def __setattr__(self, key, value):
        if key in self.__dict__.keys():
            super(XMLUniqueNode, self).__setattr__(key, value)
        else:
            node = self._get_node(key)
            node.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return '%s: %s' % (self.tagName, self.value)

    #################
    #   Properties  #
    #################
    @property
    def value(self):
        return self.getAttribute('value')

    @value.setter
    def value(self, v):
        self.setAttribute('value', v)


class XMLDocument(XMLUniqueNode):
    def __init__(self, file_path):
        super(XMLDocument, self).__init__('doc')
        self.file_path = file_path
        if os.path.exists(self.file_path):
            doc = minidom.parse(file_path)
            for node in doc.childNodes:
                self.appendChild(node)

    def save(self):
        pass


class GlobalOption(PydmConfig):
    def __init__(self, file_path, *args, **kwargs):
        PydmConfig.__init__(self, file_path, *args, **kwargs)
        self.pd_config = pd.core.config._global_config
        self.pydm_options = {}

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


if __name__ == '__main__':
    from xml.dom.minidom import parse
    import xml.dom.minidom

    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse("movies.xml")
    collection = DOMTree.documentElement

    # 在集合中获取所有电影
    movies = collection.getElementsByTagName("movie")

    # 打印每部电影的详细信息
    for movie in movies:
        t = movie.getElementsByTagName('type')[0]
        print(type(t.childNodes[0]))
        format = movie.getElementsByTagName('format')[0]
        rating = movie.getElementsByTagName('rating')[0]
        description = movie.getElementsByTagName('description')[0]
    pass


