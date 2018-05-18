import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gc


class RSObject(object):
    modedict = {'default': 0, 'highlight': 1, 'bold': 2, 'nobold': 22, 'underline': 4, 'nounderline': 24,
                'blink': 5, 'noblink': 25, 'inverse': 7, 'noinverse': 27}
    colordict = {'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'pink': 5, 'cyan': 6, 'white': 7,
                 'default': 8, 'random': -1}

    def __init__(self, name='RS-Object', msgforecolor='default', msgbackcolor='default', msgmode='default'):
        self.name = name
        self.msgforecolor = msgforecolor
        self.msgbackcolor = msgbackcolor
        self.msgmode = RSObject.modedict[msgmode]
        self.timestart = time.time()

    def _getcolor(self, colorname):
        """
        transfer color name into color num
        :param colorname: str or int
        :return:color num
        """
        if isinstance(colorname, str):
            color = RSObject.colordict[colorname]
        else:
            color = colorname
        if color == -1:
            color = random.randint(0, 8)
        return color

    def _colorstr(self, s, mode, fcolor, bcolor):
        fcolor = self._getcolor(fcolor)
        bcolor = self._getcolor(bcolor)
        s = '\033[%d;%d;%dm%s\033[0m' % (mode, fcolor+30, bcolor+40, s)
        return s

    def msg(self, msg):
        cname = self._colorstr(self.name, self.msgmode, self.msgforecolor, self.msgbackcolor)
        msg = '%s: %s' % (cname, msg)
        print(msg)

    def _submsg(self, subtitle, forecolor, msg):
        cname = self._colorstr(self.name, self.msgmode, self.msgforecolor, self.msgbackcolor)
        csubtitle = self._colorstr(subtitle, 0, forecolor, 48)
        msg = '%s[%s]: %s' % (cname, csubtitle, msg)
        print(msg)

    def warning(self, msg):
        self._submsg('warning', 3, msg)

    def error(self, msg):
        self._submsg('error', 1, msg)
        raise Exception(msg)

    def starttimer(self):
        self.timestart = time.time()

    def msgtimecost(self, start=None, msg=''):
        if start == None:
            start = self.timestart
        self._submsg('timecost', 5, '%fs %s' %(time.time() - start, msg))

    def msgtime(self, msg=''):
        localtime = time.asctime(time.localtime(time.time()))
        self._submsg(localtime, 6, msg)
        print(msg)


class RSDataProcessor(RSObject):
    def __init__(self, features2process=None, name='DataProcessor', msgforecolor='default',
                 msgbackcolor='default', msgmode='default'):
        """
        :param features2process:需要处理的特征
                        如果None，则处理所有特征
        """
        super(RSDataProcessor, self).__init__(name, msgforecolor, msgbackcolor, msgmode)
        self.features2process = features2process

    def _getFeaturesNLabel(self, data):
        """
        :param data:
        :param features2process: 需要处理的特征子集
                            为None则设置为data所有features
                            否则为feature2process∩data.columns
        :return:features, label
        """
        if self.features2process is None:
            features = data.columns[:-1]
        else:
            features = [i for i in self.features2process if i in data.columns]
        label = data.columns[-1]
        return features, label

    def fit_transform(self, data):
        """
        :param data: [X y]
        :return:[X' y]
        """
        self.error('Not implemented!')


class RSData(pd.DataFrame, RSObject):
    def __init__(self, name='RSData', data=None, index=None, columns=None, dtype=None,
                 copy=False):
        super(RSData, self).__init__(data=None, index=None, columns=None, dtype=None, copy=False)
        RSObject.__init__(self, name, 'random', 'default', 'underline')


