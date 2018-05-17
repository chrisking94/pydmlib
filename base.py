import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class RSObject(object):
    modedict = {'default': 0, 'highlight': 1, 'bold': 2, 'nobold': 22, 'underline': 4, 'nounderline': 24,
                'blink': 5, 'noblink': 25, 'inverse': 7, 'noinverse': 27}
    colordict = {'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'pink': 5, 'cyan': 6, 'white': 7,
                 'default': 8, 'random': 9}

    def __init__(self, name='RS-Object', msgforecolor='default', msgbackcolor='default', msgmode='default'):
        self.name = name
        self.msgforecolor = self._getcolor(msgforecolor)
        self.msgbackcolor = self._getcolor(msgbackcolor)
        self.msgmode = RSObject.modedict[msgmode]
        self.timestart = time.time()

    def _getcolor(self, colorname):
        color = RSObject.colordict[colorname]
        if(color == 9):
            color = random.randint(0, 8)
        return color

    def _colorstr(self, s, mode, fcolor, bcolor):
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
    def __init__(self):
        super(RSDataProcessor, self).__init__('DataProcessor')

    def _getXy(self, data, features=None):
        if features==None:
            features = data.columns[:-1]
        X = data[features]
        y = data[data.columns[-1]]
        return X, y

    def fit_transform(self, data, features=None):
        '''
        处理数据的主要成员函数
        :param data: [X y]
        :param features:需要处理的特征，可以为None
                        如果None，则处理所有特征
        :return:处理过后的[X' y]
        '''
        self.error('Not implemented!')


class RSData(RSObject, pd.DataFrame):
    def __init__(self):
        super(RSData, self).__init__('RSData', 'random', 'default')

