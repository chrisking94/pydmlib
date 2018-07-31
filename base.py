# coding=utf-8
import time
import datetime
import matplotlib
from  utils import plt, printf, pd, np
import random
import gc
import os
import warnings
import re
import copy
import prettytable as pt
from collections import Iterable
from threading import Thread
import inspect
import ctypes
from abc import abstractmethod

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.sans-serif'] = ['simhei'] # 用来正常显示中文标签
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


class RSObject(object):
    modedict = {'default': 0, 'highlight': 1, 'bold': 2, 'nobold': 22, 'underline': 4, 'nounderline': 24,
                'blink': 5, 'noblink': 25, 'inverse': 7, 'noinverse': 27}
    colordict = {'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'pink': 5, 'cyan': 6, 'white': 7,
                 'default': 8, 'random': -1}
    id_count = 9999

    def __init__(self, name='', msgforecolor='default', msgbackcolor='default', msgmode='default'):
        self.id = RSObject.id_count
        if name == '':
            name = '%s%d' % (self.__class__.__name__, self.id)
        self.name = name
        self.msgforecolor = msgforecolor
        self.msgbackcolor = msgbackcolor
        self.msgmode = RSObject.modedict[msgmode]
        self.timestart = time.time()
        self.coloredname = self.colorstr(self.name, self.msgmode, self.msgforecolor, self.msgbackcolor)
        RSObject.id_count += 1

    def _submsg(self, title, title_color, msg):
        if title == '':
            msg = '%s: %s' % (self.coloredname, msg)
        else:
            csubtitle = self.colorstr(title, 0, title_color, 48)
            msg = '%s[%s]: %s' % (self.coloredname, csubtitle, msg)
        printf(msg)

    def msg(self, msg, title=''):
        self._submsg(title, 'blue', msg)

    def warning(self, msg):
        self._submsg('warning', 3, msg)

    def error(self, msg):
        self._submsg('error', 1, msg)
        raise Exception(msg)

    def starttimer(self):
        self.timestart = time.time()

    def msgtimecost(self, start=None, msg=''):
        if start is None:
            start = self.timestart
        timecost = time.time() - start
        if timecost < 1:
            timecost = '%.2fs' % round(timecost, 2)
        else:
            m, s = divmod(timecost, 60)
            h, m = divmod(m, 60)
            timecost = '%02d:%02d:%02d' % (h, m, s)
        self._submsg('timecost', 5, '%s %s' % (timecost, msg))

    def msgtime(self, msg=''):
        self._submsg(self.strtime(), 6, msg)

    def is_me(self, id_name):
        if isinstance(id_name, str):
            if id_name[0] == '@':  # 正则匹配
                id_name = id_name[1:]
                return re.search(id_name, self.name) is not None
            else:
                return self.name == id_name
        elif isinstance(id_name, int):
            return self.id == id_name
        else:
            return self == id_name

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def __str__(self):
        return self.coloredname

    __repr__ = __str__

    @staticmethod
    def getcolor(colorname):
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

    @staticmethod
    def colorstr(s, mode, fcolor, bcolor):
        fcolor = RSObject.getcolor(fcolor)
        bcolor = RSObject.getcolor(bcolor)
        s = '\033[%d;%d;%dm%s\033[0m' % (mode, fcolor + 30, bcolor + 40, s)
        return s

    @staticmethod
    def strtime(format_='%Y-%m-%d %H:%M:%S', houroffset=0):
        t = datetime.datetime.now() + datetime.timedelta(hours=houroffset)
        t = t.strftime(format_)
        return t


class RSList(RSObject, list):
    def __init__(self, copyfrom=()):
        """
        RS-List
        :param copyfrom:
        """
        RSObject.__init__(self, 'RS-List', 'random', 'random')
        list.__init__(self, copyfrom)

    def __getitem__(self, item):
        """
        get item
        :param item: id or index
        :return:
        """
        if isinstance(item, slice):
            return self.__class__(copyfrom=list.__getitem__(self, item))
        else:
            index = self.get_index(item)
            if index is None:
                self.error('No such item [%s]' % str(item))
            else:
                return list.__getitem__(self, index)

    def __setitem__(self, key, value):
        key = self.get_index(key)
        list.__setitem__(self, key, value)

    def get_index(self, id_index):
        if (isinstance(id_index, int) and id_index > 9999) or isinstance(id_index, str):  # by id
            for i, x in enumerate(self):
                if isinstance(x, RSObject) and x.is_me(id_index):
                    return i
            return None
        else:
            return id_index

    def copy(self, deep=False):
        return self.__class__(copyfrom=list.copy(self))

    def info(self):
        return pd.Series(self)

    def __str__(self):
        return '%s：\n%s' % (self.coloredname, RSTable(self.info()).__str__())

    def __repr__(self):
        return self.__str__()


class RSTable(pt.PrettyTable, RSObject):
    def __init__(self, copy_from=None):
        pt.PrettyTable.__init__(self)
        self.max_width = 200
        if copy_from is None:
            pass
        elif isinstance(copy_from, pd.DataFrame):
            fn = ['index']
            fn.extend(copy_from.columns)
            self.field_names = fn
            for row in copy_from.itertuples():
                self.add_row(row)
        elif isinstance(copy_from, pd.Series):
            self.field_names = ['index', 'value']
            for row in copy_from.iteritems():
                self.add_row(row)
        elif isinstance(copy_from, dict):
            self.field_names = ['key', 'value']
            for row in copy_from.items():
                self.add_row(row)
        elif isinstance(copy_from, Iterable):
            n = len(copy_from)
        RSObject.__init__(self)


class RSThread(Thread, RSObject):
    def __init__(self, **kwargs):
        Thread.__init__(self, **kwargs)
        RSObject.__init__(self)
        self.state = 'pause'

    def stop(self):
        self._async_raise(self.ident, SystemExit)

    def pause(self):
        self.state = 'pause'

    def resume(self):
        self.state = 'running'

    @staticmethod
    def _async_raise(tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError('invalid thread id')
        elif res != 1:
            # """ if it returns a number greater than 1, you're in trouble. """
            # and you should call it again with exec=NULL to revert the effect
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError('PyThreadState_SetAsyncExc failed')


def test():
    return
    pass


