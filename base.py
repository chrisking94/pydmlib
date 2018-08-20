# coding=utf-8
import time
import datetime
import matplotlib
from .utils import plt, printf, pd, np, cfg
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
    mode_dict = {'default': 0, 'highlight': 1, 'bold': 2, 'nobold': 22, 'underline': 4, 'nounderline': 24,
                 'blink': 5, 'noblink': 25, 'inverse': 7, 'noinverse': 27}
    color_dict = {'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'pink': 5, 'cyan': 6, 'white': 7,
                  'default': 8, 'random': -1}
    id_count = 1
    _internal_attrs = {'name'}

    def __init__(self, name='', msgforecolor='default', msgbackcolor='default', msgmode='default'):
        self._name = ''
        self._fore_color = msgforecolor
        self._back_color = msgbackcolor
        self._msg_mode = RSObject.mode_dict[msgmode]
        self._time_start = time.time()
        self._id = RSObject.id_count
        self._colored_name = ''
        self.name = name
        RSObject.id_count += 1

    def _print_msg(self, title, title_color, msg):
        if title == '':
            msg = '%s: %s' % (self.colored_name, msg)
        else:
            csubtitle = self.color_str(title, 0, title_color, 48)
            msg = '%s[%s]: %s' % (self.colored_name, csubtitle, msg)
        printf(msg)

    def msg(self, msg, title=''):
        self._print_msg(title, 'blue', msg)

    def warning(self, msg):
        self._print_msg('warning', 3, msg)

    def error(self, msg):
        self._print_msg('error', 1, msg)
        raise Exception(msg)

    def start_timer(self):
        self._time_start = time.time()

    def msg_time_cost(self, start=None, msg=''):
        if start is None:
            start = self._time_start
        timecost = time.time() - start
        if timecost < 1:
            timecost = '%.2fs' % round(timecost, 2)
        else:
            m, s = divmod(timecost, 60)
            h, m = divmod(m, 60)
            timecost = '%02d:%02d:%02d' % (h, m, s)
        self._print_msg('timecost', 5, '%s %s' % (timecost, msg))

    def msg_current_time(self, msg=''):
        self._print_msg(self.str_current_time(), 6, msg)

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

    #################
    #   built-ins   #
    #################
    def __str__(self):
        return self.colored_name

    def __repr__(self):
        return self.__str__()

    #################
    #   Properties  #
    #################
    # Read Write
    @property
    def name(self):
        name = self._name
        if name == '':
            name = self.__class__.__name__
            name = cfg.translate_object_name(name)
            name = '%s#%d' % (name, self.id)
        return name

    @name.setter
    def name(self, s):
        self._name = s
        self._colored_name = self.color_str(self.name, self._msg_mode, self._fore_color, self._back_color)

    # Read Only
    @property
    def id(self):
        return self._id

    @property
    def colored_name(self):
        return self._colored_name

    ##############
    #   Statics  #
    ##############
    @staticmethod
    def get_color(str_color):
        """
        transfer color name into color num
        :param str_color: str or int
        :return:color num
        """
        if isinstance(str_color, str):
            color = RSObject.color_dict[str_color]
        else:
            color = str_color
        if color == -1:
            color = random.randint(0, 8)
        return color

    @staticmethod
    def color_str(s, mode, f_color, b_color):
        f_color = RSObject.get_color(f_color)
        b_color = RSObject.get_color(b_color)
        s = '\033[%d;%d;%dm%s\033[0m' % (mode, f_color + 30, b_color + 40, s)
        return s

    @staticmethod
    def str_current_time(format_='%Y-%m-%d %H:%M:%S', hour_offset=0):
        t = datetime.datetime.now() + datetime.timedelta(hours=hour_offset)
        t = t.strftime(format_)
        return t


class RSList(RSObject, list):
    def __init__(self, copyfrom=()):
        """
        RS-List
        :param copyfrom:
        """
        RSObject.__init__(self, 'RS-List')
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
        if isinstance(id_index, str):  # by name
            for i, x in enumerate(self):
                if isinstance(x, RSObject) and x.is_me(id_index):
                    return i
            return None
        else:  # by index
            return id_index

    def copy(self, deep=False):
        return self.__class__(copyfrom=list.copy(self))

    def info(self):
        return pd.Series(self)

    def __str__(self):
        return '%s：\n%s' % (self.colored_name, RSTable(self.info()).__str__())

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

    def _reshape_1d(self, data: Iterable) -> pd.DataFrame:
        pass


class RSThread(Thread, RSObject):
    def __init__(self, **kwargs):
        Thread.__init__(self, **kwargs)
        RSObject.__init__(self)
        self.state = 'pause'

    def stop(self):
        try:
            self._async_raise(self.ident, SystemExit)
        except SystemError:
            self.warning('cannot stop thread!')

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



