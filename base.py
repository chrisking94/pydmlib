#coding=utf-8
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gc
import os
import warnings
warnings.filterwarnings("ignore")


class RSObject(object):
    modedict = {'default': 0, 'highlight': 1, 'bold': 2, 'nobold': 22, 'underline': 4, 'nounderline': 24,
                'blink': 5, 'noblink': 25, 'inverse': 7, 'noinverse': 27}
    colordict = {'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'pink': 5, 'cyan': 6, 'white': 7,
                 'default': 8, 'random': -1}
    id_count = 9999

    def __init__(self, name='RS-Object', msgforecolor='default', msgbackcolor='default', msgmode='default'):
        self.name = name
        self.msgforecolor = msgforecolor
        self.msgbackcolor = msgbackcolor
        self.msgmode = RSObject.modedict[msgmode]
        self.timestart = time.time()
        self.coloredname = self.colorstr(self.name, self.msgmode, self.msgforecolor, self.msgbackcolor)
        self.id = RSObject.id_count
        RSObject.id_count += 1

    def _submsg(self, title, title_color, msg):
        csubtitle = self.colorstr(title, 0, title_color, 48)
        msg = '%s[%s]: %s' % (self.coloredname, csubtitle, msg)
        print(msg)

    def msg(self, msg, title=''):
        if title == '':
            msg = '%s: %s' % (self.coloredname, msg)
            print(msg)
        else:
            self._submsg(title, 'cyan', msg)

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
        self._submsg('timecost', 5, '%s %s' %(timecost, msg))

    def msgtime(self, msg=''):
        self._submsg(self.strtime(), 6, msg)

    def is_me(self, id_name):
        if isinstance(id_name, str):
            return self.name == id_name
        else:
            return self.id == id

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
        s = '\033[%d;%d;%dm%s\033[0m' % (mode, fcolor+30, bcolor+40, s)
        return s

    @staticmethod
    def strtime(format_='%Y-%m-%d %H:%M:%S', houroffset=12):
        t = datetime.datetime.now() + datetime.timedelta(hours=houroffset)
        t = t.strftime(format_)
        return t


class RSDataProcessor(RSObject):
    def __init__(self, features2process=None, name='RSDataProcessor', msgforecolor='default',
                 msgbackcolor='default', msgmode='default'):
        """
        :param features2process:需要处理的特征
                        如果None，则处理所有特征
        :param name:
        :param msgforecolor:
        :param msgbackcolor:
        :param msgmode:
        """
        RSObject.__init__(self, name, msgforecolor, msgbackcolor, msgmode)
        self.features2process = features2process
        self.state = 'on'  # disable this processor by set state to 'off'

    def _getFeaturesNLabel(self, data):
        """
        ignored
        :return:features, label
                features: 如果self.features2process为None则设置成data.columns
                            否则为feature2process∩data.columns
        """
        if self.features2process is None:
            features = data.columns[:-1]
        else:
            features = [i for i in self.features2process if i in data.columns]
        label = data.columns[-1]
        if isinstance(data, RSData):
            data.addhistory(self.coloredname)
        return features, label

    def _process(self, data, features, label):
        self.error('Not implemented!')

    def turn(self, state):
        """
        change processor state
        :param state: str, several value to choose as following
                        'on': turn on processor, fit_transform will works properly,
                                also get_report ...
                        'off': opposite to 'on'
        :return:
        """
        self.state = state

    def fit_transform(self, data):
        """
        :param data: [X y]
        :return:[X' y]
        """
        if self.state == 'off':
            return data
        self.starttimer()
        self.msgtime('running...')
        features, label = self._getFeaturesNLabel(data)
        data = self._process(data, features, label)
        self.msgtimecost()
        return data

    def get_report_title(self, *args):
        """
        返回当前对象输出报告的标题，可以用于制表，一般用在ModelTester中
        :return: list, 默认返回list[父类类名]
        """
        if self.state == 'off':
            return []
        return [self.__class__.__bases__[0].__name__]

    def get_report(self):
        """
        输出报告，一般用在ModelTester中
        :return:  list, 默认返回当前对象名
        """
        if self.state == 'off':
            return []
        return [self.name]

    def __call__(self, *args, **kwargs):
        return  self.fit_transform(*args)


class RSDataMetaclass(type):
    def __new__(cls, name, bases, attrs):
        func_dict = {}
        cls._rfaf([pd.DataFrame], func_dict)
        func_dict = [(k, cls._wrap_return(v)) for (k, v) in func_dict.items() if k not in attrs.keys()]
        attrs.update(func_dict)
        return type(name, bases, attrs)

    @staticmethod
    def _wrap_return(func):
        def wrappedfunc(self, *arg, **kwargs):
            ret = func(self, *arg, **kwargs)
            if isinstance(ret, pd.DataFrame):
                return RSData(self.name, ret, checkpoints=self.checkpoints)
            else:
                return ret
        return wrappedfunc

    @staticmethod
    def _rfaf(classes_, dict_ret):  # recursively find all functions
        bases = []
        for c in classes_:
            funcs = [(k, v) for (k, v) in c.__dict__.items()
                             if v.__class__.__name__=='function'
                                and k not in dict_ret.keys()]
            dict_ret.update(funcs)
            bases.extend(c.__bases__)
        if bases.__len__() > 0:
            RSDataMetaclass._rfaf(bases, dict_ret)


class RSData(pd.DataFrame, RSObject):#, metaclass=RSDataMetaclass):
    def __init__(self, name='RSData', data=None, index=None, columns=None, dtype=None,
                 copy=False, checkpoints=None):
        pd.DataFrame.__init__(self, data, index, columns, dtype, copy)
        RSObject.__init__(self, name, 'random', 'default', 'underline')
        if checkpoints is None:
            self.checkpoints = self.CheckPointMgr(self)
        else:
            self.checkpoints = checkpoints
        self.checkpoints['<root>']._save('false checkpoint, no content in.', True)

    def addhistory(self, info):
        self.checkpoints.addhistory(info)

    def toDataFrame(self):
        return super(RSData, self).copy()

    class CheckPointMgr(dict, RSObject):
        def __init__(self, wrapperobj):
            dict.__init__(self)
            RSObject.__init__(self, '%s.CheckPoints',
                              wrapperobj.msgforecolor,
                              wrapperobj.msgbackcolor,
                              'default')
            self.wrapperobj = wrapperobj
            self.lastcheckpoint = ''
            self.unsavedcheckpoint = self.CheckPoint(self, 'unsaved', None)

        def pop(self, pointname, **kwargs):
            if pointname == '<root>':
                self.error('cannot remove protected checkpoint <root>')
                return
            if pointname in self.keys():
                if pointname == self.lastcheckpoint:
                    self.lastcheckpoint = dict.__getitem__(self, pointname).parent
                dict.pop(self, pointname)
            else:
                self.error('No such check point <%s>.' % pointname)

        def addhistory(self, info):
            """
            add history info to current point
            :param info:
            :return:
            """
            self.unsavedcheckpoint.addhistory(info)

        class CheckPoint(object):
            def __init__(self, wrapperobj, pointname, parent):
                object.__init__(self)
                self.wrapperobj = wrapperobj
                self.name = pointname
                self.time = None
                self.parent = None
                self.history = []
                self.data = None
                if parent is not None:
                    parent._addchild(self)
                self.children = []

            def _addchild(self, child):
                child.parent = self
                child.history = self.history
                if child not in self.children:
                    self.children.append(child)

            def _removechild(self, child):
                self.children.remove(child)

            def _save(self, comment, bfalsepoint, data=None):
                """
                to create false point
                :param comment:
                :param bfalsepoint:
                :param data: backup outside data
                :return:
                """
                self.comment = comment
                if self.data is not None:
                    del self.data
                if not bfalsepoint:
                    if data is None:
                        self.data = pd.DataFrame.copy(self.wrapperobj.wrapperobj)
                    else:
                        self.data = data
                if self.name not in self.wrapperobj.keys():
                    dict.__setitem__(self.wrapperobj, self.name, self)
                if self.parent is not None:
                    self.parent._addchild(self)
                self.wrapperobj.lastcheckpoint = self.name
                self.wrapperobj.unsavedcheckpoint = \
                    RSData.CheckPointMgr.CheckPoint(self.wrapperobj, 'unsaved', self)
                self.time = RSData.strtime()

            def save(self, comment='', data=None):
                """
                back up data and set this checkpoint as current
                :param comment: comment for this checkpoint
                :return:
                """
                self._save(comment, False, data)

            def recover(self):
                """
                recover from this checkpoint, and set current checkpoint to this
                :return:
                """
                rsdata = self.wrapperobj.wrapperobj
                super(RSData, rsdata).__init__(self.data)
                self.wrapperobj.lastcheckpoint = self.name
                self.wrapperobj.unsavedcheckpoint = \
                    RSData.CheckPointMgr.CheckPoint(self.wrapperobj, 'unsaved', self)
                return self.wrapperobj.wrapperobj

            def drop(self):
                """
                remove this checkpoint and delete data it's holding
                if this is current checkpoint,
                drop() will move current point to this point's parent
                :return:
                """
                self.parent._removechild(self)
                del self.data
                self.wrapperobj.pop(self.name)

            def addhistory(self, info):
                self.history.append(info)

            def briefinfo(self):
                s = self.name
                if self.name == self.wrapperobj.lastcheckpoint:
                    s += '*'
                s = '%s\t%s\t-%s' % (s, str(self.time), self.comment)
                return s

            def detail(self):
                s = self.briefinfo()
                s += '\n--operation trace: '
                for track in self.history:
                    s += ' ⇒ %s' % track
                if self.history.__len__() == 0:
                    s+= 'None'
                return s

            def __str__(self):
                s = self.detail()
                s += '\n%s' % self.data.__str__()
                return s

        def __getitem__(self, pointname):
            if pointname in dict.keys(self):
                return dict.__getitem__(self, pointname)
            else:
                self.unsavedcheckpoint.name = pointname
                return self.unsavedcheckpoint

        def __str__(self):
            slist = ['%s.CheckPoints<%d point(s)>:\n' % (self.wrapperobj.coloredname, self.__len__())]
            lstitem = list(self.items())
            lstitem.sort(key=lambda x:x[1].time)
            for i, (k, v) in enumerate(lstitem):
                slist.append('\t%d.%s\n' % (i+1, v.briefinfo()))
            return ''.join(slist)

    def __str__(self):
        if self.checkpoints.lastcheckpoint != '':
            s = '%s('+ self.checkpoints.lastcheckpoint + '): \n%s'
        else:
            s = '%s: \n%s'
        return s %(self.coloredname, super(RSData, self).__str__())

    __repr__ = __str__


class RSList(RSObject, list):
    def __init__(self, copyfrom=()):
        """
        RS-List
        :param iterable:
        """
        RSObject.__init__(self, 'RS-List', 'random', 'random')
        list.__init__(self, copyfrom)

    def __getitem__(self, item):
        """
        get item
        :param item: id or index
        :return:
        """
        if (isinstance(item, int) and item>9999) or isinstance(item, str):  # by id
            ret = [x for x in self if isinstance(x, RSObject) and x.is_me(item)]
            if ret.__len__() > 0:
                return ret[0]
            else:
                self.error('No such item id=%s' % str(item))
        elif isinstance(item, slice):
            return self.__class__(copyfrom=list.__getitem__(self, item))
        else:
            return list.__getitem__(self, item)

    def __setitem__(self, key, value):
        if isinstance(key, int) and key>9999:  # by id
            for i, x in enumerate(self):
                if isinstance(x, RSObject) and x.is_me(key):
                    key = i
        list.__setitem__(self, key, value)

    def get_index(self, id_index):
        if (isinstance(id_index, int) and id_index>9999) or isinstance(id_index, str):  # by id
            for i, x in enumerate(self):
                if isinstance(x, RSObject) and x.is_me(id_index):
                    return i
            return None
        else:
            return id_index

    def copy(self):
        return self.__class__(copyfrom=list.copy(self))


def test():
    data = [[1,2],[3,4],[5,6]]
    data = RSData('R', data, columns=['A', 'B'])
    pass


