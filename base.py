#coding=utf-8
import time
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import random
import gc
import os
import inspect


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
        self.coloredname = self._colorstr(self.name, self.msgmode, self.msgforecolor, self.msgbackcolor)

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
        msg = '%s: %s' % (self.coloredname, msg)
        print(msg)

    def _submsg(self, subtitle, forecolor, msg):
        csubtitle = self._colorstr(subtitle, 0, forecolor, 48)
        msg = '%s[%s]: %s' % (self.coloredname, csubtitle, msg)
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
        timecost = time.time() - start
        if timecost<1:
            timecost = '%fs' % round(timecost,1)
        else:
            m, s = divmod(timecost, 60)
            h, h = divmod(m, 60)
            timecost = '%02d:%02d:%02d' % (h, m, s)
        self._submsg('timecost', 5, '%s %s' %(timecost, msg))

    def msgtime(self, msg=''):
        localtime = time.asctime(time.localtime(time.time()))
        self._submsg(localtime, 6, msg)
        print(msg)

    def unfoldIpynb(self, notebook, module=None, xpath=''):
        """
        unfold source in jupyter, after this cmd please refresh the web browser
        :param notebook: ?.ipynb, the ?
        :param module: e.g. sklearn.svm
        :param xpath: extra path, if target *.py does not exist, use module+xpath
                        e.g. sklearn.svm.aaa, aaa.py isn't there in sklearn.svm
                            so we use :
                                unfoldIpynb('xxx', sklearn.svm, 'aaa')
                            or use like:
                                unfoldIpynb('xxx', None, '/usr/aaa/bbb'
        :return: None
        """
        if module is None:
            modulepath = xpath
        else:
            modulepath = inspect.getsourcefile(self.__class__.__name__)
            if modulepath.endswith('__init__.py'):
                modulepath = modulepath[:-('__init__.py'.__len__())]
            else:
                modulepath += '/'
            modulepath += xpath
        ipynbpath = os.getcwd() + '/' + notebook + '.ipynb'


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

    __call__ = fit_transform


class RSData(pd.DataFrame, RSObject):
    def __init__(self, name='RSData', data=None, index=None, columns=None, dtype=None,
                 copy=False):
        super(RSData, self).__init__(data, index, columns, dtype, copy)
        RSObject.__init__(self, name, 'random', 'default', 'underline')
        self.checkpoints = self.CheckPointMgr(self)
        self.bmodified = False
        self.checkpoints['origin'].save('original data')

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
            if pointname == 'origin':
                self.error('cannot remove protected checkpoint <origin>')
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
            dict.__getitem__(self, self.lastcheckpoint).addhistory(info)

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

            def save(self, comment=''):
                """
                back up data and set this checkpoint as current
                :param comment: comment for this checkpoint
                :return:
                """
                self.comment = comment
                if self.data is not None:
                    del self.data
                self.data = pd.DataFrame.copy(self.wrapperobj.wrapperobj)
                if self.name not in self.wrapperobj.keys():
                    dict.__setitem__(self.wrapperobj, self.name, self)
                if self.parent is not None:
                    self.parent._addchild(self)
                self.wrapperobj.lastcheckpoint = self.name
                self.wrapperobj.unsavedcheckpoint =\
                    RSData.CheckPointMgr.CheckPoint(self.wrapperobj, 'unsaved', self)
                self.time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

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
                return '%s\t%s\t-%s' % (self.name, str(self.time), self.comment)

            def detail(self):
                s = ''
                return s

            def __str__(self):
                return self.data.__str__()

            __repr__ = __str__

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



def test():
    data = [[1,2],[3,4]]
    data = RSData('R', data, columns=['A', 'B'])
    print(data)
    data.checkpoints['origin'].save('kk')
    print(data.checkpoints['origin'])
    print(data.checkpoints)
    data.loc[0, 'A'] = 9
    for i in range(1000000):
        for j in range(100):
            pass
    data.checkpoints['modified'].save('this is comment.')
    print(data)
    data.checkpoints['origin'].recover()
    print(data)
    print(data.checkpoints)
    pass


