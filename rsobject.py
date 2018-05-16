import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RSObject(object):
    modedict = {'default': 0, 'highlight': 1, 'bold': 2, 'nobold': 22, 'underline': 4, 'nounderline': 24,
                'blink': 5, 'noblink': 25, 'inverse': 7, 'noinverse': 27}
    colordict = {'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'pink': 5, 'cyan': 6, 'white': 7,
                 'default': 8}

    def __init__(self, name='RS-Object', msgforecolor='default', msgbackcolor='default', msgmode='default'):
        self.name = name
        self.msgforecolor = RSObject.colordict[msgforecolor]
        self.msgbackcolor = RSObject.colordict[msgbackcolor]
        self.msgmode = RSObject.modedict[msgmode]
        self.timestart = time.time()

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



