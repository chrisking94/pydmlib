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

    def msg(self, msg):
        msg = '\033[%d;%d;%dm%s\033[0m: %s' % (self.msgmode, self.msgforecolor + 30,
                                               self.msgbackcolor + 40, self.name, msg)
        print(msg)

    def warning(self, msg):
        msg = '\033[%d;%d;%dm%s\033[0m[\033[1;33mwarning\033[0m]: %s' % (self.msgmode, self.msgforecolor + 30,
                                               self.msgbackcolor + 40, self.name, msg)
        print(msg)

    def error(self, msg):
        msg = '\033[%d;%d;%dm%s\033[0m[\033[1;31merror\033[0m]: %s' % (self.msgmode, self.msgforecolor + 30,
                                                                         self.msgbackcolor + 40, self.name, msg)
        print(msg)
        raise Exception(msg)

    def starttimer(self):
        self.timestart = time.time()

    def msgtimecost(self, start=None, msg=''):
        if start == None:
            start = self.timestart
        self.msg('%s耗时: %f s' % (msg, time.time() - start))

    def msgtime(self, msg=''):
        localtime = time.asctime(time.localtime(time.time()))
        msg = '\033[%d;%d;%dm%s\033[0m[\033[1;36m%s\033[0m]: %s' % (self.msgmode, self.msgforecolor + 30,
                                                                   self.msgbackcolor + 40, self.name,
                                                                   localtime, msg)
        print(msg)



