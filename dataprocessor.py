﻿from .base import *
from .control import CStandbyCursor, CTimer, CLabel, CTimeProgressBar, RSControl
from .costestimator import CETime
from .data import RSData


class RSDataProcessor(RSObject):
    cursor = CStandbyCursor(visible=False)
    timer = CTimer(visible=False)
    label = CLabel(visible=False)
    progressbar = CTimeProgressBar(visible=False, width=0)
    involatile_msg = [''] * 10  # 0~9
    b_multi_line_msg = False  # output each message in a new line
    s_msg_mode = 'brief'  # brief detail
    n_msg_max_len = 50  # unit: char
    time_estimation_thread = None

    class TimeEstimationThread(RSThread):
        def __init__(self, **kwargs):
            RSThread.__init__(self, **kwargs)
            self._estimator = None
            self.task_commit_time = 0

        def run(self):
            while True:
                if self.estimator is not None:
                    estimator = self.estimator
                    tcp = estimator.predict()
                    tcp -= time.time() - self.task_commit_time
                    if estimator == self.estimator:
                        self._estimator = None
                        if tcp > 1:
                            RSDataProcessor.progressbar.width = 40
                            RSDataProcessor.progressbar.reset(tcp)
                    else:
                        RSDataProcessor.progressbar.width = 0
                time.sleep(0.2)

        @property
        def estimator(self):
            return self._estimator

        @estimator.setter
        def estimator(self, e):
            self._estimator = e
            self.task_commit_time = time.time()

    def __init__(self, features2process=None, name='', msgforecolor='default',
                 msgbackcolor='default', msgmode='default'):
        """
        :param features2process:需要处理的特征
                        如果None，则处理所有特征
        :param name:
        :param msgforecolor:
        :param msgbackcolor:
        :param msgmode:
        """
        RSObject.__init__(self, name, 'random', 'default', 'underline')
        self.features2process = features2process
        self.state = 'on'  # turn off this processor by set state to 'off'
        self.messages = {}
        self._factors = None
        self.factors = []

    def turn(self, state):
        """
        change processor state
        :param state: str, several value to choose as following
                        'on': turn on processor, fit_transform will works properly,
                                also get_report ...
                        'off': opposite to 'on'
                        'disable': when disabled, it can't be turned 'on' or 'off'
                        'enable': opposite to 'disable', self will be turned 'on' when
                        state changed from 'disable' to 'enable'
        :return:
        """
        if self.state == 'disable':
            if state == 'enable':
                state = 'on'
            else:
                self.warning('processor was disabled.')
                return
        self.state = state

    def msg(self, msg, title=''):
        if title == '':
            if len(msg) > 1 and msg[0] == '@':
                RSDataProcessor.involatile_msg[int(msg[1])] = msg[2:]
            s_involatile = ''.join(RSDataProcessor.involatile_msg)
            RSDataProcessor.label.text = lambda: '%s<%s%s>: %s %s%s' % \
                                                 (self.colored_name,
                                                  self.color_str(RSDataProcessor.cursor.__str__(), 0, 6, 8),
                                                  self.color_str(RSDataProcessor.timer.__str__(), 0, 2, 8),
                                                  self.progressbar.__str__(),
                                                  s_involatile,
                                                  msg)
        else:
            RSObject.msg(self, msg, title)

    def _print_msg(self, title, title_color, msg):
        if self.s_msg_mode == 'disable':
            return
        if RSDataProcessor.b_multi_line_msg:
            RSObject._print_msg(self, title, title_color, msg)
        else:
            title = self.color_str(title, 0, title_color, 8)
            if title not in self.messages.keys():
                self.messages[title] = []
            self.messages[title].append(msg)

    def msg_time_cost(self, start=None, msg=''):
        RSObject.msg_time_cost(self, start, msg)
        if not RSDataProcessor.b_multi_line_msg:
            # out put messages
            sl = ['[%s] %s' % (x[0], ', '.join(x[1])) for x in self.messages.items()]
            RSObject._print_msg(self, '', 0, '  '.join(sl))
            self.messages = {}

    def _msg_features(self, features, columns, title):
        if self.s_msg_mode == 'brief':
            msg = '%d/%d column(s)' % (len(features), len(columns)-1)
        else:
            msg = features.__str__()
        self.msg(msg, title)

    def _process(self, data, features, label):
        self.error('Not implemented!')

    def fit_transform(self, data):
        """
        :param data: pd.DataFrame或其他
                    1.DataFrame, 格式为[X y]，这种情况会按如下参数调用子类_process()
                        self._process(data, features, label)
                        并且会输出通用的运行提示，如running...
                    2.其他，其他任何类型数据，这个数据将交给子类的_process()处理，调用方式如下：
                        self._process(data, None, None)
        :return:[X' y]
        """
        if self.state == 'on':
            if isinstance(data, pd.DataFrame):
                self.start_timer()
                self.factors.clear()
                if not isinstance(data, RSData):
                    data = RSData(data)
                RSDataProcessor.label.visible = True
                RSDataProcessor.timer.reset()
                self.msg('running...')
                self.print_params()
                # 筛选处理特征：
                # 如果self.features2process为None则设置成data.columns
                # 否则为features2process∩data.columns
                features = data.columns[:-1]
                features = features[self.features2process]
                if isinstance(features, str):
                    features = RSData.Index([features])
                if isinstance(self.features2process, str):
                    self._msg_features(features, data.columns, self.features2process)
                elif isinstance(self.features2process, tuple):
                    self._msg_features(features, data.columns, self.features2process[0])
                label = data.columns[-1]
                if features.__len__() == 0:
                    self.warning('No feature to process.')
                else:
                    self.factors.extend(data.shape)
                    self.time_estimation_thread.estimator = self.cost_estimator
                    try:
                        self.cost_estimator.start_timer()
                        data = self._process(data, features, label)
                        self.cost_estimator.memorize_experience()
                    except Exception as e:
                        raise e
                    finally:
                        self.cost_estimator.abandon_experience()
                        self.time_estimation_thread.estimator = None
                        RSDataProcessor.label.visible = False
                        self.progressbar.width = 0
            else:
                data = self._process(data, None, None)
            RSDataProcessor.label.visible = False
            self.progressbar.width = 0
            self.msg_time_cost()
        else:
            self.msg(self.state, 'state')
        return data

    def print_params(self):
        pl = []
        exclude = {'self', 'features2process', 'name', 'args', 'kwargs'}
        vdict = vars(self)
        for name in self.__init__.__code__.co_varnames:
            if name not in exclude and name in vdict.keys():
                value = vdict[name].__str__()
                if value.__len__() < 10:
                    pl.append('%s=%s' % (name, value))
        if pl.__len__() > 0:
            self.msg(', '.join(pl), 'params')

    def get_report_title(self, *args):
        """
        返回当前对象输出报告的标题，可以用于制表，一般用在ModelTester中
        :return: list, 默认返回list[父类类名]
        """
        if self.state == 'on':
            return [self.__class__.__bases__[0].__name__]
        return []

    def get_report(self):
        """
        输出报告，一般用在ModelTester中
        :return:  list, 默认返回当前对象名
        """
        if self.state == 'on':
            return [self.name]
        return []

    #################
    #   Properties  #
    #################
    @property
    def cost_estimator(self):
        ce = CETime.get_estimator(self.__class__.__name__)
        ce.factors = self.factors
        return ce

    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, lst):
        if lst is None:
            self._factors = None
        elif isinstance(lst, Iterable):
            self._factors = CETime.Factors(lst)
        else:
            raise ValueError('factors should be Iterable!')

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    def __add__(self, other):
        from .integration import ProcessorSequence
        return ProcessorSequence([self, other])

    @staticmethod
    def init():
        if RSDataProcessor.time_estimation_thread is not None:
            RSDataProcessor.time_estimation_thread.stop()
        RSDataProcessor.time_estimation_thread = RSDataProcessor.TimeEstimationThread()
        RSDataProcessor.time_estimation_thread.start()


