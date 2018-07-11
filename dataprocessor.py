from base import *
from control import CStandbyCursor, CTimer


class RSDataProcessor(RSObject):
    cursor = CStandbyCursor()
    timer = CTimer()
    cursor.visible = False
    timer.visible = False

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

    def turn(self, state):
        """
        change processor state
        :param state: str, several value to choose as following
                        'on': turn on processor, fit_transform will works properly,
                                also get_report ...
                        'off': opposite to 'on'
                        'disable': when disabled, it can't be turned 'on' or 'off'
                        'enable': opposite to 'disable', self will be turned 'on' when state changed from 'disable' to 'enable'
        :return:
        """
        if self.state == 'disable':
            if state == 'enable':
                state = 'on'
            else:
                self.warning('processor was disabled.')
                return
        self.state = state

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
                if RSDataProcessor.cursor is not None:
                    RSDataProcessor.cursor.visible = True
                    RSDataProcessor.timer.reset()
                    RSDataProcessor.timer.visible = True
                self.starttimer()
                self.msgtime('running...')
                self.print_params()
                # 筛选处理特征：
                # 如果self.features2process为None则设置成data.columns
                # 否则为features2process∩data.columns
                if self.features2process is None:
                    features = data.columns[:-1]
                elif isinstance(self.features2process, str):
                    features = data.columns[:-1][self.features2process]
                    if self.features2process[1] == '@':
                        self.msg(features.__str__(), self.features2process)
                elif isinstance(self.features2process, tuple):
                    features = data.columns[:-1][self.features2process]
                    self.msg(features.__str__(), self.features2process[0])
                else:
                    features = data.columns[self.features2process]
                label = data.columns[-1]
                if features.__len__() == 0:
                    self.warning('No feature to process.')
                else:
                    data = self._process(data, features, label)
                if RSDataProcessor.cursor is not None:
                    RSDataProcessor.cursor.visible = False
                    RSDataProcessor.timer.visible = False
            else:
                data = self._process(data, None, None)
            self.msgtimecost()
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

    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args)

    def __add__(self, other):
        from integration import ProcessorSequence
        return ProcessorSequence([self, other])


