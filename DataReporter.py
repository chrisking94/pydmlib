from base import *


class DataReporter(RSDataProcessor):
    def __init__(self, features2process, name='DataReporter'):
        super(DataReporter, self).__init__(features2process, name, 'blue', 'black', 'default')


class DRBrief(DataReporter):
    def __init__(self, features2process):
        super(DRBrief, self).__init__(features2process, '简洁数据报告')

    def fit_transform(self, data):
        self.starttimer()
        features, label = self._getFeaturesNLabel(data)
        X, y = data[features], data[label]
        self.msgtimecost()
        return data