from base import *


class NanHandler(RSDataProcessor):
    def __init__(self, featues2process, name='NanHandler'):
        super(NanHandler, self).__init__(featues2process, name, 'yellow', 'black')


class NHDrop(NanHandler):
    def __init__(self, featues2process):
        super(NHDrop, self).__init__(featues2process, '丢弃带NaN行')

    def fit_transform(self, data):
        '''
        丢弃features2process列中，含有nan值的行
        '''
        self.msg('sample count before dropping: %d' % data.shape[0])
        features, label = self._getFeaturesNLabel(data)
        data = data[data[features].dropna().index]
        self.msg('sample count after dropping: %d' % data.shape[0])
        return data


class NHToSpecial(NanHandler):
    def __init__(self, featues2process, value=0):
        '''
        :param value: 替代nan的值
        '''
        super(NHToSpecial, self).__init__(featues2process, '设置NaN为%d' % (value))
        self.special = value

    def fit_transform(self, data):
        features, label = self._getFeaturesNLabel(data)
        data[features] = data[features].fillna(self.special)
        self.msg('data shape %s' % (data.shape.__str__()))
        return data
