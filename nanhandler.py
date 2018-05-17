from base import *


class NanHandler(RSDataProcessor):
    def __init__(self, name='NanHandler'):
        super(NanHandler, self).__init__(name, 'yellow', 'black')

    def fit_transform(self, data):
        '''
        :param data:[X y]
        '''
        raise Exception('error: Not implemented!')


class NHDrop(NanHandler):
    def __init__(self):
        super(NHDrop, self).__init__('丢弃带NaN行')

    def fit_transform(self, data):
        '''
        丢弃含有nan值的行
        '''
        self.msg('sample count before dropping: %d' % data.shape[0])
        data = data.dropna()
        self.msg('sample count after dropping: %d' % data.shape[0])
        return data


class NHToSpecial(NanHandler):
    def __init__(self, value=0):
        '''
        :param value: 替代nan的值
        '''
        super(NHToSpecial, self).__init__('设置NaN为%d' % (value))
        self.special = value

    def fit_transform(self, data):
        data = data.fillna(self.special)
        self.msg('data shape: %s' % (data.shape.__str__()))
        return data
