import numpy as np
from rsobject import *

class AbnormalValueHandler(RSObject):
    def __init__(self, name='AbnormalValueHandler'):
        super(AbnormalValueHandler, self).__init__(name, 'black', 'cyan')

    def fit_transform(self):
        raise Exception('error: Not implemented!')


class AVHConfidence(AbnormalValueHandler):
    def __init__(self, confidence):
        '''
        对于某个特征，把在置信区间之外的数据设置为nan
        :param confidence:0~100
        '''
        super(AVHConfidence, self).__init__('拒绝域数据置NaN')
        self.confidence = confidence

    def fit_transform(self, data):
        '''
        :param data: pandas.DataFrame([feature1, feature2,...]), 不能包含target
        '''
        data = data.copy()
        confidence = self.confidence
        confidence /= 2.0
        low, up = data.quantile(confidence / 100), data.quantile(1 - confidence / 100)
        todrop = (((data.notna()) & (data < low)) | ((data.notna()) & (data > up)))
        self.msg('totally dropped %d items.' % (todrop.sum().sum()))
        data[todrop] = np.nan
        return data