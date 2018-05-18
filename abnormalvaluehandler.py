from base import *


class AbnormalValueHandler(RSDataProcessor):
    def __init__(self, features2process, name='AbnormalValueHandler'):
        super(AbnormalValueHandler, self).__init__(features2process, name, 'black', 'cyan')


class AVHConfidence(AbnormalValueHandler):
    def __init__(self, features2process, confidence):
        """
        对于某个特征，把在置信区间之外的数据设置为NaN
        :param confidence:0~100
        """
        super(AVHConfidence, self).__init__(features2process, '拒绝域数据置NaN')
        self.confidence = confidence

    def fit_transform(self, data):
        """
        :param datatp: pandas.DataFrame([feature1, feature2,...]), 不能包含target
        """
        data = data.copy()
        feats, label = self._getFeaturesNLabel(data)
        datatp = data[feats]
        confidence = self.confidence
        confidence /= 2.0
        low, up = datatp.quantile(confidence / 100), datatp.quantile(1 - confidence / 100)
        todrop = (((datatp.notna()) & (datatp < low)) | ((datatp.notna()) & (datatp > up)))
        self.msg('totally dropped %d items.' % (todrop.sum().sum()))
        data[todrop] = np.nan
        return data
