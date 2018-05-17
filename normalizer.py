from base import *
from sklearn.preprocessing import MinMaxScaler


class Normalizer(RSDataProcessor):
    def __init__(self, name='Normalizer'):
        super(Normalizer, self).__init__(name, 'black', 'pink', 'highlight')


class NmlzMinMax(Normalizer):
    def __init__(self):
        super(NmlzMinMax, self).__init__('MinMax归一化')

    def fit_transform(self, data, featlist):
        self.starttimer()
        data = data.copy()
        for feat in featlist:
            scl = MinMaxScaler(feature_range=(0, 1))
            xx = data[feat]
            notnaIndexs = xx[xx.notna()].index
            nn = scl.fit_transform(xx[notnaIndexs].reshape(-1, 1))
            data.loc[notnaIndexs, [feat]] = scl.fit_transform(xx[notnaIndexs].reshape(-1, 1))
            # print('normalize %s OK!\t\t 非空值个数:%d' % (feat, notnaIndexs.shape[0]))
        self.msgtimecost()
        return data
