from base import *
from sklearn.preprocessing import MinMaxScaler


class Normalizer(RSDataProcessor):
    def __init__(self,features2process, name='Normalizer'):
        super(Normalizer, self).__init__(features2process, name, 'black', 'pink', 'highlight')


class NmlzMinMax(Normalizer):
    def __init__(self, features):
        super(NmlzMinMax, self).__init__('MinMax归一化')

    def fit_transform(self, data):
        self.starttimer()
        data = data.copy()
        features, label = self._getFeaturesNLabel(data)
        X = data[features]
        X = (X - X.min()) / (X.max() - X.min())
        data[features] = X
        self.msgtimecost()
        return data
