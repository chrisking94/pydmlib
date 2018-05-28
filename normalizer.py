from base import *
from transformer import TsfmFunction


class Normalizer(RSDataProcessor):
    def __init__(self, features2process, name='Normalizer', forecolor='black'):
        super(Normalizer, self).__init__(features2process, name, forecolor, 'pink', 'highlight')


class FeatureNormalizer(TsfmFunction):
    def __init__(self, features2process, name='FeatureNormalizer'):
        #super(FeatureNormalizer, self).__init__(features2process, name, 'blue')
        TsfmFunction.__init__(self, features2process, self._transform, breplace=True)

    def _transform(self, X):
        self.error('Not implemented!')


class FNMinMax(FeatureNormalizer):
    def __init__(self, features2process):
        super(FNMinMax, self).__init__(features2process, 'MinMax归一')

    def _transform(self, X):
        return (X - X.min()) / (X.max() - X.min())


class FNAtan(FeatureNormalizer):
    def __init__(self, features2process):
        super(FNAtan, self).__init__(features2process, 'Atan归一')

    def _transform(self, X):
        return np.emath.arctanh(X) * 2 / np.pi


class FNZScore(FeatureNormalizer):
    def __init__(self, feature2process):
        super(FNZScore, self).__init__(feature2process, 'Z-Score标准化')

    def _transform(self, X):
        return (X - X.mean()) / X.std()

