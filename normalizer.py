from base import *
from transformer import TsfmFunction


class Normalizer(RSDataProcessor):
    def __init__(self, features2process, name='Normalizer', forecolor='black'):
        RSDataProcessor.__init__(self, features2process, name, forecolor, 'pink', 'highlight')


class FeatureNormalizer(Normalizer, TsfmFunction):
    def __init__(self, features2process, name='FeatureNormalizer'):
        Normalizer.__init__(self, features2process, name, 'blue')
        TsfmFunction.__init__(self, features2process, self._transform, breplace=True)

    def _transform(self, X):
        self.error('Not implemented!')


class FNMinMax(FeatureNormalizer):
    def __init__(self, features2process):
        FeatureNormalizer.__init__(self, features2process, 'MinMax归一')

    def _transform(self, X):
        return (X - X.min()) / (X.max() - X.min())


class FNAtan(FeatureNormalizer):
    def __init__(self, features2process):
        FeatureNormalizer.__init__(self, features2process, 'Atan归一')

    def _transform(self, X):
        return np.emath.arctanh(X) * 2 / np.pi


class FNZScore(FeatureNormalizer):
    def __init__(self, feature2process):
        FeatureNormalizer.__init__(self, feature2process, 'Z-Score标准化')

    def _transform(self, X):
        return (X - X.mean()) / X.std()

