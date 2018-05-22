from base import *
import sklearn.model_selection as ms


class FormatConverter(RSDataProcessor):
    def __init__(self, features2process=None, name='FormatConverter'):
        super(FormatConverter, self).__init__(features2process, name, 'green', 'blue', 'highlight')


class FCoTrainTestSet(FormatConverter):
    def __init__(self, features2process, test_size=0.2, random_state=0):
        """
        data ←→ (trainset, testset)
        :param features2process:
        """
        super(FCoTrainTestSet, self).__init__(features2process, 'data←→(trainset, testset)')
        self.test_size = test_size
        self.random_state = random_state

    def fit_transform(self, data):
        self.starttimer()
        if isinstance(data, tuple):
            self.msg('(trainset, testset) → data')
            ret = pd.concat([data[0], data[1]], axis=0)
        else:
            self.msg('data → (trainset, testset)')
            ret = ms.train_test_split(data, test_size=self.test_size, random_state=self.random_state)
        self.msgtimecost()
        return ret


class FCoDataTarget(FormatConverter):
    def __init__(self, features2process):
        super(FCoDataTarget, self).__init__(features2process, 'data←→(X y)')

    def fit_transform(self, data):
        self.starttimer()
        if isinstance(data, tuple):
            self.msg('(X y) → data')
            features = [x for x in self.features2process if x in data[0].columns]
            ret = pd.concat([data[0][features], data[1]], axis=1)
        else:
            self.msg('data → (X y)')
            features, label = self._getFeaturesNLabel(data)
            ret = data[features], data[label]
        self.msgtimecost()
        return ret