from  dataprocessor import *
import sklearn.model_selection as ms


class FormatConverter(RSDataProcessor):
    def __init__(self, features2process=None, name=''):
        RSDataProcessor.__init__(self, features2process, name, 'green', 'blue', 'highlight')


class FCovTrainTestSet(FormatConverter):
    def __init__(self, features2process, test_size=0.2, random_state=0):
        """
        data → (trainset, testset), only features2process will be preserved.
        :param features2process:
        """
        FormatConverter.__init__(self, features2process)
        self.test_size = test_size
        self.random_state = random_state

    def _process(self, data, features, label):
        features = list(features)
        features.append(label)
        self.msg('data → (trainset, testset)')
        data = data[features]
        data = ms.train_test_split(data, test_size=self.test_size, random_state=self.random_state)
        return data


class FCovDataTarget(FormatConverter):
    def __init__(self, features2process):
        """
        data←→(X y)
        :param features2process:
        """
        FormatConverter.__init__(self, features2process)

    def _process(self, data, features, label):
        self.msg('data → (X y)')
        return data[features], data[label]

