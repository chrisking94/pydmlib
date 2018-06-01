from base import *
import sklearn.model_selection as ms


class FormatConverter(RSDataProcessor):
    def __init__(self, features2process=None, name='FormatConverter'):
        RSDataProcessor.__init__(self, features2process, name, 'green', 'blue', 'highlight')


class FCovTrainTestSet(FormatConverter):
    def __init__(self, features2process, test_size=0.2, random_state=0):
        """
        data ←→ (trainset, testset)
        :param features2process:
        """
        FormatConverter.__init__(self, features2process, 'data←→(trainset, testset)')
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


class FCovDataTarget(FormatConverter):
    target = None

    def __init__(self, features2process, bXonly=False):
        """
        data←→(X y)
        :param features2process:
        :param bXonly: 1、仅返回 X， data -> X，y会被存储到FCovDataTarget.target；
                        2、或输入只有X，(X FCovDataTarget.target) -> data。
                        * bXonly在流式处理中必须成对使用，以避免数据混乱而导致错误。
        """
        FormatConverter.__init__(self, features2process, 'data←→(X y)')
        self.bXonly = bXonly

    def fit_transform(self, data):
        """
        see __init__
        :param data: data；
                     (X y)；
                     (X) 这种情况返回[X FCovDataTarget.target]
        :return:
        """
        self.starttimer()
        if isinstance(data, tuple):
            features = [x for x in self.features2process if x in data[0].columns]
            X = data[0][features]
            if data.__len__() == 1:
                y = self.target
                if y is None:
                    self.error('there is no [target] stored in!')
                elif y.shape[0] != X.shape[0]:
                    self.error('(y.len=%d) must be same with (X.len=%d)!' %(y.shape[0], X.shape[0]))
            elif data.__len__() == 2:
                y = data[1]
            else:
                y = None
                self.error('format of input tuple should be (X y)')
            self.msg('(X y) → data')
            ret = pd.concat([X, y], axis=1)
        else:
            features, label = self._getFeaturesNLabel(data)
            if self.bXonly:
                self.msg('data → X')
                ret = data[features]
            else:
                self.msg('data → (X y)')
                ret = (data[features], data[label])
        self.msgtimecost()
        return ret