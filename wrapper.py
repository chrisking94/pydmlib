from base import *
import sklearn.model_selection as cv
from reporter import ClfResult


class Wrapper(RSDataProcessor):
    def __init__(self, features2process, processor, name='Wrapper'):
        """
        包装器，负责把其他数据处理器包装成DataProcess，如包装sklearn.svm.SVC成DataProcessor
        :param features2process:
        :param name:
        """
        RSDataProcessor.__init__(self, features2process, name, 'random', 'random', 'blink')
        self.processor = processor


class WrpDataProcessor(Wrapper):
    def __init__(self, features2process, processor, name='', bXonly=True):
        """
        包装数据处理器，处理器的格式必须为
        class DP:
                    .
                    .
            def fit_transform(X, [y)
                return X
                    .
                    .
        :param features2process:
        :param processor: 处理器
        :param name:
        :param bXonly: processor是否只用输入X，默认True
        """
        if name == '':
            if bXonly:
                name = 'WrpDPX-%s'
            else:
                name = 'WrpDPXy-%s'
            name = name % (processor.__class__.__name__)
        Wrapper.__init__(self, features2process, processor, name)
        self.bXonly = bXonly

    def _process(self, data, features, label):
        """
        处理后的列会被重命名
        :param data:
        :return:
        """
        self.msg('running...')
        data = data.copy()
        X, y = data[features], data[label]
        if self.bXonly:
            X = self.processor.fit_transform(X)
        else:
            X = self.processor.fit_transform(X, y)
        if not isinstance(X, np.ndarray):
            self.error('processor\'s return value type must be np.ndarray!')
        colsname = ['%s_%d' % (self.name, x) for x in range(X.shape[1])]
        data = data.drop(columns=features)
        X = pd.DataFrame(X, columns=colsname)
        data = pd.concat([X, data], axis=1)
        return data


class WrpClassifier(Wrapper):
    def __init__(self, features2process, classifier, name='', test_size=0.2, b_train=True):
        """
        分类器的包装器
        :param features2process:
        :param clf:
        :param name:
        :param test_size: 测试集大小
        :param b_train: 是否训练分类器，可以直接输入训练好的分类器
        """
        if name == '':
            name = 'WrpClf-%s' % classifier.__class__.__name__
        Wrapper.__init__(self, features2process, classifier, name)
        self.test_size = test_size
        self.b_train = b_train

    def fit_transform(self, data):
        """
        warning： 此函数不同于通用DataProcess的同名函数
        :param data:  可以为[X y]，或者(trainset, testset)
        :return:  ClfResult
        """
        self.starttimer()
        if isinstance(data, tuple):
            trainset, testset = data[0], data[1]
            features, label = self._getFeaturesNLabel(trainset)
            X_train, X_test, y_train, y_test = trainset[features], testset[features], trainset[label], testset[label]
        else:
            features, label = self._getFeaturesNLabel(data)
            X_train, X_test, y_train, y_test = cv.train_test_split(data[features], data[label], test_size=self.test_size, random_state=0)
        if self.b_train:
            self.msg('training...')
            self.processor.fit(X_train, y_train)
        self.msg('testing on train set...')
        self.trainscore = self.processor.score(X_train, y_train)
        self.msg('predicting on test set...')
        if hasattr(self.processor, 'predict_proba'):
            y_prob = self.processor.predict_proba(X_test)
            y_pred = self.processor.classes_[y_prob.argmax(axis=1)]
        else:
            y_prob = None
            y_pred = self.processor.predict(X_test)
        self.testscore = (y_pred == y_test).sum() / y_test.shape[0]
        self.msg('%f' % (self.trainscore * 100), '训练集得分')
        self.msg('%f' % (self.testscore * 100), '测试集得分')
        self.msgtimecost()
        return ClfResult(self.processor.classes_, self.trainscore, self.testscore, y_prob, y_pred, y_test, self.name)

    def get_report_title(self, *args):
        return ['训练集得分', '测试集得分']

    def get_report(self):
        return [self.trainscore, self.testscore]


def IWrap(features2process, processor):
    """
    intelligently wrapping， select wrapping type automatically
    """
    if hasattr(processor, 'predict'):
        return WrpClassifier(features2process, processor)
    elif hasattr(processor, 'fit_transform'):
        return WrpDataProcessor(features2process, processor, bXonly=False)
    else:
        raise Exception('IWrap failed, invalid processor %s object.' % processor.__class__.__name__)

