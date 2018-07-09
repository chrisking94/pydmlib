from dataprocessor import *
import sklearn.model_selection as cv
from reporter import ClfResult


class Wrapper(RSDataProcessor):
    def __init__(self, features2process, processor, name=''):
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

    def _process(self, data, features, label):
        """
        warning： 此函数不同于通用DataProcess的同名函数
        :param data:  可以为[X y]，或者(trainset, testset)
        :return:  ClfResult
        """
        if isinstance(data, tuple):
            trainset, testset = data[0], data[1]
            X_train, X_test, y_train, y_test = trainset[features], testset[features], trainset[label], testset[label]
        else:
            X_train, X_test, y_train, y_test = cv.train_test_split(data[features], data[label], test_size=self.test_size, random_state=0)
        if self.b_train:
            self.msg('training...')
            self.processor.fit(X_train, y_train)
        self.msg('scoring on train set...')
        self.trainscore = self.processor.score(X_train, y_train)
        self.msg('predicting test X...')
        if hasattr(self.processor, 'predict_proba'):
            y_prob = self.processor.predict_proba(X_test)
            y_pred = self.processor.classes_[y_prob.argmax(axis=1)]
        else:
            y_prob = None
            y_pred = self.processor.predict(X_test)
        self.testscore = (y_pred == y_test).sum() / y_test.shape[0]
        self.msg('%f' % (self.trainscore * 100), '训练集得分')
        self.msg('%f' % (self.testscore * 100), '测试集得分')
        return ClfResult(self.processor.classes_, self.trainscore, self.testscore, y_prob, y_pred, y_test, self.name)

    def get_report_title(self, *args):
        return ['训练集得分', '测试集得分']

    def get_report(self):
        return [self.trainscore, self.testscore]


class WrpFunction(Wrapper):
    def __init__(self, features2process, func, name=''):
        """
        函数包装器
        :param features2process:
        :param func:
        :param name:
        """
        if name == '':
            name = func.__name__
        name = 'WrpFunc-%s' % name
        Wrapper.__init__(self, features2process, func, name)

    def _process(self, data, features, label):
        param_names = self.processor.__code__.co_varnames
        params = []
        for param in param_names:
            if param == 'self':
                params.append(self)
            elif param == 'data':
                params.append(data)
            elif param == 'features':
                params.append(features)
            elif param == 'label':
                params.append(label)
        param_dict = dict(zip(param_names, params))
        return self.processor(**param_dict)


def IWrap(features2process, processor):
    """
    intelligently wrapping， select wrapping type automatically
    """
    if isinstance(processor, RSDataProcessor):
        return processor
    elif isinstance(processor, IWrap.__class__):
        return WrpFunction(features2process, processor)
    elif hasattr(processor, 'predict'):
        return WrpClassifier(features2process, processor)
    elif hasattr(processor, 'fit_transform'):
        return WrpDataProcessor(features2process, processor, bXonly=False)
    else:
        raise Exception('IWrap failed, invalid processor %s object.' % processor.__class__.__name__)


def test():
    return
    pro = IWrap(None, lambda self, data, features:data*2)(123)
