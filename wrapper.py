from dataprocessor import *
import sklearn.model_selection as cv
from sklearn.base import TransformerMixin, ClassifierMixin, ClusterMixin, RegressorMixin
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import BaseCrossValidator
from misc import ConfusionMatrix, ROCCurve, PRCurve, FBetaScore
from control import RSControl


class Wrapper(RSDataProcessor):
    def __init__(self, features2process, processor, name=''):
        """
        包装器，负责把其他数据处理器包装成DataProcess，如包装sklearn.svm.SVC成DataProcessor
        :param features2process:
        :param name:
        """
        RSDataProcessor.__init__(self, features2process, name, 'random', 'random', 'blink')
        self.processor = processor
        # set(str)，在进行_process之前会对这里面的参数进行非None检查
        # 如果任一其中的参数值为None，则抛出异常
        # 可以通过 << 运算符依次传入参数
        self.not_none_attrs = set()
        self.data = None

    def fit_transform(self, data):
        b_fit = True
        for x in self.not_none_attrs:
            if getattr(self, x) is None:
                b_fit = False
                break
        if b_fit:
            return RSDataProcessor.fit_transform(self, data)
        else:
            self.data = data
            return self

    def __rshift__(self, other):
        """
        not_none_attrs中的attr存在值为None时抛出异常
        :param other:
        :return:
        """
        none_attrs = [x for x in self.not_none_attrs if getattr(self, x) is None]
        if len(none_attrs) > 0:
            self.error('invoke of >> is forbidden,\
params %s are None whereas they are not allowed to be None.You may use << to fill they up.' % none_attrs.__str__())
        else:
            data = self.fit_transform(self.data)
            if not isinstance(other, int):
                data = data >> other
            self.data = None
            return data

    def __lshift__(self, other):
        """
        赋值other到not_none_attrs中第一个为None的attr
        :param other: 输入参数
        :return:
        """
        if isinstance(other, dict):
            self.__dict__.update(other)
        else:
            for attr in self.not_none_attrs:
                if getattr(self, attr) is None:
                    setattr(self, attr, other)
                    break
        return self


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
        self.cost_estimator = CETime.get_estimator(self.name, [processor])

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
        data.drop(columns=features, inplace=True)
        X = data.__class__(pd.DataFrame(X, columns=colsname))
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
        self.plots = None
        if self.processor is not None:
            self.cost_estimator = CETime.get_estimator(self.name, [self.processor])

    def _process(self, data, features, label):
        """
        warning： 此函数不同于通用DataProcess的同名函数
        :param data:  可以为[X y]，或者(trainset, testset)
        :return:  ClfResult
        """
        if isinstance(data, tuple):
            self.train_set, self.test_set = data[0], data[1]
        else:
            self.train_set, self.test_set = cv.train_test_split(data, test_size=self.test_size)
        X_train, X_test, y_train, y_test = self.train_set[features], self.test_set[features], \
                                           self.train_set[label], self.test_set[label]
        X_train, X_test = X_train.values, X_test.values
        self._run(X_train, y_train, X_test, y_test)
        self.show_result()
        return data

    def _run(self, X_train, y_train, X_test, y_test):
        if self.b_train:
            self.msg('training...')
            self.processor.fit(X_train, y_train)
        self.msg('scoring on train set...')
        self.trainscore = self.processor.score(X_train, y_train)
        self.msg('predicting test X...')
        if hasattr(self.processor, 'predict_proba'):
            self.y_prob = self.processor.predict_proba(X_test)
            self.y_pred = self.processor.classes_[self.y_prob.argmax(axis=1)]
        else:
            self.y_prob = None
            self.y_pred = self.processor.predict(X_test)
        self.testscore = (self.y_pred == y_test).sum() / y_test.shape[0]
        self.cm = ConfusionMatrix(y_test, self.y_pred, self.processor.classes_, name='CM of %s' % self.name)
        self.fbs = FBetaScore(y_test, self.y_pred, 'FBS of %s' % self.name)
        self.plots = [self.cm, self.fbs]
        if self.y_prob is not None:
            self.roc = ROCCurve(y_test, self.y_prob[:, -1], title='ROC of %s' % self.name)
            self.pr = PRCurve(y_test, self.y_prob[:, -1], title='PR of %s' % self.name)
            self.plots.extend([self.roc, self.pr])
        else:
            self.roc = None
            self.pr = None

    def show_result(self):
        self.msg('%f' % (self.trainscore * 100), '训练集得分')
        self.msg('%f' % (self.testscore * 100), '测试集得分')
        # plots
        naxes = self.plots.__len__()
        apr = 3  # axes per row
        if naxes % apr != 0 and naxes % 2 == 0:
            apr = 2
        nrows, ncols = divmod(naxes, apr)
        nrows += 1
        ncols = apr if nrows > 1 else ncols
        fig = plt.figure(figsize=(5.5*ncols, 5*nrows))
        for i, axe in enumerate(self.plots):
            axe.plot(fig.add_subplot(nrows, ncols, i+1))
        RSControl.show()

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
        self.cost_estimator = CETime.get_estimator(func)

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


class WrpSearchCV(Wrapper):
    def __init__(self, features2process, validator, name=''):
        if name == '':
            name = validator.__class__.__name__
        name = 'WrpSCV-%s' % name
        Wrapper.__init__(self, features2process, validator, name)
        self.cost_estimator = CETime.get_estimator(self.name, [validator])

    def _process(self, data, features, label):
        self.processor.fit(data[features].values, data[label].values)
        self.msg('\n%s' % RSTable(pd.DataFrame(data=self.processor.best_params_, index=['value'])), 'best params')
        self.msg('%.3f' % self.processor.best_score_, 'best score')
        return data


class WrpCluster(Wrapper):
    def __init__(self, features2process, cluster, name=''):
        """
        Wrapper for cluster
        :param features2process:
        :param cluster:
        :param name:
        """
        if name == '':
            name = cluster.__class__.__name__
        name = 'WrpClu-%s' % name
        Wrapper.__init__(self, features2process, cluster, name)

    def _process(self, data, features, label):
        return data


class WrpCrossValidator(WrpClassifier, pd.DataFrame):
    def __init__(self, features2process, validator, estimator=None, name=''):
        """
        Wrapper for Cross Validator
        :param features2process:
        :param validator:
        :param estimator: default RandomForestClassifier
        :param name:
        """
        pd.DataFrame.__init__(self)
        WrpClassifier.__init__(self, features2process, estimator, name=name)
        self.not_none_attrs = {'processor'}
        self.cv = validator
        self._get_estimator()

    def _get_estimator(self):
        if self.processor is not None and self.cv is not None:
            name = self.name
            if name == '':
                name = '%s-%s' % (self.cv.__class__.__name__,
                                  self.processor.__class__.__name__)
            name = 'WrpCV-%s' % name
            self.name = name
            self.cost_estimator = CETime.get_estimator(self.name, [self.processor, self.cv])
        else:
            self.name = ''

    def _process(self, data, features, label):
        """
        _process will not run until self.estimator is not None
        :param data:
        :param features:
        :param label:
        :return:
        """
        X, y = data[features].values, data[label].values
        train_scores, test_scores, roc_aucs = [], [], []
        good_samples, bad_samples = np.array([]), np.array([])
        n = self.cv.get_n_splits()
        self.msg(self.processor.__class__.__name__, 'estimator')
        for i, (train_i, test_i) in enumerate(self.cv.split(X, y)):
            self.msg('@1%d/%d ' % (i+1, n))
            train_X, train_y, test_X, test_y = X[train_i], y[train_i], X[test_i], y[test_i]
            self._run(train_X, train_y, test_X, test_y)
            train_scores.append(self.trainscore)
            test_scores.append(self.testscore)
            if self.roc is not None:
                roc_aucs.append(self.roc.auc())
            good_samples = np.concatenate([good_samples, test_i[self.y_pred == test_y]])
            bad_samples = np.concatenate([bad_samples, test_i[self.y_pred != test_y]])
        self.msg('@1')  # clear involatile message
        self.good_samples = data.iloc[good_samples, ]
        self.bad_samples = data.iloc[bad_samples, ]
        pd.DataFrame.__init__(self, data={'train_score': train_scores,
                                          'test_score': test_scores,
                                          'roc_auc': roc_aucs})
        self.msg('\n%s' % RSTable(self.mean()), 'mean')
        return data

    def __lshift__(self, other):
        ret = Wrapper.__lshift__(self, other)
        self._get_estimator()
        return ret


class WrpUnknown(Wrapper):
    def __init__(self, features2process, obj):
        Wrapper.__init__(self, features2process, obj, name='WrpUnknown-%s' % obj.__class__.__name__)

    def _process(self, data, features, label):
        self.warning('No operation occurred.')
        return data


def wrap(features2process, processor, *args, **kwargs):
    """
    intelligently wrapping， select wrapping type automatically
    """
    if isinstance(processor, RSDataProcessor):
        return processor
    if isinstance(processor, ClusterMixin):
        return WrpCluster(features2process, processor)
    elif isinstance(processor, wrap.__class__):
        return WrpFunction(features2process, processor)
    elif isinstance(processor, ClassifierMixin):
        return WrpClassifier(features2process, processor)
    elif isinstance(processor, TransformerMixin):
        return WrpDataProcessor(features2process, processor, bXonly=False)
    elif isinstance(processor, BaseSearchCV):
        return WrpSearchCV(features2process, processor)
    elif isinstance(processor, BaseCrossValidator):
        return WrpCrossValidator(features2process, processor, *args, **kwargs)
    else:
        return WrpUnknown(features2process, processor)


def test():
    return
    pro = wrap(None, lambda self, data, features: data * 2)(123)
