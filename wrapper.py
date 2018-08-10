from .dataprocessor import *
import sklearn.model_selection as cv
from sklearn.base import TransformerMixin, ClassifierMixin, ClusterMixin, RegressorMixin
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection import BaseCrossValidator
from .misc import ConfusionMatrix, ROCCurve, PRCurve, FBetaScore


class Wrapper(RSDataProcessor):
    delay_fit_time_gap = 0.1  # s

    def __init__(self, features2process, processor, name=''):
        """
        包装器，负责把其他数据处理器包装成DataProcess，如包装sklearn.svm.SVC成DataProcessor
        :param features2process:
        :param name:
        """
        self.processor = processor
        RSDataProcessor.__init__(self, features2process, name, 'random', 'random', 'blink')
        # set(str)，在进行_process之前会对这里面的参数进行非None检查
        # 如果任一其中的参数值为None，则抛出异常
        # 可以通过 << 运算符依次传入参数
        self.not_none_attrs = set()
        self.data = None
        # delay one step of fit_transform
        self.b_fit = True
        self._factors = None
        self.factors = [processor]

    def fit_transform(self, data):
        if self.b_fit:
            return RSDataProcessor.fit_transform(self, data)
        else:
            self.b_fit = True
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
            self.data = None
            if not isinstance(other, int):
                data = data >> other
            else:
                if other == 0:
                    data = None
            return data

    def __lshift__(self, other):
        """
        :param other: input params, they could be:
                        1.none dict, other will be filled in the first null
                          attribute of self in self.not_none_attrs
                        2.dict, self.__dict___.update(other)
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

    def __str__(self):
        return self.processor.__str__()

    #################
    #   Properties  #
    #################
    @property
    def cost_estimator(self):
        ce = CETime.get_estimator(self.processor.__class__.__name__)
        ce.factors = self.factors
        return ce


class WrpDataProcessor(Wrapper):
    def __init__(self, features2process, processor, name=''):
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
        """
        if name == '':
            name = 'WrpDP-%s'
            name = name % processor.__class__.__name__
        Wrapper.__init__(self, features2process, processor, name)

    def _process(self, data, features, label):
        """
        处理后的列会被重命名
        :param data:
        :return:
        """
        data = data.copy()
        X, y = data[features], data[label]
        X_ = self.processor.fit_transform(X, y)
        if X.shape == X_.shape:
            data[features] = X_
        else:
            colsname = ['%s_%d' % (self.name, x) for x in range(X_.shape[1])]
            data.drop(columns=features, inplace=True)
            X = data.__class__(pd.DataFrame(X_, columns=colsname, index=data.index))
            data = pd.concat([X, data], axis=1)
            # !concat以index相等为条件来进行join
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
        self.label = ''

    def _process(self, data, features, label):
        """
        warning： 此函数不同于通用DataProcess的同名函数
        :param data:  可以为[X y]，或者(trainset, testset)
        :return:  ClfResult
        """
        self.label = label
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
        self.msg(self.label, 'target')
        self.msg('%.1f' % (self.trainscore * 100), 'train score')
        self.msg('%.1f' % (self.testscore * 100), 'test score')
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
        plt.show()

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

    #################
    #   Properties  #
    #################
    @property
    def cost_estimator(self):
        ce = CETime.get_estimator(self.processor)
        ce.factors = self.factors
        return ce

    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, lst):
        self._factors = CETime.Factors()


class WrpSearchCV(Wrapper):
    def __init__(self, features2process, validator, name=''):
        if name == '':
            name = validator.__class__.__name__
        name = 'WrpSCV-%s' % name
        Wrapper.__init__(self, features2process, validator, name)

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
    def __init__(self, features2process, validator, estimator):
        """
        Wrapper for Cross Validator
        :param features2process:
        :param validator: not None
        :param estimator: not None
        :param name:
        """
        pd.DataFrame.__init__(self)
        WrpClassifier.__init__(self, features2process, estimator)
        self.not_none_attrs = {'processor'}
        self.validator = validator
        self._reload_immutable_factors()

    def _reload_immutable_factors(self):
        if self.processor is not None and self.validator is not None:
            name = 'Wrp%s-%s' % (self.validator.__class__.__name__,
                                 self.processor.__class__.__name__)
            self.name = name
            self.factors = 'see property factor'
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
        n = self.validator.get_n_splits()
        self.msg(self.processor.__class__.__name__, 'estimator')
        for i, (train_i, test_i) in enumerate(self.validator.split(X, y)):
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

    #################
    #   Properties  #
    #################
    @property
    def cost_estimator(self):
        ce = CETime.get_estimator('%s-%s' % (self.validator.__class__.__name__, self.processor.__class__.__name__))
        ce.factors = self.factors
        return ce

    @property
    def factors(self):
        return self._factors

    @factors.setter
    def factors(self, lst):
        if self.processor is not None and self.validator is not None:
            self._factors = CETime.Factors([self.processor, self.validator])
        else:
            self._factors = None

    def __lshift__(self, other):
        ret = Wrapper.__lshift__(self, other)
        self._reload_immutable_factors()
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
        return WrpDataProcessor(features2process, processor)
    elif isinstance(processor, BaseSearchCV):
        return WrpSearchCV(features2process, processor)
    elif isinstance(processor, BaseCrossValidator):
        cv = processor
        predictor = None
        if len(args) > 0:
            predictor = args[0]
        return WrpCrossValidator(features2process, cv, predictor, **kwargs)
    elif isinstance(processor, RegressorMixin):
        pass
    else:
        return WrpUnknown(features2process, processor)


def test():
    return
    pro = wrap(None, lambda self, data, features: data * 2)(123)
