from base import *
#from sklearn.ensemble import IsolationForest
from sklearn.ensemble.iforest import IsolationForest


class OutlierHandler(RSDataProcessor):
    def __init__(self, features2process, name='OutlierHandler'):
        RSDataProcessor.__init__(self, features2process, name, 'black', 'cyan')

    def _process(self, data, features, label):
        data = data.copy()
        X = data[features]
        todrop = self._detect(X)
        # 用data[todrop] = np.nan会很慢
        # 优化
        X[todrop] = np.nan
        data[features] = X
        self.msg('totally dropped %d items.' % (todrop.sum().sum()))
        return data

    def _detect(self, X):
        """
        检测异常，在子类中实现
        :param X:
        :return: 检测后的真值表，异常值用True表示
        """
        self.error('Not implemented!')


class OHConfidence(OutlierHandler):
    def __init__(self, features2process, alpha):
        """
        对于features2process中的特征，把在置信区间之外的数据设置为NaN
        :param alpha:0~100
        """
        OutlierHandler.__init__(self, features2process, '置信区间异常检测，α=%.2f' % alpha)
        self.alpha = alpha

    def _detect(self, X):
        alpha = self.alpha
        alpha /= 2.0
        low, up = X.quantile(alpha / 100), X.quantile(1 - alpha / 100)
        todrop = (X < low) | (X > up)
        return todrop


class OH3Sigma(OutlierHandler):
    def __init__(self, features2process):
        OutlierHandler.__init__(self, features2process, '3σ异常检测，|x-u|>3σ')

    def _detect(self, X):
        #  若数据服从正态分布
        #  P（|x-u|>3σ）<= 0.003 为极小概率事件
        todrop = (X - X.mean()).abs() > 3*X.std()
        return todrop


class OHBox(OutlierHandler):
    def __init__(self, features2process):
        OutlierHandler.__init__(self, features2process, 'BoxPlot异常检测，x<Ql-1.5IQR or x>Qu+1.5IQR')

    def _detect(self, X):
        # 不在[Ql-1.5IQR ~ Qu+1.5IQR]的为异常值
        Ql, Qu = X.quantile(0.25), X.quantile(0.75)
        IQR = Qu - Ql
        todrop = (X<Ql-1.5*IQR) | (X>Qu+1.5*IQR)
        return todrop


class OHIForest(OutlierHandler):
    def __init__(self, features2process, **kwargs):
        OutlierHandler.__init__(self, features2process, 'IForest异常检测')
        self.iforest = IsolationForest(**kwargs)

    def _detect(self, X):
        self.iforest.fit(X)
        todrop = self.iforest.predict(X) == -1
        todrop = pd.DataFrame(np.array([todrop for i in range(X.shape[1])]).T, columns=X.columns)
        return todrop

