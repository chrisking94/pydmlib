from base import *
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from skfeature.function.information_theoretical_based.MRMR import mrmr
from sklearn.linear_model import LogisticRegression


class FeatureSelector(RSDataProcessor):
    def __init__(self, features2process, threshold=0.2, name='FeatureSelector'):
        """"
        选择最佳特征
        :param threshold:float, 0~1
        """
        RSDataProcessor.__init__(self, features2process, name, 'pink', 'white', 'highlight')
        self.threshold = threshold

    def _process(self, data, features, label):
        feat_count0 = data.shape[1]
        sdata, starget = data[features], data[label]
        scores = self.score(sdata, starget)
        scores = pd.Series(scores)
        # normalization
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores[scores.isnull()] = 0
        fdata = data.drop(columns=sdata.columns[scores < self.threshold])
        self.msg('feature count  %d ==> %d' % (feat_count0, fdata.shape[1]))
        return fdata

    def score(self, data, target):
        self.error('Not implemented!')


class FSNone(FeatureSelector):
    def __init__(self, features2process):
        FeatureSelector.__init__(self, features2process, name='不做特征选择')

    def _process(self, data, features, label):
        self.msgtime()
        return data.copy()


class FSChi2(FeatureSelector):
    def __init__(self, features2process, threshold=0.2):
        FeatureSelector.__init__(self, features2process, threshold, name='χ²特征选择')

    def score(self, data, target):
        skb = SelectKBest(chi2, k='all')
        skb.fit_transform(data, target)
        return skb.scores_


class FSRFC(FeatureSelector):
    def __init__(self, features2process, threshold=0.2):
        FeatureSelector.__init__(self, features2process, threshold, name='rfc特征选择')

    def score(self, data, target):
        clf = RandomForestClassifier()
        clf.fit(data, target)
        return clf.feature_importances_


class FSmRMR(FeatureSelector):
    def __init__(self, features2process, threshold=0.2):
        FeatureSelector.__init__(self, features2process, threshold, name='mRMR特征选择')

    def score(self, data, target):
        F = mrmr(data.values, target)
        scores = pd.Series(F)
        scores[pd.Index(F)] = np.arange(1, 0, float(-1) / F.shape[0])
        return scores


class FSManual(FeatureSelector):
    def __init__(self, features2process, b_except=False, name='手动特征选择'):
        """
        select feature manually
        :param features2process:
        :param b_except: if True, select features who are not in features2process
        :param name:
        """
        FeatureSelector.__init__(self, features2process, name=name)
        self.b_except = b_except

    def _process(self, data, features, label):
        if self.b_except:
            features = list(set(data.columns) - set(features))
        self.msg('%d features remaining.' % features.__len__())
        ret = pd.concat([data[features], data[label]], axis=1)
        return ret


class FSL1Regularization(FeatureSelector):
    def __init__(self, features2process, C):
        """
        去掉L1正则化后w为0的特征
        :param features2process:
        :param C:
        """
        FeatureSelector.__init__(self, features2process, threshold=0.001, name='L1正则化特征选择')
        self.C = C

    def _score(self, data, target):
        clf = LogisticRegression(C=self.C)
        clf.fit(data, target)
        return clf.coef_.sum(axis=0)

