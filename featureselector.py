from base import *
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from skfeature.function.information_theoretical_based.MRMR import mrmr


class FeatureSelector(RSDataProcessor):
    def __init__(self, features2process, threshold=0.2, name='FeatureSelector'):
        super(FeatureSelector, self).__init__(features2process, name, 'pink', 'white', 'highlight')
        self.threshold = threshold

    def fit_transform(self, data):
        '''
        选择最佳特征
        :param encdata:输入数据
        :param scorer:str,评分器，有以下几个值
            None :不做选择
            'chi2':SelectKBest(chi2).scores
            'rfr':RandomForestClassifier.feature_importance_
            'mrmr':Minimum Redundancy Maximum Relevance Feature Selection
        :param threshold:float, 0~1
        '''
        self.starttimer()
        self.msg('--feature count before selection %d' % data.shape[1])
        features, label = self._getFeaturesNLabel(data)
        sdata, starget = data[features], data[label]
        scores = self.score(sdata, starget)

        scores = pd.Series(scores)
        # normalization
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores[scores.isnull()] = 0
        fdata = data.drop(columns=sdata.columns[scores < self.threshold])

        self.msg('--feature count after selection %d' % fdata.shape[1])
        self.msgtimecost()
        return fdata

    def score(self, data, target):
        self.error('Not implemented!')


class FSNone(FeatureSelector):
    def __init__(self, features2process):
        super(FSNone, self).__init__(features2process, name='不做特征选择')

    def fit_transform(self, data):
        self.msgtime()
        return data.copy()


class FSChi2(FeatureSelector):
    def __init__(self, features2process):
        super(FSChi2, self).__init__(features2process, name='χ²特征选择')

    def score(self, data, target):
        skb = SelectKBest(chi2, k='all')
        skb.fit_transform(data, target)
        return skb.scores_


class FSRFC(FeatureSelector):
    def __init__(self, features2process):
        super(FSRFC, self).__init__(features2process, name='rfc特征选择')

    def score(self, data, target):
        clf = RandomForestClassifier()
        clf.fit(data, target)
        return clf.feature_importances_


class FSmRMR(FeatureSelector):
    def __init__(self, features2process):
        super(FSmRMR, self).__init__(features2process, name='mRMR特征选择')

    def score(self, data, target):
        F = mrmr(data.values, target)
        scores = pd.Series(F)
        scores[pd.Index(F)] = np.arange(1, 0, float(-1) / F.shape[0])
        return scores
