from dataprocessor import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from skfeature.function.information_theoretical_based.MRMR import mrmr
from sklearn.linear_model import LogisticRegression
from control import RSControl


class FeatureSelector(RSDataProcessor):
    def __init__(self, features2process, feature_count=0.02, plot=None, name=''):
        """"
        选择最佳特征
        :param feature_count: 2 types
                        1. float, 0~1,  feature_count = X.shape[1]*feature_count
                        2. int, 1~X.shape[0]
        :param plot: str
                        1.'pie'
                        2.'bar'
                        3.None (default)
        """
        RSDataProcessor.__init__(self, features2process, name, 'pink', 'white', 'highlight')
        self.feature_count = feature_count
        self.scores = None
        self.plot = plot

    def _process(self, data, features, label):
        feat_count0 = data.shape[1] - 1
        sdata, starget = data[features], data[label]
        scores = self.score(sdata, starget)
        scores = pd.Series(scores, index=features)
        # normalization
        scores /= scores.sum()
        self.scores = scores.sort_values(0, ascending=False)
        if scores.isnull().sum() != 0:
            self.error('scores contains null.')
        if self.feature_count < 1:
            feature_count = int(self.feature_count * features.__len__())
        else:
            feature_count = self.feature_count
        if self.plot == 'bar':
            self.bar(top=feature_count)
        elif self.plot == 'pie':
            self.pie(top=feature_count)
        fdata = data.drop(columns=scores.nsmallest(features.__len__()-feature_count).index)
        self.msg('%d ==> %d' % (feat_count0, fdata.shape[1] - 1), 'feature count')
        return fdata

    def score(self, data, target):
        self.error('Not implemented!')
        return np.array([])

    def _get_score_part(self, top):
        part = self.scores[self.scores > 0.01]
        if part.shape[0] > top:
            part = self.scores[:top]
        part = part.append(pd.Series([1 - part.sum()], index=['其他']))
        part.sort_values(ascending=True, inplace=True)
        return part

    def pie(self, top=10):
        part = self._get_score_part(top)
        labels = part.index
        fracs = part.values * 100
        figsize = part.shape[0] / 6
        plt.figure(figsize=(figsize, figsize))
        plt.subplot()
        plt.pie(fracs, labels=labels, autopct='%1.1f%%', pctdistance=0.9, shadow=False, rotatelabels=True)
        plt.show()

    def bar(self, top=10):
        part = self._get_score_part(top)
        labels = part.index
        fracs = part.values * 100
        y_pos = np.arange(len(fracs))
        plt.figure(figsize=(5, part.shape[0]/5))
        plt.subplot()
        plt.barh(y_pos, fracs, alpha=.8)
        plt.yticks(y_pos, labels)
        plt.title('Feature importance percentage.')
        plt.show()

    def __str__(self):
        return '%s: \n%s' % (self.coloredname, RSTable(self.scores).__str__())


class FSChi2(FeatureSelector):
    def __init__(self, features2process, feature_count=0.2):
        FeatureSelector.__init__(self, features2process, feature_count, name='χ²特征选择')

    def score(self, data, target):
        skb = SelectKBest(chi2, k='all')
        skb.fit_transform(data, target)
        return skb.scores_


class FSRFC(FeatureSelector):
    def __init__(self, features2process, feature_count=0.2):
        FeatureSelector.__init__(self, features2process, feature_count, name='rfc特征选择')

    def score(self, data, target):
        clf = RandomForestClassifier()
        clf.fit(data, target)
        return clf.feature_importances_


class FSmRMR(FeatureSelector):
    def __init__(self, features2process, feature_count=0.2):
        FeatureSelector.__init__(self, features2process, feature_count, name='mRMR特征选择')

    def score(self, data, target):
        scores = mrmr(data.values, target)  # scores[0] is the most important feature
        scores = scores.max() - scores
        return scores


class FSManual(FeatureSelector):
    def __init__(self, features2process, b_except=False):
        """
        select feature manually
        :param features2process:
        :param b_except: if True, select features who are not in features2process
        :param name:
        """
        FeatureSelector.__init__(self, features2process)
        self.b_except = b_except

    def _process(self, data, features, label):
        if self.b_except:
            features = list(set(data.columns[:-1]) - set(features))
        self.msg('%d features preserved.' % features.__len__())
        ret = pd.concat([data[features], data[label]], axis=1)
        return ret


class FSL1Regularization(FeatureSelector):
    def __init__(self, features2process, C):
        """
        去掉L1正则化后w为0的特征
        :param features2process:
        :param C:
        """
        FeatureSelector.__init__(self, features2process, feature_count=0.001, name='L1正则化特征选择')
        self.C = C

    def _score(self, data, target):
        clf = LogisticRegression(C=self.C)
        clf.fit(data, target)
        return clf.coef_.sum(axis=0)

