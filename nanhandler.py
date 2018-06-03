from base import *
from discretevalueencoder import DVEOneHot


class NanHandler(RSDataProcessor):
    def __init__(self, featues2process, name='NanHandler'):
        RSDataProcessor.__init__(self, featues2process, name, 'yellow', 'black')


class NHToSpecial(NanHandler):
    def __init__(self, featues2process, value=0):
        """
        :param value: 替代nan的值
        """
        NanHandler.__init__(self, featues2process, '设置NaN为%d' % (value))
        self.special = value

    def _process(self, data, features, label):
        data[features] = data[features].fillna(self.special)
        self.msg('data.shape.__str__()', 'data.shape')
        return data


class NHDropColumns(NanHandler):
    def __init__(self, features2process, nullrate_threshold=0.5):
        """
        以feature2process中的列为研究对象，如果某列的缺失率超过nullrate_threshhold，则丢弃该列
        :param features2process:
        :param nullrate_threshold:空值率阈值
        """
        NanHandler.__init__(self, features2process, '丢弃缺失率超过%.3f的列' % nullrate_threshold)
        self.nullrate_threshold = nullrate_threshold

    def _process(self, data, features, label):
        X = data[features]
        nullrates = X.isnull().sum() / X.shape[0]
        dropcols = X.columns[nullrates>self.nullrate_threshold]
        data = data.drop(columns=dropcols)
        self.msg(dropcols.__str__(), 'columns discarded')
        return data


class NHDropRows(NanHandler):
    def __init__(self, features2process, miss_rate_threshold=0.2, feature_weights=None):
        """
        对于某一行数据，根据feature_weights中的权重来计算其缺失率（见:param feature_weights）
        丢弃缺失率 > miss_rate_threshold的行
        :param miss_rate_threshold: 信息缺失率
        :param feature_weights: {w1,w2,...}
            features = {f1,f2,...}, 样本xj,j=0,1,2,...
            如果feature_weights is dict(f1:w1,f2:w2,...,fi:wi,...)，则按miss_rate_xj=∑(wi*(if <fi> is null then 1 else 0))
            如果feature_weights==None，则miss_rate_xj=count(xj.null)/len(xj)
        """
        NanHandler.__init__(self, features2process, '丢弃信息缺失率>=%.3f的行' % miss_rate_threshold)
        self.miss_rate_threshold = miss_rate_threshold
        if feature_weights is not None:
            feature_weights = np.array(feature_weights)
            feature_weights /= feature_weights.sum()
        self.feature_weights = feature_weights

    def _process(self, data, features, label):
        """
        丢弃features2process子集缺失率超过threshold的行
        """
        sample_count = data.shape[0]
        if self.feature_weights is None:
            miss_rate = data[features].isnull().sum(axis=1) / float(features.shape[0])
        else:
            miss_rate = (data[features].isnull() * self.feature_weights).sum(axis=1)
        keep = miss_rate <= self.miss_rate_threshold
        data = data.loc[keep, :]
        self.msg('%d ==> %d' % (sample_count, data.shape[0]), 'sample count')
        return data


class NHOneHot(NanHandler, DVEOneHot):
    def __init__(self, features2process):
        """
        对features2process中含有缺失值的列进行OneHot编码
        :param features2process:
        """
        DVEOneHot.__init__(self, features2process)
        NanHandler.__init__(self, features2process, '含NaN的列OneHot编码')

    def _process(self, data, features, label):
        nan_columns = features[data[features].isnull().sum() > 0]
        return  DVEOneHot._process(self, data, nan_columns, label)


