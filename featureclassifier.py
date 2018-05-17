from base import *


class FeatureClassifier(RSDataProcessor):
    def __init__(self, features2process, name='FeatureClassifier'):
        super(FeatureClassifier, self).__init__(name, 'blue', 'white')


class FCUniqueItemCountGe(FeatureClassifier):
    def __init__(self, features2process, cont_feat_threshold, contfeats=[], discfeats=[]):
        '''
        通过特征取值类型的个数来区分连续、离散特征
        :param features2process:
        :param cont_feat_threshold:连续特征值类型数目的阈值，大于阈值的被划分为连续特征
        :param contfeats:预设连续特征，在fit_transform时此列表中预设的特征不会被划分到discfeats
        :param discfeats:预设离散特征，在fit_transform时此列表中预设的特征不会被划分到contfeats
        '''
        super(FCUniqueItemCountGe, self).__init__(features2process, '值类型数>=%d为连续特征' % ( cont_feat_threshold))
        self.threshold = cont_feat_threshold
        self.contfeats = contfeats
        self.discfeats = discfeats

    def fit_transform(self, data):
        '''
        把特征分成连续特征，离散特征
        :param data:[X y]
        :param contfeats:预设连续特征，分类器将会保持该列表中的特征为连续特征
        :return:contfeats, discfeats
        '''
        discfeats = self.discfeats.copy()
        contfeats = self.contfeats.copy()
        features, label = self._getFeaturesNLabel(data)
        cntdict = {}
        for col in features:
            cnt = data[col].unique().shape[0]
            cntdict[col] = cnt
            if (cnt >= self.threshold):
                if col not in discfeats:
                    contfeats.append(col)
            else:
                if col not in contfeats:
                    discfeats.append(col)
        self.msg('%d个连续特征 %s' % (contfeats.__len__(), contfeats.__str__()))
        self.msg('%d个离散特征 %s' % (discfeats.__len__(), discfeats.__str__()))
        self.msg('值类型数量 %s' %(cntdict.__str__()))

        return contfeats, discfeats

