from rsobject import *


class FeatureClassifier(RSObject):
    def __init__(self, name='FeatureClassifier'):
        super(FeatureClassifier, self).__init__(name, 'blue', 'white')


class FCUniqueItemCountGe(FeatureClassifier):
    def __init__(self, cont_feat_threshold):
        super(FCUniqueItemCountGe, self).__init__('值类型数>=%d为连续特征' % (cont_feat_threshold))
        self.threshold = cont_feat_threshold

    def fit_transform(self, data, contfeats=[], discfeats=[]):
        '''
        把特征分成连续特征，离散特征
        :param data:[X y]
        :param contfeats:预设连续特征，分类器将会保持该列表中的特征为连续特征
        :return:contfeats, discfeats
        '''
        cntdict = {}
        for col in data.columns[:-1]:
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

