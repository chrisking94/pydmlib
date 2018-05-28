from base import *


class FeatureClassifier(RSDataProcessor):
    def __init__(self, features2process, name='FeatureClassifier'):
        super(FeatureClassifier, self).__init__(features2process, name, 'blue', 'white', 'bold', False)


class FCUniqueItemCountGe(FeatureClassifier):
    def __init__(self, features2process, cont_feat_threshold, contfeats, discfeats):
        """
        通过特征取值类型的个数来区分连续、离散特征
        :param features2process:
        :param cont_feat_threshold:连续特征值类型数目的阈值，大于阈值的被划分为连续特征
        :param contfeats: list, 用于存储分类结果中连续特征
                            可以预设连续特征，在fit_transform时此列表中预设的特征不会被划分到discfeats
                            * 只有在创建对象时输入该参数，之后对列表的更改不会影响预设值
        :param discfeats: list, 用于存储分类结果中离散特征
                            可以预设离散特征，在fit_transform时此列表中预设的特征不会被划分到contfeats
        """
        super(FCUniqueItemCountGe, self).__init__(features2process, '值类型数>=%d为连续特征' % ( cont_feat_threshold))
        self.threshold = cont_feat_threshold
        self.presetContFeats = contfeats.copy()
        self.presetDiscFeats = discfeats.copy()
        self.contfeats = contfeats
        self.discfeats = discfeats

    def _process(self, data, features, label):
        self.contfeats.clear()
        self.discfeats.clear()
        cntdict = {}
        for col in features:
            cnt = data[col].unique().shape[0]
            cntdict[col] = cnt
            if cnt >= self.threshold:
                if col not in self.presetDiscFeats:
                    self.contfeats.append(col)
                else:
                    self.discfeats.append(col)
            else:
                if col not in self.presetContFeats:
                    self.discfeats.append(col)
                else:
                    self.contfeats.append(col)
        self.msg('%d continuous features, %d discrete features.' % (self.contfeats.__len__(), self.discfeats.__len__()))
        return data

