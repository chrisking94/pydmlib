from dataprocessor import *


class FeatureClassifier(RSDataProcessor):
    def __init__(self, features2process, contfeats, discfeats, labelfeats, name=''):
        """
        把X中的特征分为：continuous, discrete, label
        :param features2process:
        :param contfeats: 预设continuous features，且用于存放分到此类的特征名
        :param discfeats: 预设discrete features, ...
        :param labelfeats: 预设label features, ...
        :param name:
        """
        RSDataProcessor.__init__(self, features2process, name, 'blue', 'white', 'bold')
        self.presetContFeats = set(contfeats.copy())
        self.presetDiscFeats = set(discfeats.copy())
        self.presetLabelFeats = set(labelfeats.copy())
        self.contfeats = contfeats
        self.discfeats = discfeats
        self.labelfeats = labelfeats

    def _classify(self, data, features, label):
        self.error('Not implemented!')

    def _process(self, data, features, label):
        self.contfeats.clear()
        self.discfeats.clear()
        self.labelfeats.clear()
        self._classify(data, features, label)
        self.msg('continuous=%d, discrete=%d, label=%d' %
                 (self.contfeats.__len__(), self.discfeats.__len__(), self.labelfeats.__len__())
                 , 'count')
        return data


class FCUniqueItemCountGe(FeatureClassifier):
    def __init__(self, features2process, cont_feat_threshold, *args):
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
        FeatureClassifier.__init__(self, features2process, *args, name='值类型数分类')
        self.threshold = cont_feat_threshold

    def _classify(self, data, features, label):
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
        return data


class FCLabel(FeatureClassifier):
    def __init__(self, features2process, *args):
        FeatureClassifier.__init__(self, features2process, *args, name='<?>特征分类器')

    def _classify(self, data, features, label):
        self.contfeats.extend(features['@c'])
        self.discfeats.extend(features['@d'])
        self.labelfeats.extend(features['@l'])

