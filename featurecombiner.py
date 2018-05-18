from base import *
import re


class FeatureCombiner(RSDataProcessor):
    def __init__(self, features2process, name='FeatureCombiner'):
        """
        特征组合
        :param features2process: 
        :param name: 
        """
        super(FeatureCombiner, self).__init__(features2process, name, 'cyan', 'blue', 'highlight')


class FCbArithmetical(FeatureCombiner):
    def __init__(self, features2process, operations):
        """
        对特征进行算术组合
        :param features2process: None，其他值不起作用
        :param operations:运算列表[operation1, operation2, ...]
                        operation格式示例如下：
                            1、'[f1] = [f1] + [f2]' 将f1替换为f1+f2；
                            2、'[f_new] = [f1] + [f2]' 把f1+f2结果作为新的一列f_new，保持f1、f2不变；
                            3、'[f_new] = [f1] + [f2] @replace'与2类似，不同在于3会把f1、f2从表中删除。
        """
        super(FCbArithmetical, self).__init__(features2process, '算术特征组合')
        self.operations = operations
        self.parsedOperations = []
        for operation in self.operations:
            self.parsedOperations.append(self._parse(operation))

    def _parse(self, operation):
        fregex = re.compile(r'\[([^\]]+)\]')
        features = fregex.findall(operation)
        operation = operation.replace('[', 'data[\'')
        operation = operation.replace(']', '\']')
        if features.__len__() < 2 :
            self.error('operation <%s> invalid!' % operation)
        if operation.find('@replace') != -1:
            stmp = ';data = data.drop(columns=%s)' % features[1:].__str__()
            operation = operation.replace('@replace', stmp)
        return operation

    def fit_transform(self, data):
        self.starttimer()
        features, label = self._getFeaturesNLabel(data)
        data = data.copy()
        data, target = data[features], data[label]
        for i, cmd in enumerate(self.parsedOperations):
            exec(cmd)
            self._submsg(self.operations[i], 'cyan', 'done.')
        data = pd.concat([data, target], axis=1)
        self.msgtimecost()
        return data


