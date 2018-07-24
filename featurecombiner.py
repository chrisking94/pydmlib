from  dataprocessor import *
import re


class FeatureCombiner(RSDataProcessor):
    def __init__(self, features2process, name=''):
        """
        特征组合
        :param features2process: 
        :param name: 
        """
        RSDataProcessor.__init__(self, features2process, name, 'cyan', 'blue', 'highlight')


class FCbArithmetical(FeatureCombiner):
    def __init__(self, features2process, operations, name='算术特征组合'):
        """
        对特征进行算术组合
        :param features2process: None，其他值不起作用
        :param operations:运算列表[operation1, operation2, ...]
                        operation格式示例如下：
                            1、'[f1] = [f1] + [f2]' 将f1替换为f1+f2；
                            2、'[f_new] = [f1] + [f2]' 把f1+f2结果作为新的一列f_new，保持f1、f2不变；
                            3、'[f_new] = [f1] + [f2] @replace'与2类似，不同在于3会把f1、f2从表中删除。
                            4、'[f_$1] = [f1_$1] + [f2_$2]' $代表匹配任意字符串，并将匹配到的值存入变量$1
                                *.匹配到的$1可能等于$2
                                *.单变量时，可以使用简略的$而不必写成$0,$2等形式
        """
        FeatureCombiner.__init__(self, features2process, name)
        self.operations = operations
        self.staticOperations = []
        self.dynamicOperations = []
        for operation in self.operations:
            s = operation.replace('\$', '')
            if s.find('$') == -1:  #static operation
                self.staticOperations.append(operation)
            else:
                self.dynamicOperations.append(operation)

    def _parse_static(self, operation):
        operation = operation.replace('\$', '$')
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

    def _is_valid(self, operation, features):
        """
        检查operation中的特征名是否存在于features中
        :param operation:
        :param features:
        :return: 存在则返回合法，即True
        """
        fregex = re.compile(r'\[([^\]]+)\]')
        feat_in_opr = fregex.findall(operation)
        for feat in feat_in_opr[1:]:
            if feat not in features:
                self.warning('%s bad operation, [%s] not found!' % (self._opr_to_readable(operation), feat))
                return False
        return True

    def _parse_dynamic(self, operation, features):
        """
        动态表达式，表达式必须在得知features后才能确定
        :param operation:
        :param data:
        :return: 解析出的静态operation列表
        """
        fregex = re.compile(r'\[([^\]]+)\]')
        dynamicFeats = fregex.findall(operation)
        rgx = re.compile(r'\$[0-9]{0,1}')
        fs = '\n'.join(features)
        vardict = {}
        for dynamicOpr in dynamicFeats[1:]:
            varlist = rgx.findall(dynamicOpr)
            featrgx = rgx.subn(r'(.*)', dynamicOpr)[0]
            featrgx = re.compile(featrgx)
            valuelist = featrgx.findall(fs)
            for i, var in enumerate(varlist):
                if var in vardict.keys():
                    if varlist.__len__() == 1:
                        var_value_list = [x for x in valuelist if x in vardict[var]]
                    else:
                        var_value_list = [x[i] for x in valuelist if x in vardict[var]]
                else:
                    if varlist.__len__() == 1:
                        var_value_list = [x for x in valuelist]
                    else:
                        var_value_list = [x[i] for x in valuelist]
                vardict[var] = var_value_list
        list_operation = []
        self._rgen_operation(operation, tuple(vardict.items()), list_operation)
        return list_operation

    def _rgen_operation(self, str_opr, list_var_values, list_ret):
        bEndPoint = list_var_values.__len__() == 1
        var = list_var_values[0][0]
        for value in list_var_values[0][1]:
            if bEndPoint:
                list_ret.append(str_opr.replace(var, value))
            else:
                self._rgen_operation(str_opr.replace(var, value), list_var_values[1:], list_ret)

    def _opr_to_readable(self, operation):
        return  operation .replace('[', '').replace(']', '')

    def _process(self, data, features, label):
        data = data.copy()
        feat_count0 = data.shape[1] - 1
        data, target = data[data.columns[:-1]], data[label]
        # 静态表达式
        operations = [x for x in self.staticOperations if self._is_valid(x, features)]
        # 动态表达式
        for dyopr in self.dynamicOperations:
            operations.extend(self._parse_dynamic(dyopr, features))
        for i, cmd in enumerate(operations):
            exec(self._parse_static(cmd))
            self.msg(self._opr_to_readable(cmd), 'done')
        data = pd.concat([data, target], axis=1)
        nadded = data.shape[1]-feat_count0-1
        nmodified = operations.__len__() - nadded
        self.msg('feature count\t%d ==> %d, %d added, %d replaced.' %
                 (feat_count0, data.shape[1]-1, nadded, nmodified))
        return data


