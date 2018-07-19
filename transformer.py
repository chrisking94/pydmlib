# coding=utf-8
from dataprocessor import *


class Transformer(RSDataProcessor):
    def __init__(self, features2process, name=''):
        """
        对data[features2process]做变换
        :param features2process:
        :param name:
        """
        RSDataProcessor.__init__(self, features2process, name, 'black', 'green')


class TsfmFunction(Transformer):
    def __init__(self, features2process, transform, breplace=True, name='函数转换器'):
        """
        对data[features2process]做transform函数变换
        :param features2process:
        :param transform:
        :param breplace: 是否用转换后的数据替换原数据，为False则把转换后数据追加到data中
        :param name:
        """
        Transformer.__init__(self, features2process, name)
        self.transform = transform
        self.breplace = breplace
        self.cost_estimator = CETime.get_estimator(transform)

    def _process(self, data, features, label):
        ts = self.transform(data[features])
        modified = features[(ts != data[features]).sum() > 0]
        ts = ts[modified]
        if modified.shape[0] != 0:
            self.msg('expected: %d, actual: %d' % (features.shape[0], modified.shape[0]),
                     'columns modified')
            if self.breplace:
                data = data.drop(columns=modified)
            ts = ts.rename(dict(zip(modified, modified + '_' + self.name)), axis=1)
            data = pd.concat([ts, data], axis=1)
        else:
            self.warning('no column affected.')
        return data

