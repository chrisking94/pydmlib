#coding=utf-8
from base import *


class Transformer(RSDataProcessor):
    def __init__(self, features2process, name='Transformer'):
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

    def _process(self, data, features, label):
        if self.breplace:
            data[features] = self.transform(data[features])
        else:
            ts = self.transform(data[features])
            modified = data[features].columns[(ts!=data[features]).sum()>0]
            ts = ts[modified]
            if modified.shape[0] != 0:
                ts = ts.rename(dict(zip(modified, modified + '_' + self.name)), axis=1)
                data = pd.concat([data[data.columns[:-1]], ts, data[label]], axis=1)
        return data

