from base import *


class Transformer(RSDataProcessor):
    def __init__(self, features2process, name='Transformer'):
        """
        对data[features2process]做transform函数变换
        :param features2process:
        :param transform:
        :param breplace:是否用转换后的数据替换原数据，为False则把转换后数据追加到data中
        :param name:
        """
        super(Transformer, self).__init__(features2process, name, 'black', 'green')


class TsfmFunction(Transformer):
    def __init__(self, features2process, transform, breplace=True, name='函数转换器'):
        """
        对data[features2process]做transform函数变换
        :param features2process:
        :param transform:
        :param breplace: 是否用转换后的数据替换原数据，为False则把转换后数据追加到data中
        :param name:
        """
        super(TsfmFunction, self).__init__(features2process, name)
        self.transform = transform
        self.breplace = breplace

    def fit_transform(self, data):
        self.starttimer()
        features, label = self._getFeaturesNLabel(data)
        if self.breplace:
            data[features] = self.transform(data[features])
        else:
            ts = self.transform(data[features])
            modified = data[features].columns[((ts - data[features]).max() != 0)]
            if modified.shape[0] != 0:
                ts = pd.DataFrame(ts, columns=(modified + '_' + self.name))
                data = pd.concat([data[data.columns[:-1]], ts, data[label]], axis=1)
        self.msgtimecost()
        return data
