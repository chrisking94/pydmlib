from base import *
from sklearn.preprocessing import FunctionTransformer


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
        nanindexs = data.isnull()
        data = data.fillna(0)
        if self.breplace:
            data[features] = FunctionTransformer(self.transform).fit_transform(data[features])
        else:
            ts = FunctionTransformer(self.transform).fit_transform(data[features])
            modified = data[features].columns[((ts - data[features]).max() != 0)]
            if modified.shape[0] != 0:
                ts = pd.DataFrame(ts, columns=(modified + '_' + self.name))
                data = pd.concat([data, ts], axis=1)
        data[nanindexs] = np.nan
        self.msgtimecost()

        return data
