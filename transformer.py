from base import *
from sklearn.preprocessing import FunctionTransformer


class Transformer(RSDataProcessor):
    def __init__(self, transform, name='Transformer'):
        '''
        :param transform: lambda x:...
        :param features:要转换的特征列表
        '''
        super(Transformer, self).__init__(name, 'black', 'cyan')
        self.transform = transform

    def fit_transform(self, data, features, breplace=True):
        '''
        数据变换
        :param data:数据总表
        :param features: [featurename1,featurename2,...]
        :param breplace: 是否用转换后特征替换原本特征
                            如果False: 把转换后的追加到数据集
        :return: transfered data
        '''
        self.starttimer()
        nanindexs = data.isnull()
        data = data.fillna(0)
        if (breplace):
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
