from dataprocessor import *


class FactorEncoder(RSDataProcessor):
    def __init__(self, features2process, name='DiscreteValueEncoder'):
        RSDataProcessor.__init__(self, features2process, name, 'white', 'black')

    def _encode(self, data, features, label):
        self.error('Not implemented!')

    def _process(self, data, features, label):
        dshape0 = data.shape[1] - 1
        data = self._encode(data, features, label)
        self.msg('%d ==> %d' %(dshape0, data.shape[1] - 1), 'feature count')
        return data


class FEOneHot(FactorEncoder):
    def __init__(self, features2process):
        FactorEncoder.__init__(self, features2process, 'OneHot编码')

    def _encode(self, data, features, label):
        """
        OneHot encode
        :param data:
        :param columns:
        """
        data = data.copy()
        data_class = data.__class__
        f = []
        not_encoded = []
        drop_cols = []
        for x in features:  # 不编码二值列，包含np.nan和数据类型为object的列除外
            unique_items = data[x].unique()
            if unique_items.shape[0] > 2:
                f.append(x)
            elif unique_items.shape[0] == 1:
                drop_cols.append(x)
            elif pd.Series(unique_items).dtype == 'object' or pd.Series(unique_items).isnull().sum() > 0:  # 不能直接elif np.nan in unique_items，这样判断不出nan
                f.append(x)
            else:
                not_encoded.append(x)
        if not_encoded.__len__() > 0:
            self.warning('%s contains less than 3 kind of values, so they won\'t be encoded.' % not_encoded.__str__())
        if drop_cols.__len__() > 0:
            self.warning('there is only 1 kind of value in %s,they will be discarded.' % drop_cols.__str__())

        features = f
        drop_cols.extend(features)
        X = data[features].astype('str', copy=False)
        encdisc = pd.get_dummies(X, dummy_na=False)
        encdisc = encdisc.astype('float', copy=False)
        data.drop(columns=drop_cols, inplace=True)
        data = pd.concat([encdisc, data], axis=1)
        data = data_class(data=data)
        return data
