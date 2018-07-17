from dataprocessor import *


class FactorEncoder(RSDataProcessor):
    def __init__(self, features2process, name=''):
        RSDataProcessor.__init__(self, features2process, name, 'white', 'black')

    def _encode(self, data, features, label):
        self.error('Not implemented!')

    def _process(self, data, features, label):
        dshape0 = data.shape[1] - 1
        data = self._encode(data, features, label)
        self.msg('%d ==> %d' % (dshape0, data.shape[1] - 1), 'feature count')
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
            elif pd.Series(unique_items).dtype == 'object' or pd.Series(unique_items).isnull().sum() > 0:
                # 不能直接elif np.nan in unique_items，这样判断不出nan
                f.append(x)
            else:
                # 二值列，转换值为0，1
                not_encoded.append(x)
                if 0 in unique_items:
                    if 1 in unique_items:
                        continue
                    else:
                        data[data[x] != 0, x] = 1
                elif 1 in unique_items:
                    data[data[x] != 1, x] = 0
                else:
                    data[data[x] == unique_items[0], x] = 0
                    data[data[x] == unique_items[1], x] = 1
        if drop_cols.__len__() > 0:
            self.warning('there is only 1 kind of value in %s,they will be discarded.' % drop_cols.__str__())
            data.drop(columns=drop_cols, inplace=True)
        if f.__len__() == 0:
            self.msg('no feature encoded.')
            return data
        features = f
        X = data[features].astype('str', copy=False)
        encdisc = pd.get_dummies(X, dummy_na=False)
        # encdisc = encdisc.astype('float', copy=False)
        data.drop(columns=features, inplace=True)
        data = pd.concat([encdisc, data], axis=1)
        data = data_class(data=data)
        return data
