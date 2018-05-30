from base import *


class DiscreteValueEncoder(RSDataProcessor):
    def __init__(self, features2process, name='DiscreteValueEncoder'):
        RSDataProcessor.__init__(self, features2process, name, 'white', 'black')

    def _encode(self, data, features, label):
        self.error('Not implemented!')

    def _process(self, data, features, label):
        dshape0 = data.shape[0] - 1
        data = self._encode(data, features, label)
        self.msg('data.features\t%d ==> %d' %(dshape0, data.shape[0] - 1))


class DVEOneHot(DiscreteValueEncoder):
    def __init__(self, features2process):
        DiscreteValueEncoder.__init__(self, features2process, 'OneHot编码')

    def _encode(self, data, features, label):
        """
        OneHot encode
        :param data:
        :param columns:
        """
        data = data.copy()
        X = data[features].astype('str', copy=False)
        encdisc = pd.get_dummies(X, dummy_na=False)
        encdisc = encdisc.astype('float', copy=False)
        data.drop(columns=features, inplace=True)
        data = pd.concat([encdisc, data], axis=1)
        return data
