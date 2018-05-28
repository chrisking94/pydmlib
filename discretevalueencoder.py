from base import *


class DiscreteValueEncoder(RSDataProcessor):
    def __init__(self, features2process, name='DiscreteValueEncoder'):
        super(DiscreteValueEncoder, self).__init__(features2process, name, 'white', 'black')


class DVEOneHot(DiscreteValueEncoder):
    def __init__(self, features2process):
        super(DVEOneHot, self).__init__(features2process, 'OneHot编码')

    def _process(self, data, features, label):
        """
        OneHot encode
        :param data:
        :param columns:
        """
        self.msg('shape before encoding %s' % (data.shape.__str__()))
        data = data.copy()
        X = data[features].astype('str', copy=False)
        encdisc = pd.get_dummies(X, dummy_na=False)
        encdisc = encdisc.astype('float', copy=False)
        data.drop(columns=features, inplace=True)
        data = pd.concat([encdisc, data], axis=1)
        self.msg('shape after encoding %s' % (data.shape.__str__()))
        return data
