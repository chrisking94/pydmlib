from base import *


class DiscreteValueEncoder(RSDataProcessor):
    def __init__(self, features2process, name='DiscreteValueEncoder'):
        super(DiscreteValueEncoder, self).__init__(features2process, name, 'white', 'black')


class DVEOneHot(DiscreteValueEncoder):
    def __init__(self):
        super(DVEOneHot, self).__init__('OneHot编码')

    def fit_transform(self, data):
        '''
        OneHot encode
        :param data:
        :param columns:
        '''
        self.starttimer()
        columns, label = self._getFeaturesNLabel(data)
        data = data.copy()
        data[columns] = data[columns].astype('str')
        target = data[data.columns[-1]]
        encdisc = pd.get_dummies(data[columns], dummy_na=False)
        encdisc = encdisc.astype('float')
        data.drop(columns=columns, inplace=True)
        data.drop(columns=[data.columns[-1]], inplace=True)
        data = pd.concat([data, encdisc, target], axis=1)
        self.msg('shape after encoding: %s' % (data.shape.__str__()))
        self.msgtimecost()
        return data
