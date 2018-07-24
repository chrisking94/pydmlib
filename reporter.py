from  dataprocessor import RSDataProcessor
from  base import np


class Reporter(RSDataProcessor):
    def __init__(self, features2process, name='Reporter', forecolor='blue'):
        RSDataProcessor.__init__(self, features2process, name, forecolor, 'black', 'default')


class DataReporter(Reporter):
    def __init__(self, features2process, name='DataReporter'):
        Reporter.__init__(self, features2process, name, 'blue')


class DRBrief(DataReporter):
    def __init__(self, features2process, *args):
        """
        brief data reporter
        :param features2process:
        :param args: what to report,several options as following
                    1.shape
                    2.nan: NaN count of each column
                    3.unique-items: columns and unique items of each one
                    *.if no args is provided,report [1 ,2]
        """
        DataReporter.__init__(self, features2process, 'BriefDataReporter')
        self.args = args
        self.data_shape = (0, 0)

    def _process(self, data, features, label):
        X, y = data[features], data[label]
        breportall = self.args.__len__() == 0
        self.data_shape = data.shape
        if breportall or 'shape' in self.args:
            self.msg(data.shape.__str__(), 'data.shape')
        if breportall or 'columns' in self.args:
            b_contais_nan = False
            for x in features:
                null_count = data[x].isnull().sum()
                if null_count > 0:
                    self.msg('%s -> %d' % (x, null_count), 'NaN count')
                    b_contais_nan = True
            if not b_contais_nan:
                self.msg('there isn\'t any NaN in this data set.', 'NaN count')
        if 'unique-items' in self.args:
            self.msg('↓', 'unique-items')
            for col in features:
                items, cnts = np.unique(data[col], return_counts=True)
                if items.shape[0] > 20:
                    self.msg('%s -> %d type of items.' % (col, items.shape[0]))
                else:
                    self.msg('%s -> %s' % (col, dict(zip(items,cnts)).__str__()))
        return data


