from base import *
from misc import ConfusionMatrix, ROCCurve


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
            self.msg('data.shape=%s' % data.shape.__str__())
        if breportall or 'columns' in self.args:
            b_contais_nan = False
            for x in features:
                null_count = data[x].isnull().sum()
                if null_count > 0:
                    self._submsg('NaN count', -1, '%s -> %d' % (x, null_count))
                    b_contais_nan = True
            if not b_contais_nan:
                self._submsg('NaN count', -1, 'there isn\'t any NaN in this data set.')
        if 'unique-items' in self.args:
            self._submsg('unique-items', -1, '↓')
            for col in features:
                items, cnts = np.unique(data[col], return_counts=True)
                if items.shape[0] > 20:
                    self.msg('%s -> %d type of items.' % (col, items.shape[0]))
                else:
                    self.msg('%s -> %s' % (col, dict(zip(items,cnts)).__str__()))
        return data

    def get_report_title(self, *args):
        return ['data.shape']

    def get_report(self):
        return [self.data_shape.__str__()]


class ResultReporter(Reporter):
    def __init__(self, name='ResultReporter'):
        Reporter.__init__(self, name, 'white')


class ClfResult(object):
    def __init__(self, classes, trainscore, testscore, y_prob, y_pred, y_true, clfname):
        self.labels = classes
        self.trainscore = trainscore
        self.testscore = testscore
        self.y_prob = y_prob
        self.y_pred = y_pred
        self.y_true = y_true
        self.clfname = clfname


class RRConfusionMatrix(ResultReporter):
    def __init__(self, name='RR-ConfusionMatrix'):
        ResultReporter.__init__(self, name)
        self.cm = None

    def fit_transform(self, data):
        self.cm = ConfusionMatrix(data.y_true, data.y_pred, data.labels)
        self.cm.show(True)
        self.cm.draw()
        return data

    def get_report_title(self, *args):
        titles = ['%s正确率' % str(int(x)) for x in self.cm.columns]
        return titles

    def get_report(self):
        scores = [self.cm.normalized()[i, i] for i in range(self.cm.shape[0])]
        return scores


class RRRocCurve(ResultReporter):
    def __init__(self, name='RR-RocCurve'):
        ResultReporter.__init__(self, name)
        self.roc = None

    def fit_transform(self, data):
        self.roc = ROCCurve(data.y_true, data.y_prob[:, -1], title='ROC of %s' % data.clfname)
        self.roc.plot()
        return data

    def get_report_title(self, *args):
        return ['roc-auc']

    def get_report(self):
        return [self.roc.auc()]



