from base import *
from misc import *
import sklearn.model_selection as cv


class ModelTester(RSObject):
    def __init__(self, name='ModelTester'):
        RSObject.__init__(self, name, 'red', 'default', 'highlight')
        self.data = None
        self.targetlabels = None  # target的值列表，用于混淆矩阵的label参数
        self.trainscore = 0
        self.testscore = 0
        self.cm = None

    def report(self):
        self.error('Not implemented!')

    def classify(self, classifier, test_size=0.2):
        """
        使用分类器对分类后的数据进行预测
        :param classifier:
        :param test_size:测试集大小
        :return:
        """
        self.starttimer()
        self.msgtime(msg='开始测试[%s]。' % classifier.__class__.__name__)
        data = self.data
        if isinstance(data, tuple):
            trainset, testset = data[0], data[1]
        else:
            trainset, testset = cv.train_test_split(data, test_size=test_size, random_state=0)

        x_train, y_train = trainset[trainset.columns[:-1]], trainset[trainset.columns[-1]]
        x_test, y_test = testset[testset.columns[:-1]], testset[testset.columns[-1]]
        classifier.fit(x_train, y_train)
        self.trainscore = classifier.score(x_train, y_train)
        if hasattr(classifier, 'predict_proba'):
            y_prob = classifier.predict_proba(x_test)
            y_pred = classifier.classes_[y_prob.argmax(axis=1)]
            self.roc = ROCCurve(y_test, y_prob[:,1], 'ROC of %s' % classifier.__class__.__name__)
        else:
            y_pred = classifier.predict(x_test)
            self.roc = None
            self.msg('classifier [%s] does not have method predict_proba(),roc is not available!')
        self.testscore = (y_pred == y_test).sum() / y_test.shape[0]
        self.cm = ConfusionMatrix(y_test, y_pred, self.targetlabels)
        self.msgtimecost(msg='完成.')
        return self.trainscore, self.testscore, self.cm

    def _gettargetlabels(self, data):
        self.targetlabels = data[data.columns[-1]].unique()


class MTSingle(ModelTester):
    def __init__(self, data_procr_sequence, name='SingleModelTester'):
        """
        固定单模型测试
        :param data_procr_sequence:list(数据处理器)
        :param name:
        """
        ModelTester.__init__(self, name)
        self.dataProcrSequence = data_procr_sequence

    def fit_transform(self, data):
        """
       开始测试
       :param data: 测试数据，可以是：
           1、整个数据集
           2、tuple(trainset，testset)
       """
        self.starttimer()
        self._gettargetlabels(data)
        for processor in self.dataProcrSequence:
            data = processor.fit_transform(data)
            gc.collect()
        self.data = data
        self.msgtimecost(msg='数据处理总耗时')
        return data

    def report(self):
        """
        输出分类结果相关信息
        """
        self.msg('\033[1;31;47m%s\033[0m' % ('↓' * 40))
        self._submsg('训练集得分', 1, '%f' % (self.trainscore*100))
        self._submsg('测试集得分', 1, '%f' % (self.testscore* 100))
        self.cm.show()
        self.cm.show(True)
        self.cm.plot()
        if self.roc is not None:
            self.roc.plot()
        self.msg('↑' * 40)


class MTAutoGrid(ModelTester):
    def __init__(self, data_procr_grid):
        """
        自动化网格测试
        :param data_procr_grid: 数据处理器表，表末尾必须是分类器，结构示例如下：
                            [[procr11, procr12],
                             [procr21, procr22, procr23],
                             procr3,
                             ...
                             [clf1, clf2, ...]]
        """
        ModelTester.__init__(self, 'AutoGridModelTester')
        self.data_procr_grid = data_procr_grid
        self.report_table = None  # pd.DataFrame()
        self.process_table = []
        self._gen_process_table(data_procr_grid, [])  # 存入self.process_table

    def _gen_process_table(self, data_procr_grid, processor_sequence):
        """
        生成处理流程表，存放在process_table中
        :param data_procr_grid:  处理器网格
        :param processor_sequence: list,  用于存放一条处理序列的列表
        :return: 遇到TCBreak返回False，否则返回True
        """
        cur_procrs = data_procr_grid[0]
        if not isinstance(cur_procrs, list):
            cur_procrs = [cur_procrs]
        if cur_procrs.__len__() > 1:
            # 多处理器分叉，设置数据检查点
            processor_sequence.append(TCCheckPoint())
        for i, procr in enumerate(cur_procrs):
            if i==0:
                ps = processor_sequence.copy()
            else:
                ps = [TCContinue('TCContinue-%s' % x.name) for x in processor_sequence[:-1]]
                ps.append(processor_sequence[-1])  # 复制数据检查点
            ps.append(procr)
            if isinstance(procr, TesterController):
                if isinstance(procr, TCBreak):  # break
                    self.process_table.append(ps)
                    return False
                elif isinstance(procr, TCEnd):  # end
                    self.process_table.append(ps)
                    return True
            if data_procr_grid.__len__() > 1:
                if not self._gen_process_table(data_procr_grid[1:], ps):
                    return False
            else:
                self.process_table.append(ps)
        return True

    def fit_transform(self, data):
        """
        开始测试
        :param data:输入数据集
        :return:
        """
        self.starttimer()
        b_head_done = False
        self.report_table = None
        cdata = data
        for i, sequence in enumerate(self.process_table):
            cdata = data
            self.msgtime(msg='开始执行第%d/%d个测试序列。' % ((i+1), self.process_table.__len__()))
            b_contains_continue = False
            report_table_head = []
            report_line = []
            for procr in sequence:
                cdata = procr.fit_transform(cdata)
                if isinstance(procr, TCContinue):  # continue
                    b_contains_continue = True
                report_line.extend(procr.get_report())
                if not b_contains_continue and not b_head_done:
                    report_table_head.extend(procr.get_report_title())
            # 生成表头
            if not b_contains_continue:
                if not b_head_done:
                    b_head_done = True
                    self.report_table = pd.DataFrame(self.report_table, columns=report_table_head)
            # 添加记录
            if self.report_table is None:
                self.report_table = pd.DataFrame([report_line])
            else:
                self.report_table.loc[self.report_table.shape[0], :] = report_line
            # 输出报告
            current_row = self.report_table.loc[self.report_table.shape[0] - 1]
            self._submsg('report', 'green',
                         '\n%s' % current_row.__str__())
            self.msgtimecost(msg='第%d/%d个测试完成。' % ((i+1), self.process_table.__len__()))
        self.msgtimecost(msg='总耗时。')
        self.msgtime(msg='测试完成！调用log()可获取测试日志。')
        self.log()
        self.data = cdata
        return cdata

    def log(self):
        return self.report_table

    def savelog(self, filepath='', btimesuffix=False):
        """
        save log as .csv
        :param filepath: don't carry a .csv suffix, e.g.usr/xxx is fine.
        :param btimesuffix: append a time suffix after filename,e.g filename --> filename_2018-05-22 14:53:00
        :return:
        """
        if filepath == '':
            filepath = '%s_%s.csv' % (self.name, self.strtime())
        else:
            if btimesuffix:
                filepath = '%s_%s.csv' % (filepath, self.strtime())
        self.report_table.to_csv(filepath, sep='\t')


class TesterController(RSDataProcessor):
    def __init__(self, name='TesterController'):
        RSDataProcessor.__init__(self, None, name, 'white', 'black', 'default')


class TCBreak(TesterController):
    def __init__(self, name='TC-Break'):
        TesterController.__init__(self, name)

    def fit_transform(self, data):
        self.msgtime(self._colorstr('**break*point' * 8, 0, self.msgforecolor, self.msgbackcolor))
        return data


class TCContinue(TesterController):
    def __init__(self, name='TC-Continue'):
        TesterController.__init__(self, name)

    def fit_transform(self, data):
        self.msg('skipped.')
        return data


class TCEnd(TesterController):
    def __init__(self, name='TC-End'):
        TesterController.__init__(self, name)

    def fit_transform(self, data):
        self.msgtime(self._colorstr('--end-point' * 10, 0, self.msgforecolor, self.msgbackcolor))
        return data


class TCCheckPoint(TesterController):
    def __init__(self):
        TesterController.__init__(self, 'TC-CheckPoint')
        self.data = None

    def fit_transform(self, data):
        self.msgtime(self._colorstr('++check+point' * 8, 0, self.msgforecolor, self.msgbackcolor))
        if self.data is None:
            self.data = data
        return self.data

    def get_report(self):
        return ['检查点']

    def get_report_title(self, *args):
        return ['检查点']




