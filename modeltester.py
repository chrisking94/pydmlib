from base import *
from misc import ConfusionMatrix
import sklearn.cross_validation as cv


class ModelTester(RSObject):
    def __init__(self, name='ModelTester'):
        super(ModelTester, self).__init__('ModelTester', 'red', 'default', 'highlight')
        self.data = None
        self.targetlabels = None  # target的值列表，用于混淆矩阵的label参数

    def classify(self, classifier, test_size=0.2):
        '''
        使用分类器对分类后的数据进行预测
        :param classifier:
        :param test_size:测试集大小
        :return:
        '''
        self.starttimer()
        self.msg('\033[1;31;47m%s\033[0m' % ('↓' * 40))
        data = self.data
        if isinstance(data, tuple):
            trainset, testset = data[0], data[1]
        else:
            trainset, testset = cv.train_test_split(data, test_size=test_size, random_state=0)

        x_train, y_train = trainset[trainset.columns[:-1]], trainset[trainset.columns[-1]]
        x_test, y_test = testset[testset.columns[:-1]], testset[testset.columns[-1]]

        self._submsg('使用特征',1 , x_train.columns.__str__())

        classifier.fit(x_train, y_train)
        trainscore = classifier.score(x_train, y_train)
        self._submsg('训练集得分', 1, '%f' % (trainscore*100))
        y_pred = classifier.predict(x_test)
        testscore = classifier.score(x_test, y_test)
        self._submsg('测试集得分', 1, '%f' % (testscore* 100))

        cm = ConfusionMatrix(y_test, y_pred, self.targetlabels)
        cm.show()
        cm.show(True)
        cm.draw()

        self.msgtimecost()
        self.msg('↑' * 40)

        return trainscore, testscore, cm

    def _gettargetlabels(self, data):
        self.targetlabels = data[data.columns[-1]].unique()


class MTSingle(ModelTester):
    def __init__(self, data_procr_sequence, name='SingleModelTester'):
        '''
        固定单模型测试
        :param data_procr_sequence:list(数据处理器)
        :param name:
        '''
        super(MTSingle, self).__init__(name)
        self.dataProcrSequence = data_procr_sequence

    def fit_transform(self, data):
        '''
       开始测试
       :param data: 测试数据，可以是：
           1、整个数据集
           2、tuple(trainset，testset)
       '''
        self.starttimer()
        self._gettargetlabels(data)
        for processor in self.dataProcrSequence:
            data = processor.fit_transform(data)
        self.data = data
        self.msgtimecost(msg='数据处理总耗时')


class MTAutoGrid(ModelTester):
    def __init__(self, data_procr_grid, test_size=0.2):
        '''
        自动化网格测试
        :param data_procr_grid: 数据处理器表，表末尾必须是分类器，结构示例如下：
                            [[procr11, proc12],
                             [proc21, proc22, proc23],
                             proc3,
                             ...
                             [clf1, clf2, ...]]
        '''
        super(MTAutoGrid, self).__init__('AutoGridModelTester')
        self.data_procr_grid = data_procr_grid
        self.test_size = 0.2
        self.reporttablehead = []
        for procrs in self.data_procr_grid[:-1]:
            self.reporttablehead.append(procrs.__base__.__name__)
        self.reporttablehead.extend(['分类器', '训练集得分', '测试集得分'])
        self.reporttable = None

    def _run(self, data_procr_grid, infolist):
        current_procrs = data_procr_grid[0]
        if not isinstance(current_procrs, list):
            current_procrs = [current_procrs]
        for procr in current_procrs:
            gc.collect()
            if data_procr_grid.__len__() == 1:
                # 网格最后为分类器
                trainscore, testscore, cm = self.classify(procr, self.test_size)
                infolist.extend([procr.name, trainscore, testscore])
                infolist.extend(cm.getclassscores())
                self.reporttable.loc[self.reporttable.shape[0]] = infolist  # 记录测试信息
                infolist = []
            else:
                try:
                    self.data = procr.fit_transform(self.data)
                    infolist.append(procr.name)
                except:
                    infolist.append('%s_failed' % procr.name)
                self._run(data_procr_grid[1:], infolist)

    def fit_transform(self, data):
        '''
        开始测试
        :param data:输入数据集
        :return:
        '''
        self.starttimer()
        self._gettargetlabels(data)
        # 制作reporttable表头
        head = self.reporttablehead
        head.extend(self.targetlabels + '类正确率')
        self.reporttable = pd.DataFrame(columns=head)
        self._run(self.data_procr_grid, [])
        self.msgtime('测试完成！调用report()可获取测试日志。')
        self.report()

    def report(self):
        self.msg('%s\n' % self.reporttable.__str__())
        return self.reporttable

