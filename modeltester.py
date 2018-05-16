from rsobject import *
from misc import ConfusionMatrix
import sklearn.cross_validation as cv


class ModelTester(RSObject):
    def __init__(self, classifier):
        '''
        测试器
        :param classifier:分类器
        '''
        self.clf = classifier
        super(ModelTester, self).__init__('ModelTester', 'red', 'default', 'highlight')

    def fit_transform(self, data, msg='', test_size=0.2):
        '''
       开始测试
       :param data: 测试数据，可以是：
           1、整个数据集
           2、tuple(trainset，testset)
       '''
        self.starttimer()
        clf = self.clf
        self.msg('\033[1;31;47m%s\033[0m' % ('↓' * 40))
        self.msgtime('测试开始')
        self._submsg('附加说明', 1, msg)
        self._submsg('分类器信息', 1, clf.__str__())

        if (isinstance(data, tuple)):
            trainset, testset = data[0], data[1]
        else:
            trainset, testset = cv.train_test_split(data, test_size=test_size, random_state=0)

        x_train, y_train = trainset[trainset.columns[:-1]], trainset[trainset.columns[-1]]
        x_test, y_test = testset[testset.columns[:-1]], testset[testset.columns[-1]]

        self._submsg('使用特征',1 , x_train.columns.__str__())

        clf.fit(x_train, y_train)
        trainscore = clf.score(x_train, y_train)
        self._submsg('训练集得分', 1, '%f' % (trainscore*100))
        y_pred = clf.predict(x_test)
        testscore = clf.score(x_test, y_test)
        self._submsg('测试集得分', 1, '%f' % (testscore* 100))

        unique_items, cnts = np.unique(y_test, return_counts=True)
        cm = ConfusionMatrix(y_test, y_pred)
        cm.show()
        cm.show(True)
        cm.draw()

        self.msgtimecost()
        self.msg('↑' * 40)

        return trainscore, testscore, cm
