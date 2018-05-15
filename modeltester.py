from rsobject import *
from misc import ConfusionMatrix
import sklearn.cross_validation as cv


class ModelTester(RSObject):
    def __init__(self, classifier):
        super(ModelTester, self).__init__('ModelTester', 'red', 'default', 'highlight')
        self.clf = classifier

    def fit_transform(self, data, msg=''):
        '''
       开始测试
       :param data: 测试数据，可以是：
           1、整个数据集
           2、tuple(trainset，testset)
       :param clf: 分类器
       '''
        clf = self.clf
        self.msg('\033[1;31;47m%s\033[0m \t开始时间：' % ('↓' * 40))
        printtime()
        self.msg('附加说明：%s' % msg)
        self.msg('分类器信息：%s' % clf.__str__())
        self.starttimer()

        if (isinstance(data, tuple)):
            trainset, testset = data[0], data[1]
        else:
            trainset, testset = cv.train_test_split(data, test_size=0.2, random_state=0)

        x_train, y_train = trainset[trainset.columns[:-1]], trainset[trainset.columns[-1]]
        x_test, y_test = testset[testset.columns[:-1]], testset[testset.columns[-1]]

        self.msg('使用特征：%s' % x_train.columns.__str__())

        clf.fit(x_train, y_train)
        trainscore = clf.score(x_train, y_train)
        self.msg('testing on train set,score: %f' % trainscore)
        y_pred = clf.predict(x_test)
        testscore = clf.score(x_test, y_test)
        self.msg('testing on test set,score: %f' % testscore)

        unique_items, cnts = np.unique(y_test, return_counts=True)
        cm = ConfusionMatrix(y_test, y_pred)
        cm.show()
        cm.show(True)
        cm.draw()

        self.msg('↑' * 40)
        self.msgtimecost()

        return trainscore, testscore, cm