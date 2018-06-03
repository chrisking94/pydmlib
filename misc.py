from base import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


class ConfusionMatrix(pd.DataFrame, RSObject):
    def __init__(self, y_test, y_pred, labels=None, name='ConfusionMatrix', **kwargs):
        if 'data' in kwargs.keys():
            cm = kwargs['data']
        else:
            cm = confusion_matrix(y_test, y_pred, labels)
        pd.DataFrame.__init__(self, cm, index=labels, columns=labels)
        RSObject.__init__(self, name, 'blue', 'default', 'bold')

    def normalized(self):
        return ConfusionMatrix(self.index, self.columns, labels=None, data=self.div(self.sum(axis=1), axis=0))

    def show(self, bnormalize=False):
        if (bnormalize):
            self.msg("Normalized\n%s" % self.normalized().__str__())
        else:
            self.msg('Without normalizing\n%s' % self.__str__())

    def plot(self, size=1):
        class_count = self.shape[0]
        bsize = size
        size *= class_count
        fig = plt.figure(figsize=(size, size))
        fig.suptitle(self.name)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        nmcm = self.normalized()
        blocksize = 1 / class_count
        halfbs = blocksize / 2
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_xticks(np.arange(halfbs, 1, blocksize))
        ax.set_xticklabels(self.columns)
        ax.set_yticks(np.arange(halfbs, 1, blocksize))
        ax.set_yticklabels(self.columns)
        for x in range(class_count):
            for y in range(class_count):
                ax.text(halfbs + x * blocksize, halfbs + y * blocksize, round(nmcm[y, x], 3),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=15 * bsize, color='orange')
        ax.imshow(self.normalized().values, interpolation='nearest', cmap=plt.cm.Blues)
        plt.show()

    def __getitem__(self, item):
        return super(ConfusionMatrix, self).values[item]


class ROCCurve(RSObject):
    def __init__(self, y_true, y_score, title='ROC-Curve'):
        RSObject.__init__(self, title, 'white', 'black', 'bold')
        self.fpr, self.tpr, self.thresholds = roc_curve(y_true,
                                                        y_score)  # y_score can be the probability of the POSITIVE class,...

    def plot(self, ax=None, label=''):
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
        plt.title(self.name)
        plt.xlabel('FP Rate')
        plt.ylabel('TP Rate')
        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)
        plt.legend(loc="lower right")
        plt.plot(self.fpr, self.tpr, lw=1, label=label)
        # ax.plot(self.thresholds, self.tpr)
        # ax.plot(self.fpr, self.thresholds)
        plt.text(0.7, 0.1, 'AUC=%.3f' % self.auc(), horizontalalignment='center',
          verticalalignment='center', fontdict={'size':20})
        plt.show()

    def auc(self):
        return auc(self.fpr, self.tpr)


def test():
    # cm = ConfusionMatrix([1, 2, 3, 2, 1, 2], [1, 2, 3, 2, 5, 6], [1, 2, 3])
    # print(cm)
    # print(cm[1, 1])
    # cm.show(True)
    # cm.plot()
    # from sklearn.datasets import load_iris
    # X, y = load_iris(True)
    # X, y = X[y!=2], y[y!=2]
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier()
    # clf.fit(X, y)
    # prob = clf.predict_proba(X)[:,1]
    # ROCCurve(y, prob).plot()
    # print(prob)
    pass