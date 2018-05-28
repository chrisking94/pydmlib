from base import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interpolate


class ConfusionMatrix(pd.DataFrame, RSObject):
    def __init__(self, y_test, y_pred, labels=None):
        pd.DataFrame.__init__(self, confusion_matrix(y_test, y_pred, labels), index=labels, columns=labels)
        RSObject.__init__(self, 'ConfusionMatrix', 'blue', 'default', 'bold')

    def normalized(self):
        return self / self.sum(axis=1)

    def show(self, bnormalize=False):
        if (bnormalize):
            self.msg("Normalized\n%s" % self.normalized().__str__())
        else:
            self.msg('Without normalizing\n%s' % self.__str__())

    def draw(self):
        plt.imshow(self.values, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(self.name)
        plt.show()

    def getclassscores(self):
        scores = []
        for i in range(self.shape[0]):
            scores.append(self.normalized().values[i, i])
        return scores

    def __getitem__(self, item):
        return super(ConfusionMatrix, self).values[item]


class ROCCurve(RSObject):
    def __init__(self, y_true, y_score, title='ROC-Curve'):
        RSObject.__init__(self, title, 'white', 'black', 'bold')
        self.fpr, self.tpr, self.thresholds = roc_curve(y_true,
                                                        y_score)  # y_score can be the probability of the POSITIVE class,...

    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
        ax.set_title(self.name)
        ax.set_xlabel('FP Rate')
        ax.set_ylabel('TP Rate')
        ax.legend(loc="lower right")

        #         srtd = list(zip(self.fpr, self.tpr))
        #         srtd.sort(key=lambda x:x[0])
        #         ax.scatter(self.fpr, self.tpr)
        #         x, y = [], []
        #         lastxx = -100
        #         for xx, yy in srtd:
        #             if xx != lastxx:
        #                 x.append(xx)
        #                 y.append(yy)
        #             lastxx = xx
        #         x0, y0 = x, y
        #         x2 = np.arange(0, 1, 0.05)
        #         A2, B2, C2 = optimize.curve_fit(f_2, x0, y0)[0]
        #         y2 = A2*x2*x2 + B2*x2 + C2
        #         # 拟合之后的平滑曲线图
        #         ax.plot(x2, y2)
        ax.plot(self.fpr, self.tpr, lw=1)
        # ax.plot(self.thresholds, self.tpr)
        # ax.plot(self.fpr, self.thresholds)
        plt.show()
        self.msg('AUC=%f' % self.auc())

    def auc(self):
        return auc(self.fpr, self.tpr)


def test():
    cm = ConfusionMatrix([1, 2, 3, 2, 1, 2], [1, 2, 3, 2, 5, 6], [1, 2, 3])
    print(cm)
    print(cm[1, 1])
    cm.show(True)
    cm.draw()
    pass