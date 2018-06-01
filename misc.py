from base import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interpolate


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

    def draw(self, size=1):
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

    def plot(self, ax=None, label=''):
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
        ax.plot(self.fpr, self.tpr, lw=1, label=label)
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