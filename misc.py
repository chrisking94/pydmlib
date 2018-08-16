from .base import *
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.metrics import fbeta_score


class RSPlot(RSObject):
    def __init__(self, name=''):
        RSObject.__init__(self, name=name)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.set_title(self.name)
        ax.set_label(self.name)
        self._draw(ax, **kwargs)

    def _draw(self, ax, **kwargs):
        self.error('Not implemented!')

    def _size(self):
        return 5, 5


class ConfusionMatrix(RSPlot, pd.DataFrame):
    def __init__(self, y_test, y_pred, labels=None, name='ConfusionMatrix', **kwargs):
        if 'data' in kwargs.keys():
            cm = kwargs['data']
        else:
            cm = confusion_matrix(y_test, y_pred, labels)
        pd.DataFrame.__init__(self, cm, index=labels, columns=labels)
        RSPlot.__init__(self, name)

    def normalized(self):
        return ConfusionMatrix(self.index, self.columns, labels=None, data=self.div(self.sum(axis=1), axis=0))

    def show(self, bnormalize=False):
        if (bnormalize):
            self.msg("Normalized\n%s" % self.normalized().__str__())
        else:
            self.msg('Without normalizing\n%s' % self.__str__())

    def _size(self):
        size = 5
        class_count = self.shape[0]
        size *= class_count
        return size, size

    def _draw(self, ax, **kwargs):
        class_count = self.shape[0]
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        nmcm = self.normalized()
        ax.set_title('%s\n' % self.name)
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        ax.set_xticks(np.arange(0, class_count, 1))
        ax.set_xticklabels(self.columns)
        ax.set_yticks(np.arange(0, class_count, 1))
        ax.set_yticklabels(self.columns)
        for x in range(class_count):
            for y in range(class_count):
                ax.text(x, y, round(nmcm[y, x], 3),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=30, color='orange')
        ax.imshow(nmcm.values, interpolation='nearest', cmap=plt.cm.Blues)

    def __getitem__(self, item):
        return self.values[item]

    def __str__(self):
        return pd.DataFrame.__str__(self)


class ROCCurve(RSPlot, pd.DataFrame):
    def __init__(self, y_true, y_score, pos_label=None, title='ROC-Curve'):
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
        pd.DataFrame.__init__(self, data={'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
        RSPlot.__init__(self, name=title)
        # y_score can be the probability of the POSITIVE class,...

    def _draw(self, ax, **kwargs):
        ax.set_xlabel('FP Rate')
        ax.set_ylabel('TP Rate')
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        # ax.legend(loc="lower right")
        ax.plot(self.fpr, self.tpr, lw=1, **kwargs)
        ax.plot([0, 1], [0, 1], linestyle='--')
        # ax.plot(self.thresholds, self.tpr)
        # ax.plot(self.fpr, self.thresholds)
        ax.text(0.7, 0.1, 'AUC=%.3f' % self.auc(), horizontalalignment='center',
                verticalalignment='center', fontdict={'size':20})

    def auc(self):
        return auc(self.fpr, self.tpr)


class PRCurve(RSPlot, pd.DataFrame):
    def __init__(self, y_true, y_score, pos_label=None, title='PR-Curve'):
        precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=pos_label)
        thresholds = np.concatenate((np.array([0]), thresholds))
        pd.DataFrame.__init__(self, data={'precision': precision, 'recall': recall, 'thresholds': thresholds})
        RSPlot.__init__(self, name=title)
        # y_score can be the probability of the POSITIVE class,...

    def _draw(self, ax, **kwargs):
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        # ax.legend(loc="lower right")
        ax.plot(self.precision, self.recall, lw=1, **kwargs)
        # ax.plot([0, 1], [0, 1], linestyle='--')
        # ax.text(0.7, 0.1, 'AUC=%.3f' % self.auc(), horizontalalignment='center',
        #         verticalalignment='center', fontdict={'size':20})

    def auc(self):
        return auc(self.precision, self.recall, reorder=True)


class FBetaScore(RSPlot, pd.DataFrame):
    def __init__(self, y_true, y_pred, name='FBetaScore'):
        betas = np.arange(0.0001, 1, 0.01)
        scores = betas.copy()
        for i, beta in enumerate(betas):
            scores[i] = fbeta_score(y_true, y_pred, beta, average='binary')
        pd.DataFrame.__init__(self, {'betas': betas, 'scores': scores})
        RSPlot.__init__(self, name)

    def _draw(self, ax, **kwargs):
        ax.set_xlabel('Beta')
        ax.set_ylabel('FScore')
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.plot(self.betas, self.scores, lw=1, **kwargs)
        ax.text(0.5, 0.1, 'F1Score=%.3f' % self.f1score(), horizontalalignment='center',
                verticalalignment='center', fontdict={'size':20})

    def f1score(self):
        return self['scores'].iloc[-1]


if __name__ == '__main__':
    # cm = ConfusionMatrix([1, 2, 3, 2, 1, 2], [1, 2, 3, 2, 5, 6], [1, 2, 5])
    # print(cm)
    # print(cm[1, 1])
    # cm.show(True)
    # fig = plt.figure(figsize=(10, 5))
    # cm.plot(fig.add_subplot(121))
    # from sklearn.datasets import load_iris
    # X, y = load_iris(True)
    # X, y = X[y!=2], y[y!=2]
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier()
    # clf.fit(X, y)
    # prob = clf.predict_proba(X)[:,1]
    # ROCCurve(y, prob).plot(fig.add_subplot(122))
    # plt.show()
    # print(prob)
    pass
