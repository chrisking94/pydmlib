from base import *
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(pd.DataFrame, RSObject):
    def __init__(self, y_test, y_pred, labels=None):
        super(ConfusionMatrix, self).__init__()
        RSObject.__init__(self, 'ConfusionMatrix', 'blue', 'default', 'bold')
        self.cm = confusion_matrix(y_test, y_pred, labels)

    def normalize(self):
        return self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]

    def show(self, bnormalize=False):
        if (bnormalize):
            self.msg("Normalized\n%s" % self.normalize().__str__())
        else:
            self.msg('Without normalizing\n%s' % self.cm.__str__())

    def draw(self):
        plt.imshow(self.cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(self.name)
        plt.show()

    def getclassscores(self):
        scores = []
        for i in range(self.cm.shape[0]):
            scores.append(self.cm[i, i])
        return scores

    def __getitem__(self, index):
        return self.normalize()[index]
