from base import *
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(pd.DataFrame, RSObject):
    def __init__(self, y_test, y_pred, labels=None):
        super(ConfusionMatrix, self).__init__(confusion_matrix(y_test, y_pred, labels), index=labels, columns=labels)
        RSObject.__init__(self, 'ConfusionMatrix', 'blue', 'default', 'bold')

    def normalize(self):
        return self / self.sum(axis=1)

    def show(self, bnormalize=False):
        if (bnormalize):
            self.msg("Normalized\n%s" % self.normalize().__str__())
        else:
            self.msg('Without normalizing\n%s' % self.__str__())

    def draw(self):
        plt.imshow(self.values, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(self.name)
        plt.show()

    def getclassscores(self):
        scores = []
        for i in range(self.shape[0]):
            scores.append(self.values[i, i])
        return scores

    def __getitem__(self, item):
        return super(ConfusionMatrix, self).values[item]

def test():
    cm = ConfusionMatrix([1, 2, 3, 2, 1, 2], [1, 2, 3, 2, 5, 6], [1, 2, 3])
    print(cm)
    print(cm[1, 1])
    cm.show(True)
    cm.draw()
    pass