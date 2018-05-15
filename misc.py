from rsobject import *
from sklearn.metrics import confusion_matrix

class ConfusionMatrix(RSObject):
    def __init__(self, y_test, y_pred):
        super(ConfusionMatrix, self).__init__('ConfusionMatrix', 'blue', 'default', 'bold')
        self.cm = confusion_matrix(y_test, y_pred)

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

    def __getitem__(self, index):
        return self.normalize()[index]
