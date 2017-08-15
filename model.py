import chainer
import chainer.functions as F
import chainer.links as L
class LogisticRegression(chainer.Chain):

    def __init__(self):
        super(LogisticRegression, self).__init__(
            l1=L.Linear(None, 1))
    def __call__(self, x,apply_sigmoid=True):
        if apply_sigmoid:
            return F.sigmoid(self.l1(x))
        else:
            return self.l1(x)
