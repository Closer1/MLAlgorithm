import numpy as np


class LogisticRegression(object):

    def __init__(self):
        self.b = 0
        self.n_features = None
        self.w = None
        self.learning_rate = None

    def _sigmoid(self, x):
        z = np.dot(x, self.w) + self.b
        return np.exp(z)/(1 + np.exp(z))

    def _loss(self, x, y):
        loss = -np.sum(y * np.log(self._sigmoid(x)) + (1-y)*np.log(1-self._sigmoid(x)))/len(x)
        return loss

    def train(self, train_datas, train_labels, max_epoch=100000, lr=0.001):
        self.n_features = len(train_datas[0])
        self.w = np.random.randn(self.n_features, )
        self.learning_rate = lr

        num_data = len(train_datas)
        for i in range(max_epoch):
            grad_w = - np.dot(train_labels - self._sigmoid(train_datas), train_datas)/num_data
            grad_b = - np.sum(train_labels - self._sigmoid(train_datas))/num_data

            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

            if i % 100 == 0:
                loss = self._loss(train_datas, train_labels)
                print("{} / {} , loss: {}".format(i, max_epoch, loss))

    def _predict(self, x):
        p = self._sigmoid(x)
        p[p>0.5]=1
        p[p<=0.5]=0
        return p

    def _score(self, x, y):
        predicts = self._predict(x)
        correct = np.sum(predicts == y)
        score = correct/len(y) * 100
        return score








