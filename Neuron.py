import math
import random
from statistics import mean

def Sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dertvSigmoid(x):
    return Sigmoid(x) * (1-Sigmoid(x))


def mse_loss(y, y_pred):
    return mean((y - y_pred) ** 2)


class NeuralNetwork:
    def __init__(self):
        #miałem problem z numpy dlatego starałęm się znaleźć alternatywe
        self.w0 = (random.getrandbits(1))
        self.w1 = (random.getrandbits(1))
        self.w2 = (random.getrandbits(1))
        self.w3 = (random.getrandbits(1))
        self.w4 = (random.getrandbits(1))
        self.w5 = (random.getrandbits(1))

        self.b0 = (random.getrandbits(1))
        self.b1 = (random.getrandbits(1))
        self.b2 = (random.getrandbits(1))

    def feedforward(self, x):
        H0 = Sigmoid(self.w0 * x[0] + self.w1 * x[1] + self.b0)
        H1 = Sigmoid(self.w2 * x[0] + self.w3 * x[1] + self.b1)
        O0 = Sigmoid(H0 * self.w4 + H1 * self.w5 + self.b2)
        return O0

    def train(self, data, y, learn_rate, epochs):
        for i in range(epochs):
            for x, y in data:
                # feed_forward
                h0 = Sigmoid(self.w0 * x[0] + self.w1 * x[1] + self.b0)
                h1 = Sigmoid(self.w2 * x[0] + self.w3 * x[1] + self.b1)
                o0 = Sigmoid(h0 * self.w4 + h1 * self.w5 + self.b2)

                mse = -2 * (1 - o0)

                # DELTA
                # neuron O0
                tmp_o0 = h0 * self.w4 + h1 * self.w5 + self.b2
                d_w4 = h0 * dertvSigmoid(tmp_o0)
                d_w5 = h1 * dertvSigmoid(tmp_o0)
                d_b2 = dertvSigmoid(tmp_o0)
                d_h0 = self.w4 * tmp_o0
                d_h1 = self.w5 * tmp_o0

                #  neuron H1
                tmp_h1 = self.w2 * x[0] + self.w3 * x[1] + self.b1
                d_w3 = x[1] * dertvSigmoid(tmp_h1)
                d_w2 = x[0] * dertvSigmoid(tmp_h1)
                d_b1 = dertvSigmoid(tmp_h1)

                # neuron H0
                tmp_h0 = self.w0 * x[0] + self.w1 * x[1] + self.b0
                d_w0 = x[0] * dertvSigmoid(tmp_h0)
                d_w1 = x[1] * dertvSigmoid(tmp_h0)
                d_b0 = dertvSigmoid(tmp_h0)

                # aktualizacja wag
                # Neuron H0
                self.w0 -= learn_rate * d_w0 * d_h0 * mse
                self.w1 -= learn_rate * d_w1 * d_h0 * mse
                self.b0 -= learn_rate * d_b0 * d_h0 * mse

                # Neuron H1
                self.w2 -= learn_rate * d_w2 * d_h1 * mse
                self.w3 -= learn_rate * d_w3 * d_h1 * mse
                self.b1 -= learn_rate * d_b1 * d_h1 * mse

                # Neuron O0
                self.w4 -= learn_rate * d_w4 * mse
                self.w5 -= learn_rate * d_w5 * mse
                self.b2 -= learn_rate * d_b2 * mse
            if i % 10 == 0:
                y_pred = self.feedforward(data)
                loss = mse_loss(y, y_pred)
                print(loss)
                print(y_pred)


data = [[0, 0], [0, 1], [1, 0], [1, 1]]

results = [0, 0, 0, 1]
                
learn_rate = 0.1
epochs = 1000
network = NeuralNetwork()
network.train(data, results, learn_rate, epochs)


32035689+JakubSmarzewski@users.noreply.github.com





                







