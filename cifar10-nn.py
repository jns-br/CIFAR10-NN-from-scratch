import numpy as np
import matplotlib.pyplot as plt
import helper
import os

np.random.seed(1)


class TwoLayerNeuralNetwork:
    """
    Fully-connected 2-Layer Neural Network. Input are Images as N vecotors with Dimension D. Hidden layer
    has H neurons. The output layer corresponds to the label set size, in this case a classification of 10
    image classes (airplane, automobile etc.) according to the CIFAR-10 dataset.
    https://www.cs.toronto.edu/~kriz/cifar.html

    The network is trained with a softmax-loss-function and L2 Regularization on the weight matrices (W1, W2).
    The Network uses a ReLU activation function after the first layer.
    Abstract architecture of the network.
    Input -> Fully connected layer -> ReLU -> Fully connected layer -> Softmax

    The output of the second layer are scores (probabilities) of an input image of each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialization of the Network - Weight matrices and bias vectors are initialized randomly
        Intitialisierung des Netzes - Die Gewichtungsmatrizen und die Bias-Vektoren werden mit
        Zufallswerten initialisiert.
        W1: Layer 1 weights, dimension = (D, H)
        b1: Layer 1 bias, dimensions = (H,)
        W2: Layer 2 weights, dimensions = (H, C)
        b2: Layer 2 bias, dimensions = (C,)

        :param input_size: CIFAR10 images have dimension = (32*32*3).
        :param hidden_size: number of neurons in hidden layer H.
        :param output_size: number of classes C = (10) .
        :param std: scaling factor for initialization
        :return:
        """
        self.W1 = std * np.random.randn(input_size, hidden_size)
        self.b1 = std * np.random.randn(1, hidden_size)
        self.W2 = std * np.random.randn(hidden_size, output_size)
        self.b2 = std * np.random.randn(1, output_size)

    def softmax(self, z):
        """
        Softmax function
        """
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def relu(self, x):
        """
        ReLU function
        """
        return np.maximum(0.0, x)

    def relu_derivative(self, output):
        """
        Derivative of ReLU function for backpropagation
        """
        output[output <= 0] = 0
        output[output > 0] = 1
        return output

    def loss_deriv_softmax(self, activation, y_batch):
        """
        Derivative of loss and softmax function for backpropagion
        """
        batch_size = y_batch.shape[0]
        dCda2 = activation
        dCda2[range(batch_size), y_batch] -= 1
        dCda2 /= batch_size
        return dCda2

    def loss_crossentropy(self, activation, y_batch):
        """
        Crossentropy loss for network

        :param batch_size: Number of input images in a batch over which loss must be normalized
        :param y: Vector with train labels y, y[i] contains label for X[i] and every y[i] is an integer
                  between 0 <= y[i] < C (number of classes)
        :return: loss (normalized loss of batch)
        """

        batch_size = y_batch.shape[0]
        correct_logprobs = -np.log(activation[range(batch_size), y_batch])
        loss = np.sum(correct_logprobs) / batch_size
        return loss

    def forward(self, X, y):
        """
        Forward pass of neural network, calculates loss and activations a1 and a2
        :param X: data
        :param y: labels
        :return: loss, a1, a2
        """

        mb1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(mb1)

        mb2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(mb2)

        loss = self.loss_crossentropy(a2, y)

        return loss, a1, a2

    def backward(self, a1, a2, X, y):
        """
        Backward pass of neural network - calculates gradients of weights W1, W2 and biases b1, b2 from the output
        of the network, gradients of the layers are returned in dictionary

        :param a1: Layer 1 activation
        :param a2: Layer 2 activation == output of network
        :param X: data
        :param y: labels
        :return:
        """
        grads = {}

        dC_da2 = self.loss_deriv_softmax(a2, y)
        dm2_dW2 = a1
        dm2_da1 = self.W2

        da1_dmb1 = self.relu_derivative(dm2_dW2)
        dm1_dW1 = X

        tmp1 = np.dot(dC_da2, dm2_da1.T)
        tmp2 = da1_dmb1 * tmp1
        grads['W1'] = np.dot(dm1_dW1.T, tmp2)
        grads['b1'] = np.sum(tmp2, axis=0)

        grads['W2'] = np.dot(a1.T, dC_da2)
        grads['b2'] = np.sum(dC_da2, axis=0)
        return grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95, num_iters=100,
              batch_size=200, verbose=False):
        """
        Training of neural network using stochastic gradient descent

        :param X: data, numpy array, dimensions = (N,D)
        :param y: labels, numpy array, dimensions = (N,)
        :param X_val: validation set, numpy array, dimensions = (N_val,D)
        :param y_val: validation labels, numpy array, dimensions = (N_val,)
        :param learning_rate: float, factor of learning rate of optimization process
        :param learning_rate_decay: float, adjustment of learning rate per epoch
        :param num_iters: int, number of iterations of optimization
        :param batch_size: int, number of training images for every forward pass
        :param verbose: boolean, true if there should be a console output
        :return: dict, contains loss and accuracy for every iteration/epoch
        """
        num_train = X.shape[0]
        iterations_per_epoch = int(max(num_train / batch_size, 1))

        loss_history = []
        loss_val_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):

            random_batch = np.random.choice(X_train.shape[0], batch_size)
            X_batch = X[random_batch]
            y_batch = y[random_batch]

            loss, a1, a2 = self.forward(X_batch, y_batch)
            grads = self.backward(a1, a2, X_batch, y_batch)

            loss_history.append(loss)

            loss_val, a1_val, a2_val = self.forward(X_val, y_val)
            loss_val_history.append(loss_val)

            self.W1 += -learning_rate * grads['W1']
            self.W2 += -learning_rate * grads['W2']
            self.b1 += -learning_rate * grads['b1']
            self.b2 += -learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:

                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                print('epoch done... acc', val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'loss_val_history': loss_val_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Benutzen Sie die trainierten Gewichte des 2-Layer-Netzes um die Klassen für das
        Validierungsset vorherzusagen. Dafür müssen Sie für das/die Eingabebilder X nur
        die Scores berechnen. Der höchste Score ist die vorhergesagte Klasse. Dieser wird in y_pred
        geschrieben und zurückgegeben.

        Prediction of new input images based on 2-Layer-Network already trained on training set. Calculates scores
        for every class and returns highest score.

        :param X: numpy array, dimensions = (N, D)
        :return: y_pred, numpy array, dimensions = (N,), predicted labels for every element in X
        """

        mb1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(mb1)

        mb2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(mb2)

        y_pred = np.argmax(a2, axis=1)

        return y_pred


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = helper.prepare_CIFAR10_images()

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)

    input_size = 32 * 32 * 3

    num_classes = 10

    hidden_size = 1100
    num_iter = 3000
    batch_size = 100
    learning_rate = 0.001
    learning_rate_decay = 0.97

    net = TwoLayerNeuralNetwork(input_size, hidden_size, num_classes)


    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=num_iter, batch_size=batch_size,
                      learning_rate=learning_rate, learning_rate_decay=learning_rate_decay, verbose=True)

    print('Final training loss: ', stats['loss_history'][-1])
    print('Final validation loss: ', stats['loss_val_history'][-1])

    print('Final validation accuracy: ', stats['val_acc_history'][-1])

    helper.plot_net_weights(net)
    helper.plot_accuracy(stats)
    helper.plot_loss(stats)

