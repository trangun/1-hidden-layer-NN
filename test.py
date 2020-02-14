#!/usr/bin/env python3

import numpy as np
from io import StringIO

NUM_FEATURES = 124  # features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "E:/School/@Grad School/Machine Learning/HW1/adult"  # TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are


# returns the label and feature value vector for one data point (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y, 0)  # treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature - 1] = value
    x[-1] = 1  # bias
    return y, x


# return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals], [v[1] for v in vals])
        return np.asarray([ys], dtype=np.float32).T, np.asarray(xs, dtype=np.float32).reshape(len(xs), NUM_FEATURES,
                                                                                              1)  # returns a tuple, first is an array of labels, second is an array of feature vectors


def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1, len(w2))
    else:
        # TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES)  # bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1)  # add bias column

    # At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.

    # TODO: Replace this with whatever you want to use to represent the network; you could use use a tuple of (w1,w2), make a class, etc.

    class Model():
        def __init__(self):
            self.w1 = w1
            self.w2 = w2
            self.a1 = np.zeros([w1.shape[0], 1])
            self.z1 = np.zeros([w1.shape[0] + 1, 1])  # add bias
            self.z1[-1, :] = 1 #bias =1
            self.a2 = 0
            self.y_hat = 0

        def sigmoid(self, x):
            s = 1 / (1 + np.exp(-x))
            return s

        def d_sigmoid(self, x):
            s = self.sigmoid(x)
            ds = s * (1 - s)
            return ds

        def forward(self, xs):
            self.a1 = np.dot(self.w1, xs)
            self.z1[:-1, :] = self.sigmoid(self.a1)
            self.a2 = np.dot(self.w2, self.z1)
            self.y_hat = self.sigmoid(self.a2)

        def backprop(self, ys, xs):
            w2 = self.w2
            self.w2 = self.w2 - args.lr * (self.y_hat - ys) * self.z1.T
            dz1 = self.d_sigmoid(self.a1)
            self.w1 = self.w1 - args.lr * (self.y_hat - ys) * np.dot(np.multiply(dz1, w2[:, :-1].T), xs.T)

    model = Model()
    return model


def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    # TODO: Implement training for the given model, respecting args
    dev_accuracy = 0.0

    while args.iterations:
        for i in range(len(train_ys)):
            model.forward(train_xs[i])
            model.backprop(train_ys[i], train_xs[i, :, :])
        args.iterations -= 1
        if not args.nodev:
            accuracy = test_accuracy(model, dev_ys, dev_xs)
            if accuracy > dev_accuracy:
                dev_accuracy = accuracy
    return model


def test_accuracy(model, test_ys, test_xs):
    accuracy = 0.0
    # TODO: Implement accuracy computation of given model on the test data
    for i in range(len(test_ys)):
        model.forward(test_xs[i, :, :])
        if model.y_hat <= 0.5:
            y = 0
        else:
            y = 1
        y = np.sign(y)
        if y == test_ys[i]:
            accuracy += 1
    return accuracy / len(test_ys)


def extract_weights(model):
    # TODO: Extract the two weight matrices from the model and return them (they should be the same type and shape as they were in init_model, but now they have been updated during training)
    w1 = model.w1
    w2 = model.w2
    return w1, w2


def experiment(model, train_ys, train_xs, dev_ys, dev_xs, test_ys, test_xs, args):
    import matplotlib.pyplot as plt

    iterations = 60
    lr = [1.0, 0.25, 0.1, 0.01]
    hidden_dim = [5, 15, 25, 35]
    dev_acc = np.zeros([len(lr), len(hidden_dim), iterations])
    test_acc = np.zeros([len(lr), len(hidden_dim), iterations])
    args.iterations = iterations
    for i in range(len(lr)):
        for j in range(len(hidden_dim)):
            args.hidden_dim = hidden_dim[j]
            args.lr = lr[i]
            model = init_model(args)
            for t in range(0, args.iterations):
                for p in range(train_ys.shape[0]):
                    model.forward(train_xs[p])
                    model.backprop(train_ys[p], train_xs[p, :, :])
                dev_acc[i, j, t] = test_accuracy(model, dev_ys, dev_xs)
                test_acc[i, j, t] = test_accuracy(model, test_ys, test_xs)
    ite = list(range(iterations))
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(ite, dev_acc[0, 0, :])
    axs[0, 0].plot(ite, test_acc[0, 0, :])
    axs[0, 0].set_title('lr = 1, hidden_dim = 5')

    axs[0, 1].plot(ite, dev_acc[0, 1, :])
    axs[0, 1].plot(ite, test_acc[0, 1, :])
    axs[0, 1].set_title('lr = 1, hidden_dim = 15')

    axs[1, 0].plot(ite, dev_acc[0, 2, :])
    axs[1, 0].plot(ite, test_acc[0, 2, :])
    axs[1, 0].set_title('lr = 1, hidden_dim = 25')

    axs[1, 1].plot(ite, dev_acc[0, 3, :])
    axs[1, 1].plot(ite, test_acc[0, 3, :])
    axs[1, 1].set_title('lr = 1, hidden_dim = 35')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel='accuracy')
        # ax.label_outer()
        ax.legend(['dev_acc', 'test_acc'])
    plt.show()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(ite, dev_acc[1, 0, :])
    axs[0, 0].plot(ite, test_acc[1, 0, :])
    axs[0, 0].set_title('lr = 0.25, hidden_dim = 5')

    axs[0, 1].plot(ite, dev_acc[1, 1, :])
    axs[0, 1].plot(ite, test_acc[1, 1, :])
    axs[0, 1].set_title('lr = 0.25, hidden_dim = 15')

    axs[1, 0].plot(ite, dev_acc[1, 2, :])
    axs[1, 0].plot(ite, test_acc[1, 2, :])
    axs[1, 0].set_title('lr = 0.25, hidden_dim = 25')

    axs[1, 1].plot(ite, dev_acc[1, 3, :])
    axs[1, 1].plot(ite, test_acc[1, 3, :])
    axs[1, 1].set_title('lr = 0.25, hidden_dim = 35')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel='accuracy')
        # ax.label_outer()
        ax.legend(['dev_acc', 'test_acc'])
    plt.show()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(ite, dev_acc[2, 0, :])
    axs[0, 0].plot(ite, test_acc[2, 0, :])
    axs[0, 0].set_title('lr = 0.1, hidden_dim = 5')

    axs[0, 1].plot(ite, dev_acc[2, 1, :])
    axs[0, 1].plot(ite, test_acc[2, 1, :])
    axs[0, 1].set_title('lr = 0.1, hidden_dim = 15')

    axs[1, 0].plot(ite, dev_acc[2, 2, :])
    axs[1, 0].plot(ite, test_acc[2, 2, :])
    axs[1, 0].set_title('lr = 0.1, hidden_dim = 25')

    axs[1, 1].plot(ite, dev_acc[2, 3, :])
    axs[1, 1].plot(ite, test_acc[2, 3, :])
    axs[1, 1].set_title('lr = 0.1, hidden_dim = 35')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel='accuracy')
        # ax.label_outer()
        ax.legend(['dev_acc', 'test_acc'])
    plt.show()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(ite, dev_acc[3, 0, :])
    axs[0, 0].plot(ite, test_acc[3, 0, :])
    axs[0, 0].set_title('lr = 0.01, hidden_dim = 5')

    axs[0, 1].plot(ite, dev_acc[3, 1, :])
    axs[0, 1].plot(ite, test_acc[3, 1, :])
    axs[0, 1].set_title('lr = 0.01, hidden_dim = 15')

    axs[1, 0].plot(ite, dev_acc[3, 2, :])
    axs[1, 0].plot(ite, test_acc[3, 2, :])
    axs[1, 0].set_title('lr = 0.01, hidden_dim = 25')

    axs[1, 1].plot(ite, dev_acc[3, 3, :])
    axs[1, 1].plot(ite, test_acc[3, 3, :])
    axs[1, 1].set_title('lr = 0.01, hidden_dim = 35')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel='accuracy')
        # ax.label_outer()
        ax.legend(['dev_acc', 'test_acc'])
    plt.show()


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    parser.add_argument('--plot', action='store_true', default=False, help='if provided, plot will be given')

    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))
    if args.plot:
        experiment(model, train_ys, train_xs, dev_ys, dev_xs, test_ys, test_xs, args)


if __name__ == '__main__':
    main()
