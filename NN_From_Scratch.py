from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

class MyNN:
    def __init__(self, learning_rate, layer_sizes):
        '''
        learning_rate - the learning to use in backward
        layer_sizes - a list of numbers, each number represents the number of neurons
                      to have in every layer.
                      Therfore, the length of the list
                      represents the number of layers this network has.
        '''
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.model_params = {}
        self.memory = {}
        self.grads = {}

        # Initializing weights
        for layer_index in range(len(layer_sizes) - 1):
            W_input = layer_sizes[layer_index + 1]
            W_output = layer_sizes[layer_index]
            self.model_params['W_' + str(layer_index + 1)] = np.random.randn(W_input, W_output) * 0.1
            self.model_params['b_' + str(layer_index + 1)] = np.random.randn(W_input) * 0.1

    def forward_single_instance(self, x):
        a_i_1 = x
        self.memory['a_0'] = x
        for layer_index in range(len(self.layer_sizes) - 1):
            W_i = self.model_params['W_' + str(layer_index + 1)]
            b_i = self.model_params['b_' + str(layer_index + 1)]
            z_i = np.dot(W_i, a_i_1) + b_i
            a_i = 1 / (1 + np.exp(-z_i))
            self.memory['a_' + str(layer_index + 1)] = a_i
            a_i_1 = a_i
        return a_i_1

    def log_loss(y_hat, y):
        '''
        Logistic loss, assuming a single value in y_hat and y.
        '''
        cost = -y[0] * np.log(y_hat[0]) - (1 - y[0]) * np.log(1 - y_hat[0])
        return cost

    def backward_single_instance(self, y):
        a_output = self.memory['a_' + str(len(self.layer_sizes) - 1)]
        dz = a_output - y

        for layer_index in range(len(self.layer_sizes) - 1, 0, -1):
            print(layer_index)
            a_l_1 = self.memory['a_' + str(layer_index - 1)]
            dW = np.dot(dz.reshape(-1, 1), a_l_1.reshape(1, -1))
            self.grads['dW_' + str(layer_index)] = dW
            W_l = self.model_params['W_' + str(layer_index)]
            dz = (a_l_1 * (1 - a_l_1)).reshape(-1, 1) * np.dot(W_l.T, dz.reshape(-1, 1))
            # calculate and memorize db as well:
            # Basically its the same as dz --> db[] = dz[]
            self.grads['db_' + str(layer_index)] = dz

    # update weights and biases with grads
    def update(self):
        for layer_index in range(len(self.layer_sizes) - 1):
            self.model_params['W_' + str(layer_index + 1)] -= self.learning_rate * self.grads[
                'dW_' + str(layer_index + 1)]
            self.model_params['b_' + str(layer_index + 1)] -= (
                        self.learning_rate * self.grads['db_' + str(layer_index + 1)])

    # TODO: implement forward for a batch X.shape = (network_input_size, number_of_instance)
    def forward_batch(self, X):
        A_i_1 = X
        self.memory['A_0'] = X

        for layer_index in range(len(self.layer_sizes) - 1):
            W_i = self.model_params['W_' + str(layer_index + 1)]
            b_i = self.model_params['b_' + str(layer_index + 1)]
            z_i = np.dot(W_i, A_i_1) + b_i.reshape(-1, 1)
            A_i = 1 / (1 + np.exp(-z_i))
            self.memory['A_' + str(layer_index + 1)] = A_i
            A_i_1 = A_i

        return A_i_1

    # TODO: implement backward for a batch y.shape = (1, number_of_instance)
    def backward_batch(self, y):
        A_output = self.memory['A_' + str(len(self.layer_sizes) - 1)]
        dZ = A_output - y
        mRatio = 1 / (y.shape[1])

        for layer_index in range(len(self.layer_sizes) - 1, 0, -1):
            # print(f"layer_index: {layer_index}")
            A_l_1 = self.memory['A_' + str(layer_index - 1)]
            dW = (mRatio) * (np.dot(dZ, A_l_1.T))
            self.grads['dW_' + str(layer_index)] = dW
            W_l = self.model_params['W_' + str(layer_index)]
            # calculate and memorize db as well:
            # calculate the relative effect of each bias using mean func on the 0 axis == column, y axis
            self.grads['db_' + str(layer_index)] = dZ.T.mean(0)
            dZ = (A_l_1 * (1 - A_l_1)) * np.dot(W_l.T, dZ)

    # TODO: implement log_loss_batch, for a batch of instances
    def log_loss_batch(self, y_hat, y):
        m = y_hat.shape[1]
        cost = (-1 / m) * ((np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))).sum()
        return cost

def train(X, y, epochs, batch_size, nn):
  loss_list = []
  for e in range(1, epochs + 1):
    epoch_loss = 0

    X_shuffle, y_shuffle = shuffle(X.T, y.T)

    X_batches = np.array_split(X_shuffle, batch_size)
    y_batches = np.array_split(y_shuffle, batch_size)

    batch_zip = zip(X_batches, y_batches)
    for X_b, y_b in batch_zip:
      #print(f"Forwarding batch")
      y_hat = nn.forward_batch(X_b.T)
      epoch_loss += nn.log_loss_batch(y_hat, y_b.T)
      #print(f"backwarding batch")
      nn.backward_batch(y_b.T)
      nn.update()

    loss_list.append(epoch_loss/len(X_batches))
    print(f'Epoch {e}, loss={epoch_loss/len(X_batches)}')
  return loss_list

def main():
    nn = MyNN(0.001, [6, 4, 3, 1])

    X = np.random.randn(6, 100)
    y = np.random.randn(1, 100)
    batch_size = 8
    epochs = 8

    loss_list = train(X, y, epochs, batch_size, nn)

    # loss plot in-order to use over unix machine
    # run in terminal $sudo apt-get install python3-tk
    plt.plot(loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()