import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
import wandb
import seaborn as sn

def load_data(dataset='fashion_mnist', purpose='train'):
    dataset = dataset.lower()
    purpose = purpose.lower()

    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")

    if purpose == 'train':
        x_train = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.
        y_train = np.eye(10)[y_train]
        return x_train, y_train

    elif purpose == 'test':
        x_test = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.
        y_test = np.eye(10)[y_test]
        return x_test, y_test

class FF_NN:
    def __init__(self, param):
        self.hidden_layers = param['hidden_lyrs']
        self.neurons = param['neurons']
        self.input_neurons = param['inpt_sz']
        self.output_neurons = param['oupt_sz']
        self.activation = param['activation']
        self.output_activation = param['oupt_activation']
        self.weight_initialisation = param['weight_initialisation']

        self.weights = []
        self.bias = []

        self.initialize_weights()
        self.initialize_bias()

    def initialize_bias(self):
        # Bias initialization for each hidden layer and output layer
        for _ in range(self.hidden_layers):
            self.bias.append(np.random.randn(self.neurons))
        self.bias.append(np.random.randn(self.output_neurons))

    def initialize_weights(self):
        # Initialize weights based on chosen method
        if self.weight_initialisation.lower() == 'random':
            self.weights.append(np.random.randn(self.input_neurons, self.neurons))
            for _ in range(1, self.hidden_layers):
                self.weights.append(np.random.randn(self.neurons, self.neurons))
            self.weights.append(np.random.randn(self.neurons, self.output_neurons))
        else:
            # Xavier initialization (custom setup)
            self.setup_custom_weights()

    def setup_custom_weights(self):
        # Xavier initialization for input to first hidden layer
        limit_inp_hidden = np.sqrt(6 / (self.input_neurons + self.neurons))
        self.weights.append(np.random.uniform(-limit_inp_hidden, limit_inp_hidden,
                                              (self.input_neurons, self.neurons)))

        # Xavier initialization for hidden layers
        limit_hidden_hidden = np.sqrt(6 / (self.neurons + self.neurons))
        for _ in range(1, self.hidden_layers):
            self.weights.append(np.random.uniform(-limit_hidden_hidden, limit_hidden_hidden,
                                                  (self.neurons, self.neurons)))

        # Xavier initialization from last hidden layer to output layer
        limit_hidden_out = np.sqrt(6 / (self.neurons + self.output_neurons))
        self.weights.append(np.random.uniform(-limit_hidden_out, limit_hidden_out,
                                              (self.neurons, self.output_neurons)))

    def apply_activation(self, data):
        activation_fn = self.activation.lower()
        
        if activation_fn == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(data, -500, 500)))
        
        elif activation_fn == 'relu':
            return np.maximum(0.0, data)
        
        elif activation_fn == 'tanh':
            return np.tanh(data)
        
        else:  # identity activation function
            return data

    def apply_output_activation(self, data):
        if self.output_activation.lower() == 'softmax':
            exp_data = np.exp(np.clip(data, -500, 500))
            return exp_data / np.sum(exp_data, axis=1, keepdims=True)

    def feed_forward(self, input_data):
        # Store activations and pre-activations layer-wise
        self.A = []
        self.H = [input_data]

        # Forward propagation through hidden layers
        for idx in range(self.hidden_layers):
            pre_activation = np.dot(self.H[-1], self.weights[idx]) + self.bias[idx]
            activation_output = self.apply_activation(pre_activation)

            # Save intermediate results
            self.A.append(pre_activation)
            self.H.append(activation_output)

        # Forward propagation through output layer
        final_pre_activation = np.dot(self.H[-1], self.weights[-1]) + self.bias[-1]
        
        final_output_activation = self.apply_output_activation(final_pre_activation)

        # Save final results
        self.A.append(final_pre_activation)
        self.H.append(final_output_activation)

        return final_output_activation


# Parameters for testing the neural network forward pass
PARAMETERS = {
    'inpt_sz': 784,
    'oupt_sz': 10,
    'neurons': 32,
    'hidden_lyrs': 4,
    'activation': 'sigmoid',
    'oupt_activation': 'softmax',
    'dataset': 'fashion_mnist',
    'weight_initialisation': 'xavier',
}

# Creating neural network instance and loading training data
nn = FF_NN(PARAMETERS)
x_train, y_train = load_data(PARAMETERS['dataset'], purpose='train')

# Testing forward propagation with training data
prediction = nn.feed_forward(x_train)  # shape of x_train -> (60000, 784)
print(prediction[0])
