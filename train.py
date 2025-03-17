import sys, argparse
from keras.datasets import fashion_mnist, mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import seaborn as sn
#from .\Q2.py import FF_NN,load_data
#from .\Q3(A).py import BP_NN
#from .\Q3(B).py import Optimizer



def parse_arguments():
    # Create argument parser object with a description
    parser = argparse.ArgumentParser(description='Arguments for training neural network model')

    # WandB project name argument
    parser.add_argument('-wp', '--wandb_project', type=str, default='dl_assignment_1',
                        help='Name of the project for tracking experiments in Weights & Biases dashboard')

    # WandB entity argument
    parser.add_argument('-we', '--wandb_entity', type=str, default='dl_assignment_1',
                        help='Entity name used in Weights & Biases dashboard for experiment tracking')

    # Dataset selection argument
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=["mnist", "fashion_mnist"],
                        help='Dataset to use for training (options: "mnist", "fashion_mnist")')

    # Number of epochs argument
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Total epochs for training the neural network')

    # Batch size argument
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Training batch size for neural network')

    # Loss function argument
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=["mean_squared_error", "cross_entropy"],
                        help='Loss function to optimize (options: "mean_squared_error", "cross_entropy")')

    # Optimizer selection argument
    parser.add_argument('-o', '--optimizer', type=str, default='nadam',
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help='Optimizer algorithm (options: "sgd", "momentum", "nag", "rmsprop", "adam", "nadam")')

    # Learning rate argument
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')

    # Momentum parameter argument (for momentum-based optimizers)
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum factor used in momentum and NAG optimizers')

    # Beta parameter for RMSProp optimizer
    parser.add_argument('-beta', '--beta', type=float, default=0.9,
                        help='Beta parameter for RMSProp optimizer')

    # Beta1 parameter for Adam and Nadam optimizers
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 parameter used by Adam/Nadam optimizers')

    # Beta2 parameter for Adam and Nadam optimizers
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999,
                        help='Beta2 parameter used by Adam/Nadam optimizers')

    # Epsilon parameter for numerical stability in optimizers
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-10,
                        help='Epsilon value used by optimizer algorithms to maintain numerical stability')

    # Weight decay regularization term
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005,
                        help='Weight decay regularization coefficient')

    # Weight initialization method argument
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=["random", "xavier"],
                        help='Method to initialize weights (options: "random", "xavier")')

    # Number of hidden layers in neural network architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=4,
                        help='Number of hidden layers in the feedforward neural network')

    # Size of hidden layers (number of neurons per layer)
    parser.add_argument('-sz', '--hidden_size', type=int, default=128,
                        help='Number of neurons in each hidden layer of the feedforward neural network')

    # Activation function choice argument
    parser.add_argument('-a', '--activation', type=str, default='tanh',
                        choices=["identity", "sigmoid", "tanh", "ReLU"],
                        help='Activation function to apply (options: "identity", "sigmoid", "tanh", "ReLU")')

    return parser.parse_args()



def plot_categories():

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	
    wandb.init(project="DA6401_Assignment_1", name="Q-1(test)")

    def fetch_data(dataset_name='fashion_mnist', data_type='train'):
        dataset_name, data_type = dataset_name.lower(), data_type.lower()
        (train_set, train_labels), (test_set, test_labels) = fashion_mnist.load_data()
        return transform_data(train_set, train_labels) if data_type == 'train' else transform_data(test_set, test_labels)
    
    def transform_data(data, labels):
        return data.reshape(data.shape[0], -1) / 255.0, np.eye(10)[labels]
    
    train_images, train_labels = fetch_data(data_type='train')
    test_images, test_labels = fetch_data(data_type='test')
    
    categories = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    selected_images, selected_labels = [], []
    for lbl in np.unique(train_labels, axis=0):
        first_index = np.argmax(np.all(train_labels == lbl, axis=1))  # Identify first instance
        selected_images.append(train_images[first_index])
        selected_labels.append(categories[np.argmax(lbl)])
    
    wandb.log({
        "For unique class sample images": [
            wandb.Image(img.reshape(28, 28), caption=label) for label, img in zip(selected_labels, selected_images)
        ]
    })

    wandb.finish()


def calculate_loss(y, y_pred, loss_function):
    # Convert loss function name to lowercase for consistency
    ls_fn = loss_function.lower()

    # Compute mean squared error loss if specified
    if ls_fn == "mean_squared_error":
        squared_diff = (y_pred - y) ** 2
        return np.mean(np.sum(squared_diff, axis=1))

    # Compute cross-entropy loss if specified
    elif ls_fn == "cross_entropy":
        epsilon = 1e-12  # small constant to prevent log(0)
        y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy_loss = -np.sum(y * np.log(y_pred_clipped), axis=1)
        return np.mean(cross_entropy_loss)

    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")


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



class BP_NN:
    def __init__(self, ff_nn: FF_NN, param):
        self.ff_nn = ff_nn
        self.loss = param['loss_function']
        self.activation = param['activation']
        self.output_activation = param['oupt_activation']

    def der_actvtn(self, x):
        activation_fn = self.activation.lower()
        if activation_fn == "sigmoid":
            return x * (1 - x)
        elif activation_fn == "tanh":
            return 1 - np.power(x, 2)
        elif activation_fn == "relu":
            return np.where(x > 0, 1, 0)
        elif activation_fn == "identity":
            return np.ones_like(x)

    def der_ls(self, y, yp):
        loss_fn = self.loss.lower()
        if loss_fn == "mean_squared_error":
            return yp - y
        elif loss_fn == "cross_entropy":
            return -y / yp

    def der_outpt_actvtn(self, yp):
        output_act = self.output_activation.lower()
        if output_act == "softmax":
            # Jacobian matrix for softmax derivative
            return np.diag(yp) - np.outer(yp, yp)

    def propogate_backward(self, y, y_pred):  # shapes: y=(60000,10), y_pred=(60000,10)
        self.d_h = []
        self.d_a = []
        self.delta_weights = []
        self.delta_bias = []

        # Compute derivative of loss w.r.t. predicted output
        d_loss_pred = self.der_ls(y, y_pred)
        self.d_h.append(d_loss_pred)

        # Compute derivative of output activation for each sample
        der_outpt_mat = []
        for sample_idx in range(y_pred.shape[0]):
            d_loss_sample = self.der_ls(y[sample_idx], y_pred[sample_idx])
            d_out_act_sample = self.der_outpt_actvtn(y_pred[sample_idx])
            der_outpt_mat.append(np.matmul(d_loss_sample, d_out_act_sample))
        
        der_outpt_arr = np.array(der_outpt_mat)
        self.d_a.append(der_outpt_arr)

        # Backpropagation through hidden layers
        for layer_idx in range(self.ff_nn.hidden_layers, 0, -1):
            # Compute gradients for weights and biases at current layer
            grad_weight = np.matmul(self.ff_nn.H[layer_idx].T, self.d_a[-1])
            grad_bias = np.sum(self.d_a[-1], axis=0)

            # Append computed gradients
            self.delta_weights.append(grad_weight)
            self.delta_bias.append(grad_bias)

            # Compute gradients w.r.t. hidden layer outputs
            grad_hidden_output = np.matmul(self.d_a[-1], self.ff_nn.weights[layer_idx].T)
            grad_activation_input = grad_hidden_output * self.der_actvtn(self.ff_nn.H[layer_idx])

            # Store gradients for next iteration
            self.d_h.append(grad_hidden_output)
            self.d_a.append(grad_activation_input)

        # Compute gradients for weights connecting input layer to first hidden layer
        grad_weight_input_layer = np.matmul(self.ff_nn.H[0].T, self.d_a[-1])
        grad_bias_input_layer = np.sum(self.d_a[-1], axis=0)

        # Append and reverse gradients to match forward propagation sequence
        self.delta_weights.append(grad_weight_input_layer)
        self.delta_weights.reverse()

        self.delta_bias.append(grad_bias_input_layer)
        self.delta_bias.reverse()

        # Normalize gradients by number of samples
        num_samples = y.shape[0]
        for idx in range(len(self.delta_weights)):
            self.delta_weights[idx] /= num_samples
            self.delta_bias[idx] /= num_samples

        return self.delta_weights, self.delta_bias


class Optimizer:
    def __init__(self, ff_nn: FF_NN, bp_nn: BP_NN, param):
        self.ff_nn = ff_nn
        self.bp_nn = bp_nn
        self.lr = param['learning_rate']
        self.optimizer = param['optimizer']
        self.momentum = param['momentum']
        self.decay = param['decay']
        self.B1 = param['beta1']
        self.B2 = param['beta2']
        self.eps = param['epsilon']
        self.t = 0  # Time step for Adam and Nadam optimizers

        # Initialize histories for weights and biases
        self.w_history = [np.zeros_like(w) for w in self.ff_nn.weights]
        self.b_history = [np.zeros_like(b) for b in self.ff_nn.bias]
        self.w_hm = [np.zeros_like(w) for w in self.ff_nn.weights]
        self.b_hm = [np.zeros_like(b) for b in self.ff_nn.bias]

    def optimize(self, delta_weights, delta_bias):
        optimizer_type = self.optimizer.lower()

        if optimizer_type == "sgd":
            self.SGD(delta_weights, delta_bias)
        elif optimizer_type == "momentum":
            self.MGD(delta_weights, delta_bias)
        elif optimizer_type == "nesterov":
            self.NAG(delta_weights, delta_bias)
        elif optimizer_type == "rmsprop":
            self.RMSPROP(delta_weights, delta_bias)
        elif optimizer_type == "adam":
            self.ADAM(delta_weights, delta_bias)
        elif optimizer_type == "nadam":
            self.NADAM(delta_weights, delta_bias)

    def SGD(self, delta_weights, delta_bias):
        for layer in range(self.ff_nn.hidden_layers + 1):
            # Update weights and biases using SGD
            weight_update = delta_weights[layer] + self.decay * self.ff_nn.weights[layer]
            bias_update = delta_bias[layer] + self.decay * self.ff_nn.bias[layer]

            # Apply updates
            self.ff_nn.weights[layer] -= self.lr * weight_update
            self.ff_nn.bias[layer] -= self.lr * bias_update

    def MGD(self, delta_weights, delta_bias):
        for layer in range(self.ff_nn.hidden_layers + 1):
            # Update momentum history
            self.w_history[layer] = (self.momentum * self.w_history[layer]) + delta_weights[layer]
            self.b_history[layer] = (self.momentum * self.b_history[layer]) + delta_bias[layer]

            # Apply updates with momentum
            weight_update = self.w_history[layer] + self.decay * self.ff_nn.weights[layer]
            bias_update = self.b_history[layer] + self.decay * self.ff_nn.bias[layer]

            # Update weights and biases
            self.ff_nn.weights[layer] -= self.lr * weight_update
            self.ff_nn.bias[layer] -= self.lr * bias_update
    def NAG(self, delta_weights, delta_bias):
        # Apply Nesterov Accelerated Gradient optimization
        for i in range(self.ff_nn.hidden_layers + 1):
            # Update momentum history for weights and biases
            self.w_history[i] = self.momentum * self.w_history[i] + delta_weights[i]
            self.b_history[i] = self.momentum * self.b_history[i] + delta_bias[i]
            
            # Compute weight updates using Nesterov lookahead
            weight_update = self.momentum * self.w_history[i] + delta_weights[i] + self.decay * self.ff_nn.weights[i]
            bias_update = self.momentum * self.b_history[i] + delta_bias[i] + self.decay * self.ff_nn.bias[i]
            
            # Update weights and biases
            self.ff_nn.weights[i] -= self.lr * weight_update
            self.ff_nn.bias[i] -= self.lr * bias_update

    def RMSPROP(self, delta_weights, delta_bias):
        # Apply RMSProp optimization technique
        for i in range(self.ff_nn.hidden_layers + 1):
            # Update squared gradient history for weights and biases
            self.w_history[i] = self.momentum * self.w_history[i] + (1 - self.momentum) * np.square(delta_weights[i])
            self.b_history[i] = self.momentum * self.b_history[i] + (1 - self.momentum) * np.square(delta_bias[i])
            
            # Compute weight and bias updates using RMSProp formula
            weight_update_term = delta_weights[i] / (np.sqrt(self.w_history[i]) + self.eps)
            bias_update_term = delta_bias[i] / (np.sqrt(self.b_history[i]) + self.eps)
            
            # Apply decay to weights and biases
            decay_weight_term = self.decay * self.ff_nn.weights[i]
            decay_bias_term = self.decay * self.ff_nn.bias[i]
            
            # Update weights and biases
            self.ff_nn.weights[i] -= (self.lr * weight_update_term) + (self.lr * decay_weight_term)
            self.ff_nn.bias[i] -= (self.lr * bias_update_term) + (self.lr * decay_bias_term)

    def ADAM(self, delta_weights, delta_bias):
        # Apply Adam optimization technique
        for i in range(self.ff_nn.hidden_layers + 1):
            # Update moving averages of gradients and squared gradients for weights
            self.w_hm[i] = self.B1 * self.w_hm[i] + (1 - self.B1) * delta_weights[i]
            self.w_history[i] = self.B2 * self.w_history[i] + (1 - self.B2) * np.square(delta_weights[i])
    
            # Compute bias-corrected estimates for weights
            w_hm_corrected = self.w_hm[i] / (1 - self.B1 ** (self.t + 1))
            w_history_corrected = self.w_history[i] / (1 - self.B2 ** (self.t + 1))
    
            # Update weights using Adam formula
            weight_update = w_hm_corrected / (np.sqrt(w_history_corrected) + self.eps)
            self.ff_nn.weights[i] -= self.lr * (weight_update + self.decay * self.ff_nn.weights[i])
    
            # Update moving averages of gradients and squared gradients for biases
            self.b_hm[i] = self.B1 * self.b_hm[i] + (1 - self.B1) * delta_bias[i]
            self.b_history[i] = self.B2 * self.b_history[i] + (1 - self.B2) * np.square(delta_bias[i])
    
            # Compute bias-corrected estimates for biases
            b_hm_corrected = self.b_hm[i] / (1 - self.B1 ** (self.t + 1))
            b_history_corrected = self.b_history[i] / (1 - self.B2 ** (self.t + 1))
    
            # Update biases using Adam formula
            bias_update = b_hm_corrected / (np.sqrt(b_history_corrected) + self.eps)
            self.ff_nn.bias[i] -= self.lr * (bias_update + self.decay * self.ff_nn.bias[i])
    
    def NADAM(self, delta_weights, delta_bias):
        # Apply Nadam optimization technique
        for i in range(self.ff_nn.hidden_layers + 1):
            # Update moving averages of gradients for weights
            self.w_hm[i] = self.B1 * self.w_hm[i] + (1 - self.B1) * delta_weights[i]
            w_hm_corrected = self.w_hm[i] / (1 - self.B1 ** (self.t + 1))
    
            # Update moving averages of squared gradients for weights
            self.w_history[i] = self.B2 * self.w_history[i] + (1 - self.B2) * np.square(delta_weights[i])
            w_history_corrected = self.w_history[i] / (1 - self.B2 ** (self.t + 1))
    
            # Compute Nadam weight update term
            weight_temp = (self.B1 * w_hm_corrected) + ((1 - self.B1) / (1 - self.B1 ** (self.t + 1))) * delta_weights[i]
            weight_update = weight_temp / (np.sqrt(w_history_corrected) + self.eps)
            
            # Apply weight updates
            decay_weight_term = self.decay * self.ff_nn.weights[i]
            self.ff_nn.weights[i] -= self.lr * (weight_update + decay_weight_term)
    
            # Update moving averages of gradients for biases
            self.b_hm[i] = self.B1 * self.b_hm[i] + (1 - self.B1) * delta_bias[i]
            b_hm_corrected = self.b_hm[i] / (1 - self.B1 ** (self.t + 1))
    
            # Update moving averages of squared gradients for biases
            self.b_history[i] = self.B2 * self.b_history[i] + (1 - self.B2) * np.square(delta_bias[i])
            b_history_corrected = self.b_history[i] / (1 - self.B2 ** (self.t + 1))
    
            # Compute Nadam bias update term
            bias_temp = (self.B1 * b_hm_corrected) + ((1 - self.B1) / (1 - self.B1 ** (self.t + 1))) * delta_bias[i]
            bias_update = bias_temp / (np.sqrt(b_history_corrected) + self.eps)

            # Apply bias updates
            decay_bias_term = self.decay * self.ff_nn.bias[i]  # Corrected line
            self.ff_nn.bias[i] -= self.lr * (bias_update + decay_bias_term)

def plot_confusion_mat(y, y_pred):
      classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
      import pandas as pd
      import seaborn as sn
      mp = np.zeros((len(classes),len(classes)))
      for i,j in zip(y, y_pred):
        mp[np.argmax(i)][np.argmax(j)]+=1
    
      df_cm = pd.DataFrame(mp, [i for i in classes], [i for i in classes])
      plt.figure(figsize=(8,8))
      sn.set(font_scale=1) # for label size
      sn.heatmap(df_cm, annot=True, annot_kws={"size": 7}, cmap='inferno', fmt='g')
      plt.xlabel('Prediction')
      plt.ylabel('Actual')
    
      wandb.init(project="DA6401_Assignment_1")
      wandb.run.name = f'Q-7 Confusion Matrix final'
      wandb.log({"Confusion Matrix":wandb.Image(plt)})
      wandb.log({"Test Accuracy : ":np.sum(np.argmax(prediction, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]})
      wandb.finish()


def train():
    # Initialize Weights & Biases (WandB) for experiment tracking
    wandb.init()
    PARAMETERS = wandb.config

    # Set a unique name for the WandB run based on hyperparameters
    wandb.run.name = f"HL_{PARAMETERS['hidden_lyrs']}_BS_{PARAMETERS['batch_sz']}_AC_{PARAMETERS['activation']}_LF_{PARAMETERS['loss_function']}_OPT_{PARAMETERS['optimizer']}_WI_{PARAMETERS['weight_initialisation']}_Neurons_{PARAMETERS['neurons']}"

    # Load training data based on the specified dataset
    x_train, y_train = load_data(PARAMETERS['dataset'], 'train')

    # Set random seed for reproducibility thala for reason
    np.random.seed(7)

    # Initialize feedforward neural network, backpropagation, and optimizer
    ff_nn = FF_NN(PARAMETERS)
    bp_nn = BP_NN(ff_nn, PARAMETERS)
    opt = Optimizer(ff_nn, bp_nn, PARAMETERS)

    # Extract batch size from parameters
    batch_size = PARAMETERS['batch_sz']

    # Split training data into training and validation sets
    x_train, x_train_t, y_train, y_train_t = train_test_split(x_train, y_train, test_size=0.1, random_state=7)

    # Training loop for the specified number of epochs
    for epoch in range(PARAMETERS['epochs']):
        # Iterate over batches of data
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Perform backpropagation and optimization for the current batch
            opt.optimize(*bp_nn.propogate_backward(y_batch, ff_nn.feed_forward(x_batch)))

        # Increment time step for optimizers like Adam and Nadam
        opt.t += 1

        # Compute predictions for training and validation sets
        y_pred = ff_nn.feed_forward(x_train)
        y_pred_t = ff_nn.feed_forward(x_train_t)

        # Print metrics for the current epoch
        print(f"epoch-{epoch + 1}")
        print(f"accuracy-{np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]}")
        print(f"loss-{calculate_loss(y_train, y_pred, PARAMETERS['loss_function'])}")
        print(f"val_accuracy-{np.sum(np.argmax(y_pred_t, axis=1) == np.argmax(y_train_t, axis=1)) / y_train_t.shape[0]}")

        # Log metrics to WandB for visualization and tracking
        metrics = {
            'accuracy': np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0],
            'val_accuracy': np.sum(np.argmax(y_pred_t, axis=1) == np.argmax(y_train_t, axis=1)) / y_train_t.shape[0],
            'epoch': epoch + 1,
            'loss': calculate_loss(y_train, y_pred, PARAMETERS['loss_function']),
            'val_loss': calculate_loss(y_train_t, y_pred_t, PARAMETERS['loss_function'])
        }
        wandb.log(metrics)

    return ff_nn





if __name__ == "__main__": 
	inp = vars(parse_arguments())
	sweep_config = {
		"method": "grid",
		"metric": {"goal": "maximize", "name": "val_accuracy"},
		"parameters": {
			"inpt_sz": {"values": [784]},
			"oupt_sz": {"values": [10]},
			"oupt_activation": {"values": ["softmax"]},
			"dataset": {"values": [inp['dataset']]},
			"loss_function": {"values": [inp['loss']]},
			"beta": {"values": [inp['beta']]},
			"beta1": {"values": [inp['beta1']]},
			"beta2": {"values": [inp['beta2']]},
			"neurons": {"values": [inp['hidden_size']]},
			"hidden_lyrs": {"values": [inp['num_layers']]},
			"activation": {"values": [inp['activation']]},
			"learning_rate": {"values": [inp['learning_rate']]},
			"optimizer": {"values": [inp['optimizer']]},
			"momentum": {"values": [inp['momentum']]},
			"batch_sz": {"values": [inp['batch_size']]},
			"epochs": {"values": [inp['epochs']]},
			"weight_initialisation": {"values": [inp['weight_init']]},
			"decay": {"values": [inp['weight_decay']]},
			"epsilon": {"values": [inp['epsilon']]},
		}
	}

	sweep_id = wandb.sweep(sweep_config, project=inp['wandb_project'])
	wandb.agent(sweep_id, function=train)     


