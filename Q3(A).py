import numpy as np

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

