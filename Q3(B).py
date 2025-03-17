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
