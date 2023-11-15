import numba, numpy as np
from numba import *
from numba.experimental import jitclass
from functools import partial
from typing import *

spec = [
    ('model', double[:, :, :]),
    ('heights', int32[:])
]
 
class Model:
    def __init__(self, model, heights, hidden_function="tanh", output_function="softmax", cost_function="cross_entropy"):
        self.model = model
        self.heights = heights

        self.hidden_layer_activation_function = getattr(self, "_"+hidden_function)
        self.output_layer_activation_function = getattr(self, "_"+output_function)
        self.cost_function = getattr(self, "_"+cost_function)
        
    def _sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)

        return ( 1 / ( 1 + np.exp(-x) ) )

    def _tanh(self, x, deriv=False):
        if deriv:
            return (1 - x ** 2)

        return np.tanh(x)

    def _relu(self, x, deriv=False):

        negative_slope = 10 ** -9

        if deriv:
            return 1 * (x > 0) + (negative_slope * (x < 0))

        return 1 * x * (x > 0) + (negative_slope * x * (x < 0))

    def _softmax(self, x, deriv=False):

        if deriv:
            softmax_output = np.exp(x) / np.sum(np.exp(x))
            return softmax_output * (1 - softmax_output)

        e_x = np.exp(x)

        if np.isnan(e_x / e_x.sum()).any():
            print(e_x, x)

        return e_x / e_x.sum()
            
        
    def _cross_entropy(self, outputs, expected_outputs, deriv=False):
        if deriv:
            return (outputs - expected_outputs) / outputs.shape[0]

        epsilon = 1e-12

        outputs = np.clip(outputs, epsilon, 1.0 - epsilon)

        return -(expected_outputs * np.log(outputs + epsilon))
        
    def _mse(self, outputs, expected_outputs, deriv=False):
        if deriv:
            return 2 * (outputs - expected_outputs)
            
        return (outputs - expected_outputs) ** 2

    def gradient(self, output_activations, expected_output, weight_decay = 0):
        model = self.model
        heights = self.heights
        length = model.shape[0]
        height = model.shape[1]
        activations = output_activations
        inputs = len(output_activations[0])

        old_node_values = np.zeros(height)
        gradient = np.zeros(model.shape)

        output = output_activations[-1]
        cost = self.cost_function(output, expected_output)

        average_cost = cost.mean()

        for count, layer in enumerate(activations[::-1]):

            if count == length:
                break

            index = -(count + 1)

            num_inputs = heights[index - 1]
            height = heights[index]

            old_height = heights[index + 1]
            old_weights = model[index + 1, :old_height, :height]

            weights = model[index, :height, :num_inputs]
            biases = model[index, :height, num_inputs]
            
            input_layer = activations[index - 1][:num_inputs]
            output = layer[:height]


            if not count:
                cost_derivatives = self.cost_function(output, expected_output, deriv=True)

                activation_derivatives = self.output_layer_activation_function(output, deriv=True)

                node_values = cost_derivatives * activation_derivatives

            else:
                activation_derivatives = self.hidden_layer_activation_function(output, deriv=True)

                node_values = activation_derivatives * np.dot(old_weights.T, old_node_values)

            w_decay = (2 * weight_decay * weights)
            b_decay = (2 * weight_decay * biases)

            
            weights_derivative = np.array([node_value * input_layer for node_value in node_values])
            bias_derivative = 1 * node_values

            # print("Input: ", input_layer.tolist()) 
            # print("Output: ", output.tolist()) 
            # print("Expected Output: ", expected_output) 
            # print("Node Values", node_values) 
            # print("Activation Derivatives", activation_derivatives)
            # print("Cost Derivatives", cost_derivatives)
            # print("Weights: ", weights)
            # print("Old Weights", old_weights)

            # print("Weights Gradient: ", weights_derivative)
            # print("Bias Gradient: ", bias_derivative)

            # print("\n\n")

            old_node_values = node_values
            
            gradient[index, :height, :num_inputs] = weights_derivative
            gradient[index, :height, num_inputs] = bias_derivative

        return gradient, average_cost

    def eval(self, input, dropout_rate = 0, training=False):
        model = self.model
        heights = self.heights
        length = model.shape[0]
        default_height = model.shape[1]

        input_activations = input
        layer_outputs = [input_activations]        

        for idx, (height, layer) in enumerate(zip(heights[1:], model)):

            layer = layer[:height]


            num_inputs = len(input_activations)
            output = np.sum(layer[:height, :num_inputs] * input_activations, axis=1) + layer[:, num_inputs]

            # Node Activation
            if idx + 1 == length:
                output_activations = self.output_layer_activation_function(output)

            else:
                output_activations = self.hidden_layer_activation_function(output)

                # Droupout
                if training and dropout_rate:
                    mask = (np.random.rand(*output_activations.shape) > dropout_rate) / ( 1 - dropout_rate)
                    output_activations *= mask

            input_activations = output_activations
            layer_outputs.append(output_activations)

        return layer_outputs
