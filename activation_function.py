"""
For neural Network to acheive their maximum predictive power
we need to apply an activation function for the hidden layers.
It is used to capture the non linearities. We apply them to the
input layers, hidden layers with some equation on the values.
"""


import numpy as np

print("Enter the two values for input layers")

print('a = ')
a = int(input())
# 2
print('b = ')
b = int(input())

weights = {
    'node_0': np.array([2, 4]),
    'node_1': np.array([[4, -5]]),
    'output_node': np.array([2, 7])
}

input_data = np.array([a, b])


def relu(input):
    # Rectified Linear Activation
    output = max(input, 0)
    return(output)


node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

hidden_layer_outputs = np.array([node_0_output, node_1_output])

model_output = (hidden_layer_outputs * weights['output_node']).sum()

print(model_output)
