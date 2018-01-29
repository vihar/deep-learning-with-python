import numpy as np

print("Enter the two values for input layers")

print('a = ')
a = int(input())
# 2
print('b = ')
b = int(input())
# 3

input_data = np.array([a, b])

weights = {
    'node_0': np.array([1, 1]),
    'node_1': np.array([-1, 1]),
    'output_node': np.array([2, -1])
}

node_0_value = (input_data * weights['node_0']).sum()
# 2 * 1 +3 * 1 = 5
print('node 0_hidden: {}'.format(node_0_value))

node_1_value = (input_data * weights['node_1']).sum()
# 2 * -1 + 3 * 1 = 1
print('node_1_hidden: {}'.format(node_1_value))

hidden_layer_values = np.array([node_0_value, node_1_value])

output_layer = (hidden_layer_values * weights['output_node']).sum()

print("output layer : {}".format(output_layer))
