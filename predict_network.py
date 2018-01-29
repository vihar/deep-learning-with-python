import numpy as np

input_data = [np.array([3, 5]), np.array(
    [1, -1]), np.array([0, 0]), np.array([8, 4])]

weights = {
    'node_0': np.array([2, 4]),
    'node_1': np.array([[4, -5]]),
    'output_node': np.array([2, 7])
}


def relu(input):
    # Rectified Linear Activation
    output = max(input, 0)
    return(output)


def predict_with_network(input_data_row, weights):
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    input_to_final_layer = (hidden_layer_outputs *
                            weights['output_node']).sum()
    model_output = relu(input_to_final_layer)
    return(model_output)


results = []
for input_data_row in input_data:
    prediction = predict_with_network(input_data_row, weights)
    results.append(prediction)
print(results)
