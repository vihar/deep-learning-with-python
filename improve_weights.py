import numpy as np
weights = np.array([0, 2, 1])
input_data = np.array([1, 2, 3])

learning_rate = 0.01

preds = (weights * input_data).sum()

target = 0
# Calculate the error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - (learning_rate * slope)

# Get updated predictions
preds_updated = (weights_updated * input_data).sum()

# Print
print(error)
print(weights_updated)
print(slope)
print(preds_updated)
