import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

n_updates = 20
mse_hist = []
learning_rate = 0.01
input_data = np.array([1, 2, 3])
weights = np.array([-0.49929916, 1.00140168, -0.49789747])
target = 0


def pred(input_data, target, weights):
    return ((input_data * weights).sum())


def get_slope(input_data, target, weights):
    preds = (weights * input_data).sum()
    error = preds - target
    return 2 * input_data * error


def get_mse(input_data, target, weights):
    preds = pred(input_data, target, weights)
    return mean_squared_error([preds], [target])


for i in range(n_updates):
    slope = get_slope(input_data, target, weights)

    weights = weights - (learning_rate * slope)

    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)

    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
