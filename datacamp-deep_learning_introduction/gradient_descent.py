import numpy as np

weights = np.array([1, 2])
input_data = np.array([3, 4])

target = 6
learning_rate = .01

preds = (weights * input_data).sum()
error = preds - target
print(error)

gradient = 2 * input_data * error
print("Gradient:", gradient)

weights_updated = weights - learning_rate * gradient
print("Weight Updated:", weights_updated)

preds_update = (weights_updated * input_data).sum()
error_updated = preds_update - target
print(error_updated)