import numpy as np

# Forward Propagation code
input_data = np.array([1, 1])
weights = {
  'node_00': np.array([2, 4]),
  'node_01': np.array([4, -5]),
  'node_10': np.array([0, 1]),
  'node_11': np.array([1, 1]),
  'output': np.array([5, 1])
}

def iden(value):
    return max(-1, value)

# first hidden layer
node_00_input = (input_data * weights["node_00"]).sum()
node_00_output = iden(node_00_input)


node_01_input = (input_data * weights['node_01']).sum()
node_01_output = iden(node_01_input)

# second hidden layer
node_layer_1 = np.array([node_00_output, node_01_output])

node_10_input = (node_layer_1 * weights["node_10"]).sum()
node_10_output = iden(node_10_input)


node_11_input = (node_layer_1 * weights['node_11']).sum()
node_11_output = iden(node_11_input)

#output
node_layer_2 = np.array([node_10_output, node_11_output])
output = (node_layer_2 * weights['output']).sum()
print(output)