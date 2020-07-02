import numpy as np


output_file = 'imagenet_validation_sizes_110_10.npy'
x = np.load(output_file)

print(x[0])

print(x[50000-1])