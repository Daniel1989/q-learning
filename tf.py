# import tensorflow as tf
# # tf.config.set_visible_devices([], 'GPU')
#
# if __name__ == '__main__':
#     with tf.device('/cpu:0'):
#         print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np

grid = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
flattened_grid = grid.flatten()
input_data = np.reshape(flattened_grid, (1, 9))

print(input_data)  # Output: (1, 9)