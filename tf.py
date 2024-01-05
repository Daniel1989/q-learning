import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
