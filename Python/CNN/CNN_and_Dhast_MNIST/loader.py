
def load_mnist_data( normalize = 1, squeeze = 0):
    import numpy as np
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras import backend as K

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    if normalize == 1:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train =  x_train / 255
        x_test  =  x_test / 255
    else:
        print('Data is NOT normalized')
        print(40*'=')

    if squeeze == 1:
        x_train = np.squeeze( x_train )
        y_train = np.squeeze( y_train )
        x_test = np.squeeze( x_test )
        y_test = np.squeeze( y_test )

    return x_train, y_train, x_test, y_test
