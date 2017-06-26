import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import mnist
from keras import optimizers

def cnn(batch_size = 128, epochs = 1):
    '''
    Trains a simple convnet on the MNIST dataset.
    '''
    num_classes = 10

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

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split= 0.10)
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model


def fit_two_layer_NN(trainX, testX, trainY, testY, num_epochs = 2, verbose_out = 1):
    '''
    fit_two_layer_NN(trainX, testX, trainY, testY, num_epochs = 2, verbose_out = 1):
    '''

    # convert class vectors to binary class matrices
    num_classes = 10
    trainY = keras.utils.to_categorical(trainY, num_classes)
    testY = keras.utils.to_categorical(testY, num_classes)

    # dimensions
    num_features = trainX.shape[1]
    num_labels = trainY.shape[1]

    # create model
    model = Sequential()
    model.add( Dense( units = int(num_features), input_dim = num_features, activation = 'relu'))
    model.add( Dense( units = int(num_features/10), input_dim = num_features, activation = 'relu'))
    model.add( Dense( num_labels, activation = 'softmax'))

    #summary
    print(model.summary())
    # define optimizer
    sgd = optimizers.SGD(lr = 0.0008, decay = 1e-6, momentum=0.9, nesterov=True)

    # compile
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # run
    history = model.fit(trainX, trainY, batch_size = 128,
                        epochs = num_epochs, verbose = verbose_out,
                                            validation_split= 0.33)
    # evaluate
    score = model.evaluate(testX, testY, verbose = 5)
    print('Test score:',score[0])
    print('Test accurracy:',score[1])

    return model
