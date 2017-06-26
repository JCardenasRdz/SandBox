from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

import numpy as np
from dataloader import *

np.set_printoptions(threshold=np.nan)

trainX,trainY,testX,testY = import_data()

# Just want to see what the data looks like
print(trainX[0])
print(trainY[0])
print("Shape of training data:", trainX.shape)
print("Shape of training labels:", trainY.shape)

# The code below literally does everything that the spam-ham
# TF program did.
num_features = trainX.shape[1]
num_labels = trainY.shape[1]

# Define a model with sequential layers (not related to RNN's or time series).
# Sequential just means that we're stacking layers of activations on top of
# each other.
model = Sequential()
# Add a layer to the model (this is a dense, e.g. fully-connected, layer).
model.add(Dense(num_labels, input_dim=num_features, activation='softmax'))

# Number of epochs we want to run through.
num_epochs = 3000

# Create an optimizer with special parameters (could have alternatively passed)
# the 'optimizer' parameter the string "sgd" for default learning parameters.
sgd = optimizers.SGD(lr=0.0008, decay=1e-6, momentum=0.9, nesterov=True)

# Categorical cross-entropy means that you're using one-hot encodings for the
# true class prediction. A metric is what's used to evaluate the model in the
# evaluation step.
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# The fitting procedure is pretty neat; it returns a 'History' type object
# that has info collected during training (error rates, accuracy, etc.)
history = model.fit(trainX, trainY, batch_size=128, nb_epoch=num_epochs,
	verbose=1, validation_split=0.3)

# Runs the trained model on the given set of test data, measuring the 
# metric we supplied in the .compile() method (accuracy, in our case).
score = model.evaluate(testX, testY, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])