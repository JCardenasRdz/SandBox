# keras-tutorial
Keras Tutorial for Tucson Data Science Meetup

Relevant Keras Documentation [(Sequential Models)](https://keras.io/models/sequential/)
----------------------------

  * [The Model class](https://keras.io/models/model/)
  * [Commonly used layers](https://keras.io/layers/core/)
  * [Optimizers](https://keras.io/optimizers/)
  * [Saving a model](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)
  * [Regularization](https://keras.io/regularizers/)

A sequential model is just a network that consists of a series of layers ([input]->[hidden #1]->...->[output]), the word *sequential* has nothing to do with RNN's.

What's cool about sequential models?
  * No need to declare weights or biases
  * Only need to declare number of nodes in the layer and the activation type
  * All of the most commonly used layer types (fully-connected, convolutional, recurrent, etc.) are just a callback away
  * You can save a network similarly to the TensorFlow method
  * Regularization is *so* easy
  * Much easier to implement than in TensorFlow

[Functional API Model (we'll talk about this if we have time)](https://keras.io/getting-started/functional-api-guide/)
--------------------------------------------------------------

  * [Example code](https://keras.io/getting-started/functional-api-guide/#more-examples)
  
The functional API is similar to what you'd find in TensorFlow; you define a sequence of operations, then add a loss function and optimization at the end of the model. Operations are called through the Keras API, but they are applied programmatically (e.g. through variables). Contrast this with the sequential models which *don't* require you to assign operations to variables.

