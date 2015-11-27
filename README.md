# NeuralModels
A library for Recurrent Neural Networks built on top of Theano. It allows quick prototyping of complicated structures of Recurrent Neural Networks. Checkout some of the existing architectures (and research papers) implemented using NeuralModels https://github.com/asheshjain399/RNNexp/

## INSTALL

python setup.py develop

In order to check NeuralModels is correctly installed, try the character-rnn example. 
``` 
python char-rnn.py 
```

## Description

neuralmodels: Python module containing definition of layers, optimization methods, and few models. 

## Models

NeuralModels comes with some pre-implemented models in the models directory

```
models/DRA.py
``` 
is the structural-RNN (S-RNN) code for doing deep learning on spatio-temporal graphs. The paper is present here http://www.cs.stanford.edu/people/ashesh/srnn See the repository https://github.com/asheshjain399/RNNexp/ for examples to use S-RNN

```
models/RNN.py
```
This is a simple RNN implementation.

In order to implement strucutures of RNN see examples such as models/SharedRNN.py
