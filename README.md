# ECE5268 ICP
Theory of Neural Networks Project

This project provides a systematic approach to exploring and comparing different MLP architectures to find the combination that produces the best performance on the MNIST classification task.

The provided code implements a multilayer perceptron (MLP) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The MLP architecture consists of one or more hidden layers, followed by an output layer. The number of hidden layers and the number of nodes in each layer are hyperparameters that can be varied to explore different model configurations. The code utilizes a grid search approach to iterate over different combinations of hyperparameters. For each combination of hyperparameters, the model is trained using the Adam optimizer, Stochasitic Gradient Descent and the sparse categorical cross-entropy loss function. The dataset is normalized to scale pixel values between 0 and 1. After training, the code plots the training and validation losses for each combination of hyperparameters and for different numbers of epochs. This allows for visualizing how the loss changes over epochs and varies with different model configurations. A Confusion Matrix was developed to visualize the performance of the best model. 

This project was developed by Ceana Palaico. For troubleshooting, ChatGPT was utilized. 
