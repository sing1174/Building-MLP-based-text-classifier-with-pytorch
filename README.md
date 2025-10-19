# MLP Based Text Classifier on IMDB dataset

Built a text classifier using a two-layer Multi-Layer Perceptron (MLP) in PyTorch. Selected a dataset from torchtext.datasets called IMDb Movie Reviews, and used it to train and evaluate the model. 
The core objective is to stack an additional intermediate layer on top of the single-layer MLP model we developed in class. The classifier will take textual inputs, which will be tokenized and passed through an embedding layer. The two-layer MLP will then process the embeddings by passing them through a hidden layer of 100 dimensions, followed by a final output layer that predicts the class label.

Throughout this assignment, I conducted multiple experiments to understand the performance of the two-layer MLP model. I compared its performance with a one-layer MLP model, analyzed how changing the dimensionality of the intermediate layer (from 100 to 200) affects test accuracy, and reviewed errors in test set predictions. My approach involved training the model, plotting accuracy curves for each epoch, and examining misclassified examples to identify areas where the model struggled.
