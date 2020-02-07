# ML_101

101 machine learning scripts:

## RL 

Elementary gridworld environments and standard, exact RL algo to solve the underlying MDP (SARSA, Q-Learning, with or without experience replay).

## circle_classificaiton

Showcase of scikit-learn classifier on the circle dataset.

## graphs

Demo scripts for graphs in ML. Graph neural networks using dgl and torch.

## lasso 

Tensorflow implementation of Lasso regression, with plots to track the sparsity of regularized predictors.

## probabilistic_ml_101

Examples of classification (Linear, Logistic, LDA, QDA), clustering (KMeans, EM), and approximate inference (Gibbs sampling, mean field variational inference) implemented from scratch using Numpy. Inspired from Bishop's [**Pattern Recognition and Machine Learning**](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).

## Misc

* Discrete Random Number Generator: comparison of numpy.rd.choice vs manually implemented generators (using only numpy.rd.uniform).

* Ensemble learning 101: again, showcase of scikit-learn classifiers and a few aggregation methods (voting, boosting, bagging).

* Labeled Faces In The Wild : again. showcase of scikit-learn classifiers on face recognition (no deep learning involved).

* Temporal regularization in MDP : study of an article (https://arxiv.org/pdf/1811.00429.pdf) about solving MDP by optimizing simultaneously the orignal and time-reversed chains. This leads to a biased yet typically lower variance solution.

* double_sloped_curve : following the bias-variance tradeoff, generalization error typically follows a U-curve, with increasing errors due to overfitting when the complexity of the model is too high. However it has been shown, first in deep convolution networks setting, that the generalization error actually decreases again after reaching a peak when the model becomes largely overparametrized. This is an attempt to exhibit this doulbe-sloped curve in a simpler regression setting. 

* lazy_training_nn : investigate generalization power of lazy training (training only a subset of available layers, considering the rest as randomly generated features) on a high-dimensional regression problem.

* reservoir_sampling : toy example of a sampling technique from an array of unknown size.
