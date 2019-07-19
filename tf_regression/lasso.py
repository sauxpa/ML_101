#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Lasso():
    def __init__(self, X=None,
                 y=None,
                 optim_params=dict(),
                 l1_strength=1e-1,
                 n_epochs=int(1e2),
                 verbose=False):
        self._X = X
        self._y = y
        self._optim_params = optim_params
        self._l1_strength = l1_strength
        self._theta = None
        self._n_epochs = n_epochs
        self._verbose = verbose

    @property
    def X(self):
        """ Feature matrix : (n_data, n_features)
        """
        return self._X
    @X.setter
    def X(self, new_X):
        self._X = new_X

    @property
    def y(self):
        """ target vector : (n_data, 1)
        """
        return self._y
    @y.setter
    def y(self, new_y):
        self._y = new_y

    @property
    def optim_params(self):
        return self._optim_params
    @optim_params.setter
    def optim_params(self, new_optim_params):
        self._optim_params = new_optim_params

    @property
    def optimizer_name(self):
        return self.optim_params.get('optimizer_name', '')

    @property
    def learning_rate(self):
        return self.optim_params.get('learning_rate', None)

    @property
    def momentum(self):
        return self.optim_params.get('momentum', None)

    @property
    def l1_strength(self):
        return self._l1_strength
    @l1_strength.setter
    def l1_strength(self, new_l1_strength):
        self._l1_strength = float(new_l1_strength)

    @property
    def n_epochs(self):
        return self._n_epochs
    @n_epochs.setter
    def n_epochs(self, new_n_epochs):
        self._n_epochs = new_n_epochs

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, new_verbose):
        self._verbose = new_verbose

    @property
    def n_features(self):
        return self._X.shape[1]

    @property
    def n_samples(self):
        return self._X.shape[0]

    @property
    def optimizer(self):
        if self.optimizer_name == 'gradient_descent':
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.learning_rate,                                              momentum=self.momentum,                                              use_nesterov=False)
        elif self.optimizer_name == 'nesterov':
            return tf.train.MomentumOptimizer(learning_rate=self.learning_rate,                                              momentum=self.momentum,                                              use_nesterov=True)
        elif self.optimizer_name == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate=self.learning_rate)
        else:
            raise NameError('Optimizer {} not implemented'.format(self.optimizer_name))

    @property
    def l1_regularizer(self):
        return tf.contrib.layers.l1_regularizer(
           scale=self.l1_strength, scope=None
        )

    def n_active_features(self, threshold):
        return (self.theta>threshold).sum()

    def fit(self):
        X = tf.constant(self._X, dtype=tf.float32, name='X')
        y = tf.constant(self._y, dtype=tf.float32, name='y')
        theta = tf.Variable(tf.random_uniform([self.n_features, 1], -1.0, 1.0), name='theta')
        y_pred = tf.matmul(X, theta, name='predictions')
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name='mse')

        l1_regularizer = self.l1_regularizer
        optimizer = self.optimizer

        regularization_penalty = tf.contrib.layers.apply_regularization(
            l1_regularizer,
            [theta])

        regularized_loss = mse + regularization_penalty # this loss needs to be minimized

        training_op = optimizer.minimize(regularized_loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.n_epochs+1):
                if self.verbose and epoch % 100 == 0:
                    print('Epoch', epoch, 'MSE =', mse.eval())
                sess.run(training_op)
            self.theta = theta.eval().flatten()

    def bar_plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        index = range(self.n_features)
        ax.bar(index, self.theta)
        ax.set_xlabel(r'Component of $\theta$')
        ax.set_ylabel('Value')
        plt.tight_layout()
        plt.show()

    def lasso_path(self,
                   n_l1_strength=10,
                   eps=0.001,
                   plot_path=False,
                   plot_active_features_path=False,
                   threshold=1e-1):
        """Lasso regularization path
        l1_strength_grid covers [eps, 1] in a logarithmic scale,
        threshold determines which features are inactive (less than threshold away from zero)
        """
        l1_strength_grid = np.logspace(np.log(eps)/np.log(10), 0, n_l1_strength)
        path = []
        active_features_path = []
        for l1_strength in l1_strength_grid:
            self.l1_strength = l1_strength
            self.fit()
            path.append(self.theta)
            active_features_path.append(self.n_active_features(threshold=threshold))

        path = np.array(path).T

        if plot_path:
            fig, ax = plt.subplots(figsize=(17, 11), nrows=1, ncols=1)
            for i, theta in enumerate(path):
                ax.plot(l1_strength_grid, theta, label='Feature {}'.format(i+1))
                ax.scatter(l1_strength_grid, theta, marker='.')
            ax.set_xlabel('l1 strength')
            ax.set_ylabel(r'$\theta$ coefficients')
            ax.legend(loc='lower right')
            plt.tight_layout()
            plt.show()

        if plot_active_features_path:
            fig, ax = plt.subplots(figsize=(17, 11), nrows=1, ncols=1)
            ax.plot(l1_strength_grid, active_features_path)
            ax.set_xlabel('l1 strength')
            ax.set_ylabel('Number of active features')
            plt.tight_layout()
            plt.show()

        return path, active_features_path, l1_strength_grid
