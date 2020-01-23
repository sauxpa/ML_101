import numpy as np

# utils to build a custom sklearn classifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# to get multivariate Gaussian density
from scipy.stats import multivariate_normal

class LDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.pi = None
        self.mu_1 = None
        self.mu_0 = None
        self.cov = None
        self.name = 'LDAClassifier'
        self.threshold_ = 0.0

    @property
    def threshold(self):
        return self.threshold_

    @threshold.setter
    def threshold(self, new_threshold):
        self.threshold_ = new_threshold

    def get_params(self, deep=True):
        return {
                'pi': self.pi,
                'mu_1': self.mu_1,
                'mu_0': self.mu_0,
                'cov': self.cov
                }

    def data_dim(self, X):
        return X.shape[1]

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.threshold = 0.0

        n = len(self.y_)
        count_1 = list(self.y_).count(1)
        count_0 = list(self.y_).count(0)

        # check there is no other class than 0 and 1 and that they are all accounted for
        assert count_0 + count_1 == n

        self.pi = count_1 / n

        self.mu_1 = np.sum(np.multiply(self.X_, np.array([self.y_, self.y_]).T), axis=0) / count_1
        self.mu_0 = np.sum(np.multiply(self.X_, np.array([1-self.y_, 1-self.y_]).T), axis=0) / count_0

        X_centered_1 = self.X_ - self.mu_1
        X_centered_0 = self.X_ - self.mu_0

        p = self.data_dim(X)

        self.cov = (np.sum([np.matmul(xj.reshape(p, 1), xj.reshape(1, p)) for xj, yj in zip(X_centered_0, self.y_) if yj == 0], axis=0)
                    + np.sum([np.matmul(xj.reshape(p, 1), xj.reshape(1, p)) for xj, yj in zip(X_centered_1, self.y_) if yj == 1], axis=0))/n

        # Return the classifier
        return self

    def decision_boundary(self, xx):
        """
        For a grid xx evaluate the line equation for p(y=1|x)=0.5 in 2D.
        """
        w = np.linalg.solve(self.cov, self.mu_1-self.mu_0)
        a = np.linalg.solve(self.cov, self.mu_1)
        b = np.linalg.solve(self.cov, self.mu_0)
        l = 0.5*(np.dot(a, self.mu_1) - np.dot(b, self.mu_0))
    
        return (l - w[0]*xx)/w[1]
    
    def decision_function(self, X):
        """
        p(y=1|x) - p(y=0|x)
        Attribution to class 1 or 0 can be decided based on the sign of this quantity
        """
        return self.pi * multivariate_normal.pdf(X, mean=self.mu_1, cov=self.cov) - (1-self.pi) * multivariate_normal.pdf(X, mean=self.mu_0, cov=self.cov)

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['pi', 'mu_1', 'mu_0', 'cov'])

        # Input validation
        X = check_array(X)

        D = self.decision_function(X)
        return self.classes_[[0 if d<self.threshold else 1 for d in D]]
