import numpy as np
from scipy.optimize import minimize

# utils to build a custom sklearn classifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class LinearClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.w = None
        self.b = None
        self.name = 'LinearClassifier'
        self.threshold_ = 0.0

    @property
    def threshold(self):
        return self.threshold_

    @threshold.setter
    def threshold(self, new_threshold):
        self.threshold_ = new_threshold

    def get_params(self, deep=True):
        return {
                'w': self.w,
                'b': self.b,
                }

    def data_dim(self, X):
        return X.shape[1]

    def add_intercept(self, X):
        return np.append(X, np.ones((X.shape[0],1)), axis=1)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        X_intercept_ = self.add_intercept(X)
        self.X_intercept_ = X_intercept_
        self.y_ = y

        self.threshold = 0.0

        # theta = [w, b]
        # solve the normmal equation to derive theta analytically
        theta = np.linalg.solve(np.matmul(X_intercept_.T, X_intercept_), np.matmul(X_intercept_.T, y))
        self.w = theta[:self.data_dim(self.X_)]
        self.b = theta[-1]

        # Return the classifier
        return self

    def decision_boundary(self, xx):
        """
        For a grid xx evaluate the line equation for p(y=1|x)=0.5 in 2D.
        """
        return (0.5-self.b - self.w[0]*xx)/self.w[1]
    
    def decision_function(self, X):
        """
        Attribution to class 1 or 0 can be decided based on the sign of this quantity
        """
        return np.matmul(X, self.w)+self.b

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['w', 'b'])

        # Input validation
        X = check_array(X)

        D = self.decision_function(X)
        # by convention we set threshold to be around 0,
        # in the case of linear regression with 0-1 labels,
        # the meaningful threshold is 0.5 so shift it.
        return self.classes_[[0 if d<self.threshold+0.5 else 1 for d in D]]
