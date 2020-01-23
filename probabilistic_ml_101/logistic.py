import numpy as np
from scipy.optimize import minimize

# utils to build a custom sklearn classifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class LogisticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.w = None
        self.b = None
        self.name = 'LogisticClassifier'
        self.threshold = 0.0

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

    @property
    def y_scaled(self):
        """
        Remap the labels to be +1 or -1
        """
        return self.y_*2-1

    def add_intercept(self, X):
        return np.append(X, np.ones((X.shape[0],1)), axis=1)

    def objective_func(self, theta):
        """
        negative log-likelihood to minimize
        """
        k = np.multiply(self.y_scaled, (np.matmul(self.X_intercept_, theta)))
        return -np.sum(np.log(1/(1+np.exp(-k))))

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.X_intercept_ = self.add_intercept(X)
        self.y_ = y

        self.threshold = 0.0

        # theta = [w, b]
        theta_init = np.random.uniform(0, 1, self.data_dim(self.X_intercept_))

        fit_info = minimize(self.objective_func,
                            theta_init,
                            method='L-BFGS-B',
                           )

        if fit_info.success:
            theta = fit_info.x
            self.w = theta[:self.data_dim(self.X_)]
            self.b = theta[-1]
        else:
            raise Exception('Failed calibration...')

        # Return the classifier
        return self

    def decision_boundary(self, xx):
        """
        For a grid xx evaluate the line equation for p(y=1|x)=0.5 in 2D.
        """
        return (-self.b - self.w[0]*xx)/self.w[1]
    
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
        return self.classes_[[0 if d<self.threshold else 1 for d in D]]
