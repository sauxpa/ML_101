import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import truncnorm_plus, truncnorm_minus

class MeanFieldProbit(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 X,
                 y,
                 Xy_eval=(),
                 T=1000,
                 T_warmup=500,
                 tau=1e2,
                ):
        # data
        self.X = X
        self.y = y
        
        # evaluation set
        self.X_eval, self.y_eval = Xy_eval
        
        # number of iterations for the joint updates of q(beta) and q(z) 
        self.T = T
        
        # prior variance on beta
        self.tau = tau

        # posterior covariance matrix of beta
        # calculated here as it depends only on the data
        # (it's not updated during the sampling epochs).
        self.cov_beta = np.linalg.inv(
            1/self.tau * np.eye(self.p) + np.dot(self.X.T, self.X)
        )
        
    @property
    def p(self):
        """Number of predictors.
        """
        return self.X.shape[1]
    
    @property
    def n(self):
        """Number of data points.
        """
        return self.X.shape[0]
    
    def get_mu_beta(self):
        """Get the mean parameter of q(beta).
        """
        return np.dot(self.cov_beta, np.dot(self.X.T, self.expectation_z()))
        
    def get_mu_z(self):
        """Get the mean parameter of q(z).
        Note that q(z) is a truncated Gaussian,
        therefore its expectation is not the mean parameter
        of the Gaussian (only the latter is calculated here).
        """
        return np.dot(self.X, self.mu_beta)
    
    def expectation_z(self):
        """Compute the expectation of q(z) which is a truncated Gaussian.
        """
        phi = scipy.stats.norm.pdf(-self.mu_z)
        Phi = scipy.stats.norm.cdf(-self.mu_z)
        exp_z = [
            self.mu_z[i] + phi[i]/(1-Phi[i]) if self.y[i] == 1 else self.mu_z[i] - phi[i]/Phi[i] for i in range(self.n)
        ]
        return np.array(exp_z)
        
    def reset(self):
        self.mu_beta = np.zeros(self.p)
        self.mu_z = np.zeros(self.n) 
        self.acc = np.empty(self.T)
        
    def fit(self):
        """Gibbs sampling (with cyclic scan).
        """
        self.reset()
        
        postfix = {'eval_accuracy': 0.0}
        
        pbar = tqdm(range(self.T), postfix=postfix)
        for t in pbar:
            # q(z) = N(mu_z, I)
            # with mu_z = X*E_{q(beta)}[beta] = X*mu_beta
            self.mu_z = self.get_mu_z()        
            
            # q(beta) = N(mu_beta, cov_beta)
            # with cov_beta = (tau*I + X^T*X)^-1 
            # and  mu_beta = cov_beta*X^T*E_{q(z)}[z]
            self.mu_beta = self.get_mu_beta()
            
            # Eval accuracy using mean a posteriori on beta.
            y_pred = self.predict(self.X_eval)
            acc = accuracy_score(y_pred, self.y_eval)
            self.acc[t] = acc
            postfix['eval_accuracy'] = acc
            pbar.set_postfix(postfix)
            
        # Return the classifier
        return self
    
    def predict(self, X_test):
        """Predict label using mean a posterio on beta.
        """
        return np.sign(np.dot(X_test, self.mu_beta))