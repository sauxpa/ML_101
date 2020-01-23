import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import truncnorm_plus, truncnorm_minus

class GibbsSamplerProbit(BaseEstimator, ClassifierMixin):
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
        
        # mixing time 
        self.T = T
        # assume the Markov chain has reached the invariant state after T_warmup
        self.T_warmup = T_warmup
        
        # prior variance on beta
        self.tau = tau
        
        # posterior covariance matrix of beta
        # calculated here as it depends only on the data
        # (it's not updated during the sampling epochs).
        self.beta_posterior_cov = np.linalg.inv(
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

    def beta_posterior_mean(self, z):
        """Mean of p(beta|y,z)
        """
        return np.dot(self.beta_posterior_cov, np.dot(self.X.T, z))
    
    def sample_beta_posterior(self, z):
        """Sample from the Gaussian distribution p(beta|z).
        """
        return np.random.multivariate_normal(
            self.beta_posterior_mean(z),
            self.beta_posterior_cov,
        )
    
    def sample_z_conditional(self, beta):
        """Sample from p(z_i|beta, y_i) for i=1,...,n.
        """
        mu = np.dot(self.X, beta)
        z = [
            truncnorm_plus(mu[i]) if self.y[i] == 1 else truncnorm_minus(mu[i]) for i in range(self.n)
        ]
        return np.array(z)
        
    def reset(self):
        self.beta = np.zeros((self.T+1, self.p))
        self.z = np.empty((self.T+1, self.n)) 
        self.z[0] = np.random.randn(self.n)
        self.acc = np.empty(self.T)
        
    def fit(self):
        """Gibbs sampling (with cyclic scan).
        """
        self.reset()
        
        postfix = {'eval_accuracy': 0.0}
        
        pbar = tqdm(range(self.T), postfix=postfix)
        for t in pbar:
            # Gibbs updates
            self.beta[t+1] = self.sample_beta_posterior(self.z[t])
            self.z[t+1] = self.sample_z_conditional(self.beta[t+1])
            
            # Eval accuracy using mean a posteriori on beta.
            beta_MAP = np.mean(self.beta[:t+1], axis=0)
            y_pred = np.sign(np.dot(self.X_eval, beta_MAP))
            acc = accuracy_score(y_pred, self.y_eval)
            self.acc[t] = acc
            postfix['eval_accuracy'] = acc
            pbar.set_postfix(postfix)
            
        self.beta = self.beta[self.T_warmup:]
        self.z = self.z[self.T_warmup:]
        
        # Return the classifier
        return self
    
    def predict(self, X_test):
        """Predict label using mean a posterio on beta.
        """
        beta_MAP = np.mean(self.beta, axis=0)
        return np.sign(np.dot(X_test, beta_MAP))