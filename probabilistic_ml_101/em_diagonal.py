import numpy as np

# utils to build a custom sklearn clustering algorithm
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array

class EMDiagonal(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=50):
        self.pi = None
        self.mu = None
        self.D = None
        self.name = 'EMDiagonalCovariance'
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    
    def get_params(self, deep=True):
        return {
                'pi': self.pi,
                'mu': self.mu,
                'D': self.D
                }
    
    
    def reset(self, X):
        """Initialized parameters.
        """
        # initial uniform distribution for the latent probability
        self.pi = np.ones(self.n_clusters)/self.n_clusters
        
        # initialize centroids to randomly picked data points 
        self.mu = np.random.permutation(X)[:self.n_clusters].T
    
        # unit variance init
        self.D = np.ones((X.shape[1], self.n_clusters))
    
    
    def cov_matrix(self, k):
        """Returns the diagonal covariance matrix for cluster k.
        """
        return np.diag(self.D[:, k])
    
    
    def gaussian_pdf_k(self, k):
        """Returns a lambda for the Gaussian density of x conditional to the
        being in the k-th cluster.
        """
        n_features = self.D.shape[0]
        
        # determinant of a diagonal matrix is simply the product of the diagonal elements
        det_k = np.sqrt(np.prod(self.D[:, k]))
        D_inv_k = np.diag(1/self.D[:, k])
        mu_k = self.mu[:, k]
        
        return lambda x: 1/((2*np.pi)**(n_features/2) * det_k) \
    * np.exp(-0.5*np.dot(x-mu_k, np.dot(D_inv_k, x-mu_k)))
        
    
    def expectation(self, X):
        """E step : calculate p(z=k|x_i).
        """
        pzk = np.array([[self.pi[k]*self.gaussian_pdf_k(k)(x) for k in range(self.n_clusters)] for x in X])
        pzk /= np.sum(pzk, axis=1).reshape(-1, 1)        
        return pzk
    
    
    def fit(self, X, y=None):
        """y: as per sklearn doc, for API compatibility only, not actually used.
        """
        
        # (n_samples, n_features)
        X = check_array(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        self.reset(X)
        
        for t in range(self.max_iter):
            pzk = self.expectation(X)
            # sum p(z|x_i) over i
            pzk_sum = np.sum(pzk, axis=0)
            self.pi = pzk_sum / n_samples
            self.mu = np.dot(X.T, pzk) / pzk_sum
            
            v = np.repeat(np.expand_dims(X.T, axis=2), self.n_clusters, axis=2)-np.repeat(np.expand_dims(self.mu, axis=1), n_samples, axis=1)
            self.D = np.sum(np.repeat(np.expand_dims(pzk, axis=0), n_features, axis=0) * (v*v), axis=1) / pzk_sum
            
        return self
    
    def predict(self, X):
        return np.argmax(self.expectation(X), axis=1)
    
    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.predict(X)