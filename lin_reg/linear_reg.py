import numpy as np 

class LinearRegressionClosedForm:
    def __init__(self):
        self.w = None
        self.b = None 
    
    def as_2d(self,x):
        x = np.asarray(x,dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1,1)
        return x
    
    def fit(self,X,y):
        X = self.as_2d(X)
        y = self.as_2d(y)
        N, D = X.shape

        X_aug = np.hstack([X,np.ones((N,1))])
        theta =  np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y 
        self.w = theta[:-1]
        self.b = theta[-1]
    
    def loss(self,y_pred,y_true):
        return np.mean((y_pred-y_true)**2)
    
    def predict(self,X):
        X = self.as_2d(X)
        return X @ self.w + self.b
    
    def fit_predict(self,X,y):
        self.fit(X,y)
        return self.predict(X)
