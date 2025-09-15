import numpy as np 

class Logic_Regression:
    def __init__(self,le=1e-4,epochs=100,seed=0):
        self.w = None
        self.b = None 
        self.rng = np.random.default_rng(seed)
        self.le = le
        self.epochs= epochs

    def as_2d(self,x):
        X = np.asarray(x,dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1,1)
        return X
    
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))
    
    def fit(self,X,y,verbose=False):
        X = self.as_2d(X)
        Y = self.as_2d(y)
        N, D = X.shape
        for epoch in range(self.epochs):
            if epoch == 0:
                self.w = np.zeros((D,1))
                self.b = 0.0
            if self.w is None:
                self.w = np.zeros((D,1))
            if self.b is None:
                self.b = 0.0
            y_pred = self.sigmoid(X@self.w+self.b)
            dw = (1.0/N) * (X.T@(y_pred-Y))
            db = (1.0/N) * np.sum(y_pred-Y)
            self.w -= self.le*dw
            self.b -= self.le*db 
            if verbose and epoch % 100 == 0:
                loss = self.loss(y_pred,Y)
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")
        return self

    def predict_proba(self,X):
        X = self.as_2d(X)
        z = X @ self.w + self.b 
        return self.sigmoid(z)
    
    def loss(self,y_pred,y_true):
        y_pred = np.asarray(y_pred,dtype=float).reshape(-1,1)
        y_true = np.asarray(y_true,dtype=float).reshape(-1,1)
        return -np.mean(y_true*np.log(y_pred)+ (1-y_true)*np.log(1-y_pred))
    
    def predict(self,X):
        predict = self.predict_proba(X)
        return (predict >= 0.5).astype(int)
    
    def fit_predict(self,X,y):
        self.fit(X,y)
        return self.predict(X)
    

