import numpy as np

class Softmax_Regression:
    def __init__(self,lr=0.01,epochs=1000):
        self.w = None 
        self.b = None 
        self.num_classes = None
        self.lr = lr 
        self.epochs = epochs 
    
    def as_2d(self,x):
        x = np.asarray(x,dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1,1)
        return x 

    def softmax(self,z):
        z = z -np.max(z,axis=1,keepdims=True)
        return np.exp(z)/ np.sum(np.exp(z),axis=1,keepdims=True)
    
    def predict_proba(self,x):
        X = self.as_2d(x)
        logits = X@self.w+self.b 
        return self.softmax(logits)
    
    def predict(self,x):
        proba = self.predict_proba(x)
        return np.argmax(proba,axis=1)

    def fit(self,X,y,verbose=False):
        X = self.as_2d(X)
        y = np.asarray(y,dtype=int)
        N , D = X.shape
        self.num_classes = np.unique(y).size 
        self.w = np.zeros((D,self.num_classes))
        self.b = np.zeros((1,self.num_classes))
        y_onehot = np.zeros((N,self.num_classes))
        y_onehot[np.arange(N),y] = 1
        for epoch in range(self.epochs):
            logits = X@self.w + self.b 
            proba = self.softmax(logits)
            error = proba-y_onehot 
            grad_w = (X.T@error)/N
            grad_b = np.sum(error,axis=0,keepdims=True)/N
            self.w -= self.lr*grad_w
            self.b -= self.lr*grad_b
            if verbose and epoch % 100 == 0:
                loss = self.loss(proba,y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return self
    
    def loss(self,y_pred,y_true):
        y_true = np.asarray(y_true,dtype=int)
        N = y_true.size 
        log_likelihood = -np.log(y_pred[np.arange(N),y_true]+1e-15)
        return np.mean(log_likelihood)
    
    def fit_predict(self,X,y):
        self.fit(X,y)
        return self.predict(X)
    

    

