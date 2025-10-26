import numpy as np 

class LinearRegressionClosedForm:
    def __init__(self,method='closed_form', weights=None, bias = 0.0 , learning_rate= 1e-3,random_state=42,epochs=1000):
        np.random.seed(random_state)
        self.weights=  weights 
        self.method = method 
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def show_available_methods(self):
        return ['closed_form','gradient_descent']

    def is_2D(self,X:np.array)-> np.array:
        N,D = X.shape
        if D == 1:
            return X.reshape(N,-1)
        return X
    
    def loss(self,y_true:np.array,y_predict:np.array)-> float:
        N = len(y_true)
        return (1/(2*N))* np.sum((y_true-y_predict)**2)
    
    def fit(self,X:np.array,y:np.array)-> None:
        X = self.is_2D(X)
        y = self.is_2D(y)
        N,D = X.shape

        if self.weights is None:
            self.weights = np.zeros(D)
        if self.bias is None:
            self.bias = 0
        
        if self.method == 'closed_form':
            x_bias = np.hstack((X,np.ones((N,1))))
            theta = np.linalg.pinv(x_bias.T@x_bias) @ x_bias.T @ y
            self.weights = theta[:-1].flatten()
            self.bias = theta[-1].item()
        elif self.method == "gradient_descent":
            for epoch in range(self.epochs):
                y_pred = np.dot(X,self.weights) + self.bias 
                error = y_pred - y.flatten()
                dw = (1/N)* np.dot(X.T,error)
                db = (1/N)* np.sum(error)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        else:
            raise ValueError("Method not recognized. Use 'closed_form' or 'gradient_descent'.") 

    def predict(self,X:np.array)-> np.array:
        X = self.is_2D(X)
        y_pred = np.dot(X,self.weights) + self.bias 
        return y_pred

    def fit_predict(self,X:np.array,y:np.array)-> np.array: 
        self.fit(X,y)
        return self.predict(X)


