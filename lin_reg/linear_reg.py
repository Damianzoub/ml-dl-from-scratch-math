import numpy as np

class LinearRegressionClosedForm:
    def __init__(self, method="lstsq"):  # "lstsq" | "pinv" | "solve"
        self.method = method
        self.w = None  # (D,)
        self.b = None  # scalar

    def _as_2d(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X, y):
        X = self._as_2d(X)
        y = np.asarray(y, float).reshape(-1)
        N = X.shape[0]
        X_aug = np.c_[X, np.ones(N)]

        if self.method == "lstsq":
            theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        elif self.method == "pinv":
            theta = np.linalg.pinv(X_aug) @ y
        elif self.method == "solve":
            XtX = X_aug.T @ X_aug
            Xty = X_aug.T @ y
            theta = np.linalg.solve(XtX, Xty)
        else:
            raise ValueError("method must be 'lstsq', 'pinv', or 'solve'")

        self.w = theta[:-1]
        print(theta)
        self.b = float(theta[-1])
        return self

    def predict(self, X):
        X = self._as_2d(X)
        return X @ self.w + self.b
