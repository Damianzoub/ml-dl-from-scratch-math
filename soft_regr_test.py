import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as SKLogReg
from softm_regression.softm_regression import Softmax_Regression
# 1) Generate toy 2D dataset with 3 classes
X, y = make_blobs(n_samples=1000, centers=11, n_features=10, random_state=42)

plt.scatter(X[:,0], X[:,1], c=y, cmap="viridis", edgecolor="k")
plt.title("Toy 3-class dataset")
plt.show()

# 2) Train your softmax regression
model = Softmax_Regression(lr=0.1, epochs=1000)
model.fit(X, y, verbose=True)

y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print("Custom Softmax Regression Accuracy:", accuracy)

