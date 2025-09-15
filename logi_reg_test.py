import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SKLogistic
from logisti_regression.log_regression import Logic_Regression
# reproducibility
rng = np.random.default_rng(0)

# --- 1. Synthetic dataset (2 classes, separable) ---
N = 100
X_class0 = rng.normal(loc=-2.0, scale=1.0, size=(N//2, 2))
X_class1 = rng.normal(loc=+2.0, scale=1.0, size=(N//2, 2))

X = np.vstack([X_class0, X_class1])    # shape (100, 2)
y = np.array([0]*(N//2) + [1]*(N//2)) # shape (100,)

#-- 3. Fit custom logistic regression ---
model = Logic_Regression()
model.fit(X,y)
proba = model.predict_proba(X)
pred = model.predict(X)
loss = model.loss(proba,y)
print("Custom weights:", model.w)
print("Custom bias:", model.b)
print("Custom loss:", loss)
print("Train accuracy: ",(pred==y).mean())

plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolor="k")
x1_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
if model.w[1] != 0:   # avoid divide by zero
    x2_vals = -(model.w[0]/model.w[1]) * x1_vals - (model.b/model.w[1])
    plt.plot(x1_vals, x2_vals, color="black", linewidth=2, label="Decision boundary")
plt.title("Logistic Regression (custom)")
plt.show()


# --- 2. Fit sklearn logistic regression ---
sk_model = SKLogistic()
sk_model.fit(X, y)

print("Sklearn weights:", sk_model.coef_)
print("Sklearn bias:", sk_model.intercept_)

# --- 3. Predict on a grid (for visualization) ---
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = sk_model.predict_proba(grid)[:, 1].reshape(xx.shape)

# --- 4. Plot decision boundary ---
plt.contourf(xx, yy, probs, levels=[0,0.5,1], alpha=0.2, colors=["blue","red"])
plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolor="k")
plt.title("Logistic Regression (sklearn)")
plt.show()
