import numpy as np 
import matplotlib.pyplot as plt 
from lin_reg.linear_reg import LinearRegressionClosedForm
from sklearn.linear_model import LinearRegression as SKLinearRegression
rng = np.random.default_rng(0)

N= 100 
X = rng.uniform(0,10,size=(N,1))
true_w , true_b= 2.0 , 1.0

y = true_w * X + true_b + rng.normal(0,1.0,size=(N,1)) 

#1) Traind GD model 
gd_model = LinearRegressionClosedForm()
gd_history = gd_model.fit(X,y)
print(f"GD model learned w: w={gd_model.w}")
print(f"GD model learned b: b={gd_model.b}")

sk_linear = SKLinearRegression()
sk_linear.fit(X,y)
print("Sklearn learnd w: ",sk_linear.coef_)
print("Sklearn learnd b: ",sk_linear.intercept_)
plt.scatter(X, y, s=20, alpha=0.7, label="data")

# line from your GD model
xs = np.linspace(0, 10, 100).reshape(-1, 1)
ys_gd = gd_model.predict(xs)
plt.plot(xs, ys_gd, color="red", label="GD fit")

# line from sklearn
ys_sk = sk_linear.predict(xs)
plt.plot(xs, ys_sk, color="green", linestyle="--", label="Sklearn fit")

plt.legend()
plt.title("Linear Regression: GD vs Sklearn")
plt.show()

