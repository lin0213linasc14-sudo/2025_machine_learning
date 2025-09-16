import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# True function
def true_function(x):
    return 1 / (1 + 25*x**2)

# Generate training data
# Sample some points in the overall interval [-1, 1], 
X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = true_function(X) 

# Create polynomial features. (The degree can be adjusted)
deg = 32
poly = PolynomialFeatures(degree = deg)
X_poly = poly.fit_transform(X)

# Polynomial regression
model = LinearRegression()
model.fit(X_poly, y)

# test point
X_test = np.linspace(-1, 1, 400).reshape(-1, 1)
X_test_poly = poly.transform(X_test)
y_true = true_function(X_test)
y_pred = model.predict(X_test_poly)

# MSE
mse = mean_squared_error(y_true, y_pred)
print("(MSE):", mse)

# plot Figure
plt.figure(figsize=(8, 6))
plt.plot(X_test, y_true, color="green", linewidth=2, label="True function f(x)=1/(x^2+25)")
plt.plot(X_test, y_pred, color="red", linestyle="--", linewidth=2, label= f"Polynomial regression approximation in degree {deg}")
plt.title(f"Polynomial regression vs True function (MSE = {mse})")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
