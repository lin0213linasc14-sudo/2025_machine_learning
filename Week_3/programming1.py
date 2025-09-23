import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# true function
def true_function(x):
    return 1 / (1 + 25*x**2)

# data
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = true_function(X).ravel()

# 訓練/驗證/測試
X_train, Y_train = X[::3], y[::3]
X_val,   Y_val   = X[1::3], y[1::3]
X_test,  Y_test  = X[2::3], y[2::3]

mse_threshold = 1e-3
val_mse = np.inf   # intitation

iteration = 0
max_retries = 50  

while (val_mse >= mse_threshold  and iteration < max_retries):
    iteration += 1
    print(f"Training attempt {iteration}...")
    lr = 0.01 if iteration==1 else 0.005

    # crate model
    model = MLPRegressor(
        hidden_layer_sizes=(50,50),
        activation='tanh',
        solver='lbfgs', # 適合小資料回歸
        learning_rate_init= lr,
        max_iter=5000,
        random_state=np.random.randint(0,1000)  # 權重每次不同初始化
    )

    # training
    model.fit(X_train, Y_train)

    epsilon = 1e-3
    # validation
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_val_pred)

    y_plus  = model.predict(X_val + epsilon)
    y_minus = model.predict(X_val - epsilon)
    dy_dx_pred = (y_plus - y_minus) / (2*epsilon)
    dy_dx_true = -50*X_val / (1 + 25*X_val**2)**2
    val_mse_der = mean_squared_error(dy_dx_true, dy_dx_pred)
    val_mse = val_mse + val_mse_der
    print(f"Validation MSE: {val_mse:.6f}")

if val_mse < mse_threshold :
    print("Training successful!")
else:
    print("Reached max retries but validation MSE still too high.")

# test
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(Y_test, y_test_pred)
print("Test MSE:", test_mse)

# draw
X_plot = np.linspace(-1, 1, 400).reshape(-1, 1)
y_plot_true = true_function(X_plot)
y_plot_pred = model.predict(X_plot)

plt.figure(figsize=(8,5))
plt.plot(X_plot, y_plot_true, label='True f(x)', linewidth=2)
plt.plot(X_plot, y_plot_pred, '--', label='MLP prediction', linewidth=2)
plt.legend()
plt.grid(True)
plt.show()