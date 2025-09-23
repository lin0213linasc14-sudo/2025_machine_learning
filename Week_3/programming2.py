import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# trun finction
def true_function(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def true_derivative(x):
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

# data
X = np.linspace(-1, 1, 100).reshape(-1, 1).astype(np.float32)
y = true_function(X).ravel().astype(np.float32)

# 訓練/驗證/測試
X_train, Y_train = X[::3], y[::3]
X_val,   Y_val   = X[1::3], y[1::3]
X_test,  Y_test  = X[2::3], y[2::3]

# trun to TensorFlow Tensor
X_train_tf = tf.constant(X_train)
Y_train_tf = tf.constant(Y_train.reshape(-1, 1))
X_val_tf   = tf.constant(X_val)
Y_val_tf   = tf.constant(Y_val.reshape(-1, 1))
X_test_tf  = tf.constant(X_test)
Y_test_tf  = tf.constant(Y_test.reshape(-1, 1))

# create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(50, activation='tanh'),
    tf.keras.layers.Dense(1)
])

# 更新權重的演算法
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

epochs = 1000

# training
iteration = 0
max_retries = 50  
mse_threshold = 1e-5
val_mse = float("inf")
val_mse_der = float("inf")

while (val_mse >= mse_threshold or val_mse_der >= mse_threshold) and iteration < max_retries:
  iteration += 1
  for epoch in range(epochs+1):
      with tf.GradientTape() as tape:

          y_pred = model(X_train_tf, training=True)

          with tf.GradientTape() as tape_x:
              tape_x.watch(X_train_tf)
              y_train_pred_for_grad = model(X_train_tf, training=True)
          dy_dx_pred = tape_x.gradient(y_train_pred_for_grad, X_train_tf)

          dy_dx_true_train = true_derivative(X_train_tf)

          # Loss fuction (include div. part)
          loss_val = tf.reduce_mean((y_pred - Y_train_tf)**2) \
                  + tf.reduce_mean((dy_dx_pred - dy_dx_true_train)**2)

      grads = tape.gradient(loss_val, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      if epoch % 200 == 0:
          print(f"Epoch {epoch+1}, Loss: {loss_val.numpy():.6f}")

  # validation
  y_val_pred = model(X_val_tf, training=False).numpy().ravel()
  val_mse = mean_squared_error(Y_val, y_val_pred)

  with tf.GradientTape() as tape_x_val:
      tape_x_val.watch(X_val_tf)
      y_val_pred_tf = model(X_val_tf, training=False)
  dy_dx_val_pred = tape_x_val.gradient(y_val_pred_tf, X_val_tf).numpy().ravel()
  dy_dx_val_true = true_derivative(X_val_tf).numpy().ravel()
  val_mse_der = mean_squared_error(dy_dx_val_true, dy_dx_val_pred)

  val_mse = val_mse + val_mse_der

  print(f"\nValidation MSE: {val_mse:.6f}")

if val_mse < mse_threshold:
  print("Training successful!")
else:
  print("Reached max retries but validation MSE still too high.")

# test
y_test_pred = model(X_test_tf, training=False).numpy().ravel()
test_mse = mean_squared_error(Y_test, y_test_pred)

with tf.GradientTape() as tape_x_test:
    tape_x_test.watch(X_test_tf)
    y_test_pred_tf = model(X_test_tf, training=False)
dy_dx_test_pred = tape_x_test.gradient(y_test_pred_tf, X_test_tf).numpy().ravel()
dy_dx_test_true = true_derivative(X_test_tf).numpy().ravel()
test_mse_der = mean_squared_error(dy_dx_test_true, dy_dx_test_pred)
test_mse = test_mse + test_mse_der
print(f"Test MSE: {test_mse :.6f}")

# draw
X_plot = np.linspace(-1, 1, 400).reshape(-1, 1).astype(np.float32)
X_plot_tf = tf.constant(X_plot)
y_plot_true = true_function(X_plot)
y_plot_pred = model(X_plot_tf).numpy()

plt.figure(figsize=(8,5))
plt.plot(X_plot, y_plot_true, label='True f(x)', linewidth=2)
plt.plot(X_plot, y_plot_pred, '--', label='MLP prediction', linewidth=2)
plt.legend()
plt.grid(True)
plt.show()
