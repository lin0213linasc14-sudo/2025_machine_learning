import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("regression_dataset.csv")

print(data.head())
# 經度與緯度
X = data[["lon", "lat"]].values
# 溫度
y = data["value"].values.reshape(-1, 1)

# 標準化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 訓練/測試
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=100,
    verbose=0,
    validation_split = 0.2
)

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {mae:.4f}") 

# loss curve
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()


lon_min, lon_max = data['lon'].min(), data['lon'].max()
lat_min, lat_max = data['lat'].min(), data['lat'].max()
grid_resolution = 50
lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

grid_points = np.c_[lon_mesh.ravel(), lat_mesh.ravel()]
grid_points_scaled = scaler_X.transform(grid_points)

pred_scaled = model.predict(grid_points_scaled)
pred_values = scaler_y.inverse_transform(pred_scaled)
pred_values_mesh = pred_values.reshape(lon_mesh.shape)

# draw
plt.figure(figsize=(10,4))

# 預測溫度分布
plt.subplot(1,2,1)
plt.scatter(grid_points[:,0], grid_points[:,1], c=pred_values.ravel(), cmap='jet', s=80)
plt.colorbar(label='Predicted Temperature')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Predicted Temperature')

# 原始資料
plt.subplot(1,2,2)
plt.scatter(data['lon'], data['lat'], c=data['value'], cmap='jet', s=80)
plt.colorbar(label='Original Temperature')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Original Temperature')

plt.tight_layout()
plt.show()
