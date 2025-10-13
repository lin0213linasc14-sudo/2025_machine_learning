import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================================================================
# 讀取資料
# =========================================================================
data_c = pd.read_csv("classification_dataset.csv")
X = data_c[["lon", "lat"]]
y_C = data_c["label"]

data_r = pd.read_csv("regression_dataset.csv")
X_R = data_r[["lon", "lat"]]
y_R = data_r["value"].values.reshape(-1, 1)

# =========================================================================
# 模型 C(x): SVM 分類器
# =========================================================================
print("--- 1. 訓練分類模型 C(x) (SVM) ---")
X_C_train, X_C_test, y_C_train, y_C_test = train_test_split(
    X, y_C, test_size=0.2, random_state=42, stratify=y_C
)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

svc = svm.SVC(kernel='rbf')
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(X_C_train, y_C_train)

best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
C_model = grid_search.best_estimator_

print(f"SVM 最佳參數: C={best_C}, gamma={best_gamma}")
C_train_acc = accuracy_score(y_C_train, C_model.predict(X_C_train))
C_test_acc = accuracy_score(y_C_test, C_model.predict(X_C_test))
print(f"C(x) 訓練集準確度: {C_train_acc:.4f}, 測試集準確度: {C_test_acc:.4f}")

# =========================================================================
# 模型 R(x): 深度神經網絡 (Keras)
# =========================================================================
print("\n--- 2. 訓練迴歸模型 R(x) (Keras NN) ---")

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_R)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_R)

X_R_train, X_R_test, y_R_train, y_R_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

R_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
R_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

R_model.fit(
    X_R_train, y_R_train,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

# =========================================================================
# a) 實作複合模型 h(x)
# =========================================================================
def model_h(X_input):
    """
    h(x) = R(x) if C(x)=1, else -999
    X_input: ndarray shape (n_samples, 2)
    """
    C_pred = C_model.predict(X_input)
    h_output = np.full(len(X_input), -999.0)

    if np.any(C_pred == 1):
        X_reg_scaled = scaler_X.transform(X_input[C_pred == 1])
        R_pred_scaled = R_model.predict(X_reg_scaled, verbose=0)
        R_pred = scaler_y.inverse_transform(R_pred_scaled)
        h_output[C_pred == 1] = R_pred.ravel()

    return h_output, C_pred

# =========================================================================
# b) 驗證分段定義
# =========================================================================
print("\n--- b) 驗證分段定義 ---")

# 對測試集進行預測
h_test_pred, C_test_pred = model_h(X_C_test.values)

# 對照 R(x) 的結果
X_test_scaled = scaler_X.transform(X_C_test.values)
R_test_pred_scaled = R_model.predict(X_test_scaled, verbose=0)
R_test_pred_original = scaler_y.inverse_transform(R_test_pred_scaled).flatten()

# 驗證分段條件
idx_C_0 = (C_test_pred == 0)
idx_C_1 = (C_test_pred == 1)

h_output_C_0 = h_test_pred[idx_C_0]
h_output_C_1 = h_test_pred[idx_C_1]
R_output_C_1 = R_test_pred_original[idx_C_1]

is_C_0_correct = np.all(h_output_C_0 == -999.0)
is_C_1_correct = np.all(np.isclose(h_output_C_1, R_output_C_1))

print(f"C(x)=0 樣本數: {np.sum(idx_C_0)}")
print(f"C(x)=1 樣本數: {np.sum(idx_C_1)}")
print(f"驗證 C(x)=0 時 h(x) 是否為 -999: {'成功' if is_C_0_correct else '失敗'}")
print(f"驗證 C(x)=1 時 h(x) 是否等於 R(x): {'成功' if is_C_1_correct else '失敗'}")

verification_table = pd.DataFrame({
    'C(x)': C_test_pred[:10],
    'R(x) 輸出': R_test_pred_original[:10].round(2),
    'h(x) 輸出': h_test_pred[:10].round(2),
    '預期行為': np.where(C_test_pred[:10] == 1, 'R(x)', '-999')
})
print("\n測試集前 10 筆 h(x) 輸出驗證:")
print(verification_table)

# =========================================================================
# c) 複合函數解釋
# =========================================================================
print("\n--- c) 複合函數 h(x) 構建解釋 ---")
print("""
複合函數 h(x) 是一個「守門員」(Gatekeeper) 模型：
1. **分類門控 (C(x))**: 決定該資料點是否屬於「有效」類別。
2. **條件執行 (R(x))**:
   - 若 C(x)=1，執行 R(x) 並回傳預測值。
   - 若 C(x)=0，輸出固定值 -999。
確保只有通過分類篩選的資料才會進入迴歸階段。
""")

# =========================================================================
# d) 視覺化模型行為
# =========================================================================
print("\n--- d) 模型行為繪圖 ---")

plt.figure(figsize=(8, 12))
plt.scatter(
    X_C_test[idx_C_0]['lon'], X_C_test[idx_C_0]['lat'],
    c='red', marker='o', s=80, alpha=0.6,
    label=f'C(x)=0 → h(x)=-999 (n={np.sum(idx_C_0)})'
)

scatter_R = plt.scatter(
    X_C_test[idx_C_1]['lon'], X_C_test[idx_C_1]['lat'],
    c=h_test_pred[idx_C_1], cmap='viridis', marker='x', s=100, linewidth=2,
    label=f'C(x)=1 → h(x)=R(x) (n={np.sum(idx_C_1)})'
)

cbar = plt.colorbar(scatter_R)
cbar.set_label('h(x) output value ', rotation=270, labelpad=15)

plt.title('Combined Piecewise Model h(x) on test set')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()