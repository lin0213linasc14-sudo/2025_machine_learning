import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("classification_dataset.csv")


print(data.head())
# 經度與緯度
X = data[["lon", "lat"]]
# label (0/1)
y = data["label"]

# 訓練/測試
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 設定 SVM 參數搜尋範圍
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

# SVM訓練並找出最適合的C跟gamma
svc = svm.SVC(kernel='rbf')
grid_search = GridSearchCV(svc, param_grid, cv=5) 
grid_search.fit(X_train, y_train)

# 顯示最佳參數
best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
print(f"Best C: {best_C}, Best gamma: {best_gamma}")

# 使用最佳參數訓練模型
model = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True)
model.fit(X_train, y_train)  
y_prob = model.predict_proba(X_test)

# 測試集準確率
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_acc:.2f}")

#  可視化決策邊界
xx, yy = np.meshgrid(
    np.linspace(X['lon'].min()-0.05, X['lon'].max()+0.05, 500),
    np.linspace(X['lat'].min()-0.05, X['lat'].max()+0.05, 500)
)

Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')  # 決策邊界
plt.scatter(X['lon'], X['lat'], c=y, cmap=plt.cm.coolwarm, s=5, edgecolors='k')
# plt.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1],
            # s=100, facecolors='none', edgecolors='yellow')  # 支援向量
plt.title(f"非線性 SVM 分類 (Best C={best_C}, gamma={best_gamma})")
plt.show()

# 可視化測試集決策邊界與資料點

plt.figure(figsize=(6,6))
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')  # 決策邊界
plt.scatter(X_test['lon'], X_test['lat'], c=y_test, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
plt.title(f"非線性 SVM 測試集分類 (Best C={best_C}, gamma={best_gamma})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

