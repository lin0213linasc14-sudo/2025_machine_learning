import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

data = pd.read_csv("classification_dataset.csv")
print("資料前五筆：")
print(data.head())

# 經度與緯度
X = data[["lon", "lat"]]
# label = 0(無效) or 1(有效)
y = data["label"]

# 訓練/測試分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 將訓練資料轉為 numpy 陣列以便計算
X_train_np = X_train.values
y_train_np = y_train.values

## a) 實作 GDA 演算法

class GaussianDiscriminantAnalysis:
    """
    自定義的高斯判別分析 (GDA) 實作。
    採用 QDA 形式 (獨立的共變異數矩陣)。
    """
    def __init__(self):
        self.phi = 0  # 類別 1 的先驗機率 P(y=1)
        self.mu_0 = None # 類別 0 的均值向量
        self.mu_1 = None # 類別 1 的均值向量
        self.sigma_0 = None # 類別 0 的共變異數矩陣
        self.sigma_1 = None # 類別 1 的共變異數矩陣

    def fit(self, X, y):
        """
        訓練模型：估計高斯分佈的參數 P(y), P(x|y=0), P(x|y=1)。
        """
        # 1. 計算先驗機率 (Priors)
        m = len(y)
        m_1 = np.sum(y == 1)
        self.phi = m_1 / m

        # 2. 劃分資料
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        
        # 3. 計算均值向量 (Means)
        self.mu_0 = np.mean(X_0, axis=0)
        self.mu_1 = np.mean(X_1, axis=0)
        
        # 4. 計算共變異數矩陣 (Covariances) - 採用 QDA 形式 (獨立的 Sigma)
        # np.cov 預設會除以 N-1，因此使用 rowvar=False
        self.sigma_0 = np.cov(X_0, rowvar=False) 
        self.sigma_1 = np.cov(X_1, rowvar=False)

    def predict(self, X):
        """
        預測類別：使用貝氏定理計算後驗機率並進行分類。
        """
        
        # 計算 P(x|y=0) 和 P(x|y=1)
        # 使用 scipy.stats.multivariate_normal 計算多元高斯機率密度函數 (PDF)
        
        # 確保共變異數矩陣是可逆且正定的，以避免數值錯誤
        try:
            p_x_given_y0 = multivariate_normal.pdf(X, mean=self.mu_0, cov=self.sigma_0)
            p_x_given_y1 = multivariate_normal.pdf(X, mean=self.mu_1, cov=self.sigma_1)
        except np.linalg.LinAlgError as e:
            print(f"警告：共變異數矩陣可能奇異，導致數值錯誤: {e}")
            return np.zeros(len(X))

        # 計算判別函數 (Discriminant Function): $\log P(y=k|x) \propto \log P(x|y=k) + \log P(y=k)$
        
        # 對數機率 (Log-Likelihood)
        log_p_x_given_y0 = np.log(p_x_given_y0 + 1e-100) # 加上微小值避免 log(0)
        log_p_x_given_y1 = np.log(p_x_given_y1 + 1e-100)
        
        # 類別 0 的對數後驗機率 (未正規化)
        log_posterior_0 = log_p_x_given_y0 + np.log(1 - self.phi)
        
        # 類別 1 的對數後驗機率 (未正規化)
        log_posterior_1 = log_p_x_given_y1 + np.log(self.phi)

        # 決策規則：選擇對數後驗機率較大的類別
        return (log_posterior_1 > log_posterior_0).astype(int)

# 訓練模型
gda_model = GaussianDiscriminantAnalysis()
gda_model.fit(X_train_np, y_train_np)

## c) 訓練模型與報告性能 (準確度)

# 在測試集上進行預測
y_pred = gda_model.predict(X_test.values)

# 計算準確度
accuracy = np.mean(y_pred == y_test.values)

print("\n" + "="*50)
print("c) 模型性能報告")
print("="*50)
print(f"模型在測試集上的準確度 (Accuracy): {accuracy:.4f}")

## d) 繪製決策邊界

# 創建網格點
lon_min, lon_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
lat_min, lat_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(lon_min, lon_max, 200),
                     np.linspace(lat_min, lat_max, 200))

# 預測網格點的類別
Z = gda_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪圖
plt.figure(figsize=(10, 7))

# 繪製決策區域
plt.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['blue', 'red'])

# 繪製決策邊界 (Z=0.5 處的等高線)
# QDA 的決策邊界是 log(P(x|y=1)P(y=1)) = log(P(x|y=0)P(y=0))
# 等價於 P(x|y=1)/P(x|y=0) = P(y=0)/P(y=1)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

# 繪製訓練資料點
# scatter = plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, 
#                       marker='o', edgecolor='k', label='Training Data')
#
# 繪製測試資料點
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, 
            marker='x', edgecolor='k', linewidth=2, label='Test Data')

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GDA model’s decision boundary and the distribution of data")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()