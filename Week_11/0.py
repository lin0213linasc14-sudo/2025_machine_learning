import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# 系統 A 參數
m1, c1, k1 = 1.0, 0.5, 2.0

def system_A(t, y):
    x, v = y
    dxdt = v
    dvdt = -(c1*v + k1*x)/m1
    return [dxdt, dvdt]

# 系統 B 參數（AI 未知）
m2, c2, k2 = 2.0, 1.0, 4.0

def system_B(t, y):
    x, v = y
    dxdt = v
    dvdt = -(c2*v + k2*x)/m2
    return [dxdt, dvdt]

# 時間範圍與初始條件
t_eval = np.linspace(0, 10, 200)
y0 = [1.0, 0.0]

# 模擬系統 A 和 B
sol_A = solve_ivp(system_A, [0,10], y0, t_eval=t_eval)
sol_B = solve_ivp(system_B, [0,10], y0, t_eval=t_eval)

# 簡單線性映射 x2 = a * x1 + b
reg = LinearRegression()
reg.fit(sol_A.y[0].reshape(-1,1), sol_B.y[0])
a = reg.coef_[0]
b = reg.intercept_

print("最佳線性對應: x2 ~ {:.2f} * x1 + {:.2f}".format(a, b))

# 視覺化比較
x2_pred = a * sol_A.y[0] + b
rmse = np.sqrt(mean_squared_error(sol_B.y[0], x2_pred))
print("RMSE = {:.4f}".format(rmse))

plt.figure(figsize=(8,5))
plt.plot(t_eval, sol_B.y[0], label="System B True")
plt.plot(t_eval, x2_pred, '--', label="Predicted Mapping from A")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.legend()
plt.title("Cross-System Mapping via Linear Regression")
plt.show()
