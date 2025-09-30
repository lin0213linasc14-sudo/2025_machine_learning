from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# 讀檔
with open("O-A0038-003.xml", "r", encoding="utf-8") as f:
    xml_data = f.read()
soup = BeautifulSoup(xml_data, "xml")

# 抓時間
time = soup.find("DateTime").text
print("資料時間:", time)

# 抓範圍
geo = soup.find("GeoInfo")
bottom_left_lon = float(geo.find("BottomLeftLongitude").text)
bottom_left_lat = float(geo.find("BottomLeftLatitude").text)

# 抓溫度
content = soup.find("Content").text.strip().replace("\n", ",").split(",")
values = np.array([float(v) for v in content])

# 經度、緯度
grid = values.reshape(120, 67)
n_lat, n_lon = grid.shape
d = 0.03
lon = bottom_left_lon + np.arange(n_lon) * d  
lat = bottom_left_lat + np.arange(n_lat) * d
lon_grid, lat_grid = np.meshgrid(lon, lat)

# 分類資料集
# label = 0 (無效), 1 (有效)
label = np.where(grid == -999.0, 0, 1)
classification_data = pd.DataFrame({
    "lon": lon_grid.flatten(),
    "lat": lat_grid.flatten(),
    "label": label.flatten()
})
classification_data[["lon", "lat"]] = classification_data[["lon", "lat"]].round(2)
classification_data.to_csv("classification_dataset.csv", index=False, encoding="utf-8-sig")

# 回歸資料集
mask = grid != -999.0
regression_data = pd.DataFrame({
    "lon": lon_grid[mask],
    "lat": lat_grid[mask],
    "value": grid[mask]
})
regression_data[["lon", "lat"]] = regression_data[["lon", "lat"]].round(2)
regression_data.to_csv("regression_dataset.csv", index=False, encoding="utf-8-sig")

print("已輸出 classification_dataset.csv 和 regression_dataset.csv")

