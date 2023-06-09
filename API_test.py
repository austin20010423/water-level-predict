import water_level_pre
import read_data as rd
import pandas as pd
import matplotlib.pyplot as plt

data, max, min = rd.read()
re = water_level_pre.predict_water_level(
    data, max, min)

df = pd.read_csv('data.csv')
result = []
level = df['昨日有效蓄水量(萬立方公尺)'].tolist()
level = level[2901:3201]
water_in = re[0]
water_out = re[1]
tmp = 0
for i in range(0, len(level)):
    tmp = level[i] + water_in[i] - water_out[i]
    result.append(tmp)


l = df['有效蓄水量(萬立方公尺)'].tolist()
need = l[2901:3201]
print('need: ', need)
print('result: ', result)
dis = []

for j in range(0, len(need)):
    dis.append(float(need[j]) - float(result[j]))

print('differ: ', dis)

x = [i for i in range(1, 301)]

# 畫圖
plt.plot(x, result, label='predict')
plt.plot(x, need, label='initial data')

# 設定標題和軸標籤
plt.title("reservoir water level")
plt.xlabel("2021/1/9~2021/2/17")
plt.ylabel("Effective storage capacity")
plt.legend()

# 顯示圖形
plt.show()
