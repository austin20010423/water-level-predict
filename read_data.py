import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np


def read():
    df = pd.read_csv('data.csv')
    df = df.drop(labels=["水庫名稱", "有效容量(萬立方公尺)",
                         "統計時間", "集水區降雨量(毫米)", "與昨日水位差(公尺)", "水位(公尺)", "有效蓄水量(萬立方公尺)", "蓄水量百分比(%)", "Unnamed: 0"], axis="columns")
    # print(df)
    train_data = df[['昨日有效蓄水量(萬立方公尺)', '進水量(萬立方公尺)', '出水量(萬立方公尺)']]

    # 正規化資料
    data_normalized = (train_data - np.min(train_data)) / \
        (np.max(train_data) - np.min(train_data))
    print(data_normalized)
    return data_normalized, np.max(train_data), np.min(train_data)


# read()
