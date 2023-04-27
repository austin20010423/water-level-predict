import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import read_data as rd


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        predictions = self.linear(lstm_out.view(len(input), -1))
        return predictions[-1]


def predict_water_level(data, max, min, input_size=3, hidden_size=32, output_size=3, num_epochs=1000):

    # 定義訓練集和測試集的大小
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    print('train size:', train_size, 'test size:', test_size)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 將資料轉換成numpy array
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    # 定義超參數
    learning_rate = 0.01

    # 初始化模型
    lstm = LSTM(input_size, hidden_size, output_size)

    # 定義損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)

    # 訓練模型
    for epoch in range(num_epochs):
        train_loss = 0
        for i in range(len(train_data) - 1):
            input = torch.tensor(train_data[i:i+1]).float()
            target = torch.tensor(train_data[i+1:i+2]).float()
            # print(input.shape)
            optimizer.zero_grad()
            output = lstm(input)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        # if epoch % 20 == 0:
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch+1, train_loss / len(train_data)))

    # 測試模型

    pre_water_level = []
    pre_in_water = []
    pre_loss_water = []
    with torch.no_grad():
        input = torch.tensor(test_data[0:1]).float()
        # print(input.shape)
        for i in range(test_size):
            output = lstm(input)
            # print(output.size())
            pre_water_level.append(output[0].item())
            pre_in_water.append(output[1].item())
            pre_loss_water.append(output[2].item())
            output = output.view(1, 3)
            # print(output.size())
            input = output

    print(np.array(pre_water_level[:10]))
    # 輸出預測結果
    water_level_restored = np.array(
        pre_water_level[:10]) * (max[0] - min[0]) + min[0]
    print('water level: ', water_level_restored)
    in_water_restored = np.array(
        pre_in_water[:10]) * (max[1] - min[1]) + min[1]
    print('in water: ', in_water_restored)
    loss_water_restored = np.array(
        pre_loss_water[:10]) * (max[2] - min[2]) + min[2]
    print('loss water: ', loss_water_restored)
    return (water_level_restored, in_water_restored, loss_water_restored)


data, max, min = rd.read()
predict_water_level(data, max, min)
