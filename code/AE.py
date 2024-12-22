# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:06:11 2024

@author: admin
"""
import torch
import torch.nn as nn
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F  # 导入函数式接口
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): 数据的顶层目录路径。
        """
        self.root_dir = root_dir
        self.file_list = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.txt'):
                    filepath = os.path.join(subdir, file)
                    self.file_list.append(filepath)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.loadtxt(file_path, delimiter=' ')  # 根据你的数据分隔符进行调整
        data = data.reshape(1, 251, 31)  # 重新塑形为模型期望的形状
        data = torch.tensor(data, dtype=torch.float32)  # 转换为torch.FloatTensor
        return data
    def get_file_path(self, idx):
        return self.file_list[idx]
    def get_file_name(self, idx):
            """
            获取指定索引的数据文件的文件名，用于提取事件编号、开始时间和结束时间。
            """
            file_path = self.file_list[idx]
            file_name = os.path.basename(file_path)  # 仅获取文件名，不包含路径
            return file_name

# class CustomDataset(Dataset):
#     def __init__(self, root_dir):
#         """
#         Args:
#             root_dir (string): 数据的顶层目录路径。
#         """
#         self.root_dir = root_dir
#         self.file_list = []
#         for subdir, dirs, files in os.walk(root_dir):
#             for file in files:
#                 if file.endswith('.txt'):
#                     filepath = os.path.join(subdir, file)
#                     self.file_list.append(filepath)
        
#     def __len__(self):
#         return len(self.file_list)
    
#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         data = np.loadtxt(file_path, delimiter=' ')  # 根据你的数据分隔符进行调整
#         data = data.reshape(1, 251, 31)  # 重新塑形为模型期望的形状
        
#         # 添加标准化步骤
#         mean_value = np.mean(data)
#         std_value = np.std(data)
#         epsilon = 1e-8  # 避免除以零
#         data_standardized = (data - mean_value) / (std_value + epsilon)
        
#         data_tensor = torch.tensor(data_standardized, dtype=torch.float32)  # 转换为torch.FloatTensor
#         return data_tensor

#     def get_file_path(self, idx):
#         return self.file_list[idx]

#     def get_file_name(self, idx):
#         """
#         获取指定索引的数据文件的文件名，用于提取事件编号、开始时间和结束时间。
#         """
#         file_path = self.file_list[idx]
#         file_name = os.path.basename(file_path)  # 仅获取文件名，不包含路径
#         return file_name



# 定义CBAMBlock2D 注意力模块类
class ChannelAttention2D(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention2D, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBlock2D(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAMBlock2D, self).__init__()
        self.channelattention = ChannelAttention2D(channel, ratio)
        self.spatialattention = SpatialAttention2D(kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

import torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # Output: (8, 251, 31)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # Output: (8, 251, 31)
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 126, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 63, 8)
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 32, 4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.cbam = CBAMBlock2D(channel=32)  # CBAM模块

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 16 * 2, 32),
            nn.BatchNorm1d(32),###########################
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(32, 32 * 16 * 2),
            nn.ReLU()
        )

        # Updated transposed conv layers for decoding
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(0,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=(1,0)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )


        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )


        # self.conv6 = nn.Sequential(
        #     nn.ConvTranspose2d(8, 1, kernel_size=(5, 6), stride=1, padding=(2, 2)),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(1, 1, kernel_size=(5, 6), stride=1, padding=(2, 2)),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(),
        # )
        # 移除最后几层的 BatchNorm2d(1) 和 ReLU
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=(5, 6), stride=1, padding=(2, 2)),
            nn.BatchNorm2d(1),  # 移除
            nn.ReLU(),          # 移除
            nn.ConvTranspose2d(1, 1, kernel_size=(5, 6), stride=1, padding=(2, 2)),
            nn.BatchNorm2d(1),  # 移除
            nn.ReLU(),          # 移除
            # nn.Sigmoid()  
        )


        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # 编码部分
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 如果需要使用 CBAM 模块，取消注释
        # x = self.cbam(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        encoded = self.fc2(x)

        # 解码部分
        x = self.fc3(encoded)
        # x = self.dropout(x)
        x = self.fc4(x)
        x = x.view(-1, 32, 16, 2)
        x = self.conv4(x)
        x = self.conv5(x)
        decoded = self.conv6(x)
        decoded = F.interpolate(decoded, size=(251, 31), mode='bilinear', align_corners=False)

        return encoded, decoded



if __name__ == '__main__':
    # 创建你的数据集
    root_dir = 'E:/data'  # 你的数据文件夹路径
    dataset = CustomDataset(root_dir)
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    
    # 随机分割数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建对应的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 模型实例化
    model = AE().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # 训练设定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 300
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    # 早停策略
    early_stopping_patience = 10
    early_stopping_counter = 0
    
    # 训练和验证循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            _, decoded = model(data)
            loss = criterion(decoded, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
    
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                _, decoded = model(data)
                loss = criterion(decoded, data)
                val_loss += loss.item() * data.size(0)
    
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
    
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
        # 检查是否需要早停
        if val_loss < best_loss:
            best_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    # 绘制训练和验证损失图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    print('---------------------------------train over------------------------------------')
    
    
    


    from datetime import datetime

    # 假设已经定义了某种损失函数，例如MSELoss
    criterion = torch.nn.MSELoss()
    
    # 加载模型
    model = AE().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()  # 设置为评估模式
    
    # 新的数据集根目录
    new_root_dir = 'E:/data'  # 新的数据文件夹路径
    
    # 使用你自定义的 CustomDataset 类加载数据
    dataset = CustomDataset(new_root_dir)
    
    # 创建 DataLoader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 创建文件保存encoded、decoded数据和loss
    encoded_data = []
    decoded_data = []
    loss_logging = []
    catalog = []  # 保存最终数据
    
    for i, data in enumerate(data_loader):
        data = data.to(device)
        
        # 使用 get_file_name 获取文件名
        file_name = dataset.get_file_name(i)
        
        # 提取事件编号、开始时间和结束时间
        event_number = file_name.split('_')[1]  # 提取事件编号 "0"
        start_time = file_name.split('_')[3] + '_' + file_name.split('_')[4]  # 提取开始时间 "20240421_000005184"
        end_time = file_name.split('_')[6] + '_' + file_name.split('_')[7].split('.')[0]  # 提取结束时间 "20240421_000009184"
        
        folder_name = os.path.basename(os.path.dirname(dataset.get_file_path(i)))
    
        # 解析出channel号
        channel = folder_name.split('_')[-1]  # 假设channel信息总是在文件夹名的最后部分
        
        # 将日期时间字符串转换为 datetime 对象
        fmt = "%Y%m%d_%H%M%S%f"
        dt_start = datetime.strptime(start_time, fmt)
        dt_end = datetime.strptime(end_time, fmt)
        
        # 计算持续时间
        duration = (dt_end - dt_start).total_seconds()
    
        # 将事件信息存储为原始格式
        eventname_dic = {
            'event_number': event_number,
            'start_time': start_time,
            'duration': duration,  # 存储为秒
            'channel': channel    # 添加channel信息
        }
        # 不计算梯度
        with torch.no_grad():
            # 前向传播，获取编码和解码结果
            encoded, decoded = model(data)
    
            # 计算重建误差（损失）
            loss = criterion(data, decoded)
    
            # 将数据从GPU转移到CPU并转换为NumPy数组
            encoded = encoded.cpu().numpy().flatten()  # 将编码特征转换为NumPy数组并展平
            loss_value = loss.cpu().numpy()  # 将损失值转换为NumPy数组
    
            # 将 eventname_dic、编码特征和损失组合为一行
            row = [eventname_dic] + list(encoded) + [loss_value]
            catalog.append(row)
    
    # 转换为 NumPy 数组
    catalog = np.array(catalog, dtype=object)
    
    # 目标目录
    output_dir = 'E:/data/out/'
    
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 现在可以安全地打开文件进行写入了
    with open(os.path.join(output_dir, 'features_loss.dat'), 'wb') as f:
        pickle.dump(catalog, f)

    
    
'''
import pickle

# 文件路径
file_path = 'E:/data/out/features_loss.dat'

# 使用 pickle 加载文件
try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        # print(data)  # 打印加载的数据
        print(data[:2])
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")
  
'''    
    
    # # 加载模型
    # model = AE().to(device)
    # model.load_state_dict(torch.load('best_model.pth'))
    # model.eval()  # 设置为评估模式
    
    
    # # 假设 dataset 是已经加载好的全数据集
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # 单个样本处理以简化保存过程
    
    # # 创建文件保存encoded和decoded数据
    # encoded_data = []
    # decoded_data = []
    
    # # 处理数据
    # for data in data_loader:
    #     data = data.to(device)
    #     with torch.no_grad():
    #         encoded, decoded = model(data)
    #         encoded_data.append(encoded.cpu().numpy())  # 转换为NumPy数组并存储
    #         decoded_data.append(decoded.cpu().numpy())
    
    
    
    # np.save('encoded_data.npy', np.array(encoded_data))  # 保存编码数据
    # np.save('decoded_data.npy', np.array(decoded_data))  # 保存解码数据
