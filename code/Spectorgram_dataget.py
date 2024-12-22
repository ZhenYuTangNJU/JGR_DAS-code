# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:31:01 2024

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:25:08 2024

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import scipy as sp
import scipy.signal
from scipy.signal import convolve, correlate
import sys, obspy, os
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth as gps2DistAzimuth
from scipy.signal import spectrogram  # 确保这样导入
import requests
from datetime import datetime, timedelta
from io import BytesIO
import warnings
from matplotlib.colors import Normalize,LogNorm
warnings.filterwarnings('ignore')

from tkinter.filedialog import askopenfilenames
import tkinter as tk
import time
from scipy.signal import welch
from scipy.signal import iirnotch, lfilter
from scipy.signal import butter, filtfilt


def sta_lta_ratio(data, n_sta, n_lta):
    # STA和LTA的长度应为样本数，例如，n_sta = 100, n_lta = 1000
    sta = np.convolve(data, np.ones(n_sta), mode='valid') / n_sta
    lta = np.convolve(data, np.ones(n_lta), mode='valid') / n_lta

    sta = sta[n_lta//2:len(lta)+n_lta//2]
    lta = lta[:len(sta)]
    # 为了使STA和LTA数组长度相同，我们需要对LTA进行切片
    sta_lta_ratio = sta[:len(lta)] / lta

    return sta_lta_ratio


def trigger_event(ratio, trigger_threshold, detrigger_threshold, fs):
    events = []
    on_event = False
    last_event_time = -fs * 4  # 初始化为负的间隔，确保第一个事件可以被检测
    fs_interval = fs * 4  # 三秒的样本数

    for i, r in enumerate(ratio):
        if r > trigger_threshold and not on_event and (i - last_event_time) > fs_interval:
            events.append((i, 'ON'))
            on_event = True
            last_event_time = i  # 更新最近一次事件触发的时间
        elif r < detrigger_threshold and on_event:
            events.append((i, 'OFF'))
            on_event = False

    return events


def extract_and_transform(data, events, fs, pre, post):
    fs = int(fs)  # 确保采样频率是整数
    window_length = fs  # 1秒窗口用于频谱计算
    overlap = int(window_length * 0.9)  # 90% 重叠
    spectrograms = []
    normalized_spectrograms = []

    for event in events:
        if event[1] == 'ON':
            start = max(int(event[0] - pre), 0)
            end = min(int(start + pre + post), len(data))

            # 初始化最大能量和最佳开始秒
            max_energy = 0
            best_second_start = start

            # 逐点遍历每一秒以计算能量
            for second_start in range(start, end - fs + 1):  # 确保有完整的一秒数据
                second_end = second_start + fs
                segment = data[second_start:second_end]

                # 计算当前秒的频谱
                _, _, Sxx = spectrogram(segment, fs=fs, window='hann', nperseg=window_length, noverlap=overlap, scaling='spectrum')
                energy = np.sum(Sxx)  # 计算总能量

                # 更新最大能量和最佳开始秒
                if energy > max_energy:
                    max_energy = energy
                    best_second_start = second_start

            # 以能量最高的那一秒为中心，提取四秒数据
            centered_start = max(best_second_start - int(1.5 * fs), 0)
            centered_end = min(centered_start + int(4 * fs), len(data))
            centered_segment = data[centered_start:centered_end]

            # 计算以能量最高一秒为中心的四秒数据的spectrogram
            f_centered, t_centered, Sxx_centered = spectrogram(centered_segment, fs=fs, window='hann', nperseg=window_length, noverlap=overlap, scaling='spectrum')

            # 归一化处理
            max_value = np.max(Sxx_centered)
            Sxx_normalized = Sxx_centered / max_value if max_value != 0 else Sxx_centered  # 避免除以0

            spectrograms.append((f_centered, t_centered, Sxx_centered))
            normalized_spectrograms.append((f_centered, t_centered,centered_start, centered_end, Sxx_normalized))

    return spectrograms, normalized_spectrograms

def export_spectrogram_data(spectrograms, path):
    for i, (f, t, centered_start, centered_end, Sxx_normalized) in enumerate(spectrograms):
                
                # 设置采样频率
        sampling_rate = 500  # Hz
        time_interval = timedelta(seconds=1 / sampling_rate)
        # 设置开始时间
        start_time = datetime(2024, 5, 8, 12, 0, 2)
        
        # 计算索引对应的时间
        time_start_index = start_time + centered_start * time_interval
        time_end_index = start_time + centered_end * time_interval
        
        # 格式化时间字符串，确保无冒号和其它特殊字符
        formatted_start_time = time_start_index.strftime('%Y%m%d_%H%M%S%f')[:-3]  # 精确到毫秒
        formatted_end_time = time_end_index.strftime('%Y%m%d_%H%M%S%f')[:-3]  # 精确到毫秒
        
        filename = f'event_{i}_start_{formatted_start_time}_end_{formatted_end_time}.txt'
        filepath = os.path.join(path, filename)
        # np.savetxt(filepath, spectrogram, fmt='%.5f')
        np.savetxt(filepath, Sxx_normalized, fmt='%.6f')
        # print("Shape of Sxx_normalized:", Sxx_normalized.shape)

def process_and_export(channel_data, channel_index, date):
    # 计算STA/LTA比
    ratio = sta_lta_ratio(channel_data, int(1 * 500), int(10 * 500))
    
    # 设置触发和解除触发阈值
    trigger_threshold = 10
    detrigger_threshold = 3
    
    # 检测事件
    events = trigger_event(ratio, trigger_threshold, detrigger_threshold, 500)
    
    # 提取并转换为时频谱图
    spectrograms, normalized_spectrograms = extract_and_transform(channel_data, events, 500, int(1*500), int(3*500))
    
    # 导出路径和文件名
    export_path = f'E:/data/{date}_Channel{channel_index}/'
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    
    # 导出数据
    export_spectrogram_data(normalized_spectrograms, export_path)


# Example usage
if __name__ == "__main__":
    # 设置root窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏根窗口

    # 用于存储数据的字典
    DAS_data = {}

    # 初始化文件选择
    i = 1
    continue_selecting = True

    # 循环直到用户决定停止选择文件
    while continue_selecting:
        filenames = askopenfilenames(filetypes=[('dat files', '*.dat')], title='Select a File')
        if not filenames:
            print("No files selected, exiting.")
            continue_selecting = False
            continue

        for filename in filenames:
            fpath = os.path.abspath(filename)
            with open(fpath, 'rb') as f:
                A = np.frombuffer(f.read(), dtype=np.float32)
            C = A[0:64]  # Header file
            B = A[64:]   # Data
            fs = C[10]   # Sampling frequency
            N = int(C[16])  # Channels
            length = int(C[17])  # File length
            DAS_data1 = np.reshape(B, (N, int(length * fs)), order='F')
            var_name = f"data{i}"
            DAS_data[var_name] = DAS_data1.T
            i += 1

        # 询问用户是否继续选择文件
        answer = tk.messagebox.askyesno("Continue", "Do you want to select more files?")
        if not answer:
            continue_selecting = False




    DAS_data_values = [DAS_data[key] for key in DAS_data]
    DAS_data = np.concatenate(DAS_data_values, axis=0)  
    DAS_data=DAS_data.T#所有的DAS数据，每一行代表一个通道


    # filter_range= [0.1,50]         #设置滤波范围
    start="2023-07-21T14:09:59"
    Channel_start=0#传感段开始
    Channel_end=N#传感段结束
    #注意：后续的分析的道数是在去除首段之后的数据
    Channel_key=540-Channel_start#关键道
    # Channel_key_start=540-Channel_start#展示分析段开始
    # Channel_key_end=540-Channel_start#展示分析段结束
    # # 选择进行时频分析的道数
    # num_start=540-Channel_start
    # num_end=540-Channel_start

    #原始数据处理-时域
    DAS_data1=[]
    for i in range(Channel_start,Channel_end):
        data1=DAS_data[i,:]
        tr1=obspy.Trace(data=data1)
        tr1.stats.sampling_rate=fs
        tr1.stats.station="channal"+f"{i-1}"
        tr1.stats.starttime=start
        t1=tr1.stats.starttime
        tr1.detrend(type="linear")
        tr1.taper(0.05,type="cosine")
        # tr1.filter("bandpass",freqmin=filter_range[0],freqmax=filter_range[1],zerophase="true",corners=4)
        DAS_data1.append(tr1.data) #滤波后的所有传感段的数据
        # tr1 = normalize(tr1, norm_method="lbit") #归一化
        # tr1 = whiten(tr1, filter_range[0], filter_range[1])#谱白化
        
    DAS_data_array = np.array(DAS_data1) #list转化为矩阵 DAS_data_array已经去除跳线光纤




    # # 假设data是你的多通道光纤数据，shape为(samples, channels)
    # signal = DAS_data_array[110, :]
    
    
    for channel_index in range(110, 146):
        signal = DAS_data_array[channel_index, :]
        date = '20240508PM'  # 假设这是你设置的日期
        process_and_export(signal, channel_index, date)
    
    
    
    # # Define STA and LTA window lengths
    # n_sta = int(1 * 500)  # 0.5秒窗口
    # n_lta = int(10 * 500)   # 10秒窗口


    # # Calculate STA/LTA ratio
    # ratio = sta_lta_ratio(signal, n_sta, n_lta)


    # # Set trigger and detrigger thresholds
    # trigger_threshold = 10
    # detrigger_threshold = 3
    
    
    # # min_duration = 0.5 * 500  # 假设事件至少需要持续10秒才被视为有效

    # # Detect events
    # events = trigger_event(ratio, trigger_threshold, detrigger_threshold, 500)
    
    # print("Detected events:")
    # for event in events:
    #     print(f"Index: {event[0]}, Status: {event[1]}")
        

    # spectrograms, normalized_spectrograms = extract_and_transform(signal, events, fs,int(1*fs),int(3*fs))
    
    # # for f, t, Sxx in spectrograms:
    # #     print(f"Frequency bins: {f.shape[0]}, Time bins: {t.shape[0]} Spectrogram shape: {Sxx.shape}")

    # from matplotlib.colors import LinearSegmentedColormap
    # # 自定义颜色
    # colors = [(63/255, 99/255, 171/255), (255/255, 247/255, 177/255), (178/255, 11/255, 36/255)]  # 深蓝，浅绿，黄色
    # cmap_name = 'my_custom_cmap'
    
    # # 创建颜色映射
    # custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    # # 绘制谱图
    # for i, (f, t, Sxx) in enumerate(normalized_spectrograms):
    #     plt.figure(figsize=(10, 4))
    #     plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap=custom_cmap, vmin=-20, vmax=0)#, vmin=-50, vmax=-10
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    #     plt.title(f'Time-Frequency Spectrogram of Event {i+1}')
    #     plt.colorbar(label='Intensity [dB]')
    #     plt.show()
    # # # Plotting the spectrograms
    # # for i, (f, t, Sxx) in enumerate(spectrograms):
    # #     plt.figure(figsize=(10, 4))
    # #     plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', vmin=-30, vmax=-10)
    # #     plt.ylabel('Frequency [Hz]')
    # #     plt.xlabel('Time [sec]')
    # #     plt.title(f'Time-Frequency Spectrogram of Event {i+1}')
    # #     plt.colorbar(label='Intensity [dB]')
    # #     plt.show()
    
    
    # # 绘制原始数据
    # plt.subplot(2, 1, 1)  # 两行一列的第一个图
    # plt.plot(signal, label='Original Data', color='blue')
    # plt.title('Original Data')
    # plt.xlabel('Sample Number')
    # plt.ylabel('Amplitude')
    # plt.legend()
    
    # # 绘制STA/LTA比
    # plt.subplot(2, 1, 2)  # 两行一列的第二个图
    # plt.plot(ratio, label='STA/LTA Ratio', color='red')
    # plt.title('STA/LTA Ratio over Time')
    # plt.xlabel('Sample Number')
    # plt.ylabel('Ratio')
    # plt.legend()

    # plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    # plt.show()
    
    
    # # 定义导出的文件路径
    # export_path = '‪C:/Users/admin/Desktop/data/'.strip('\u202a')

    # # 确保导出路径存在
    # if not os.path.exists(export_path):
    #     os.makedirs(export_path)

    # # 调用函数导出数据
    # export_spectrogram_data(normalized_spectrograms, export_path)
    # # #################引入最小持续时间
    # min_duration = 10 * 500  # 假设事件至少需要持续10秒才被视为有效
