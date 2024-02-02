import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data2.csv')
comluns_data = df[['p1_points_won','p2_points_won']]

p1_points_won = comluns_data['p1_points_won']
p2_points_won = comluns_data['p2_points_won']
p1_points_won = p1_points_won.to_numpy()
p2_points_won = p2_points_won.to_numpy()

from scipy.fftpack import fft, ifft

data_denoised = np.zeros(len(p1_points_won))
np.set_printoptions(formatter={'float': '{:.6f}'.format})

def fit(start, end):
    p1 = p1_points_won[start:end]
    p2 = p2_points_won[start:end]
    # with open('output.txt', 'w') as f:
    #     print(p1 - p2, file=f)
    data_fft = fft(p1 - p2)
    # data_fft_len = np.abs(data_fft)
    # plt.figure(figsize=(12, 6))
    # plt.plot(np.arange(len(data_fft_len)), np.abs(data_fft))
    threshold = 10
    data_fft_abs = np.abs(data_fft)
    data_fft[data_fft_abs < threshold] = 0
    data_denoised[start:end] = np.real(ifft(data_fft))
    # with open('output.txt', 'w') as f:
    #     print(data_denoised[start:end], file=f)
        
    # original_data = p1 - p2
    # denoised_data = data_denoised[start:end]

    # plt.figure(figsize=(12, 6))
    # print(original_data, denoised_data)
    # x = np.arange(len(original_data))
    # plt.plot(x, original_data, label='Original')
    # plt.plot(x, denoised_data, label='Denoised')
    # # sns.lineplot(x=np.arange(len(original_data)), y=original_data, label='Original')
    # # sns.lineplot(x=range(len(denoised_data)), y=denoised_data, label='Denoised')
    # plt.legend()
    # plt.title('DFT Denoising Result')
    # plt.show()
    
pre = 0
for i in range(len(p1_points_won)):
    if(i != 0):
        if(p1_points_won[i] < p1_points_won[i - 1] or p2_points_won[i] < p2_points_won[i - 1]):
            fit(pre, i) 
            pre = i
            # exit()
fit(pre, len(p1_points_won))

## p1_denoised - p2_denoised = data_denoised
## p1_denoised + p2_denoised = p1_points_won + p2_points_won
p1_denoised = (p1_points_won + p2_points_won + data_denoised) / 2.0
p2_denoised = (p2_points_won + p2_points_won - data_denoised) / 2.0

with open('output.txt', 'w') as f:
    for i in range(len(p1_denoised)):
        print(p1_denoised[i], p2_denoised[i], sep=',', file=f)