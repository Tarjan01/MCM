## use DFT to denoise the data
## the optimal threshold is 726, which is 3times of the mean of the absolute value of the frequency
## the plot of the frequency before and after denoised  can show the effect of the denoising
## the output is saved in the output.txt, which contains the denoised data of p1 and p2

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

def res_show():
    original_data = p1_points_won - p2_points_won
    denoised_data = data_denoised

    fft_before = np.fft.fft(original_data)
    fft_after = np.fft.fft(denoised_data)
    freq = np.fft.fftfreq(len(original_data))
    
    print(3*np.abs(fft_before).mean())
    
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(freq, np.abs(fft_before))
    # plt.title('freq_before')
    # plt.subplot(1, 2, 2)
    # plt.plot(freq, np.abs(fft_after))
    # plt.title('freq_after')
    # plt.show()
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.lineplot(x=freq, y=np.abs(fft_before))
    plt.title('freq_before_denoised')
    plt.subplot(1, 2, 2)
    sns.lineplot(x=freq, y=np.abs(fft_after))
    plt.title('freq_after_denosied')
    plt.show()

def fit(start, end):
    p1 = p1_points_won[start:end]
    p2 = p2_points_won[start:end]
    data_fft = fft(p1 - p2)
    threshold = 726
    data_fft_abs = np.abs(data_fft)
    data_fft[data_fft_abs < threshold] = 0
    data_denoised[start:end] = np.real(ifft(data_fft))
    # with open('output.txt', 'w') as f:
    #     print(data_denoised[start:end], file=f)
    
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

res_show()

with open('output.txt', 'w') as f:
    for i in range(len(p1_denoised)):
        print(p1_denoised[i], p2_denoised[i], sep=',', file=f)