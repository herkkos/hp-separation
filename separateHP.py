# -*- coding: utf-8 -*-
"""
Herkko Salonen
5.3.2019

All rights reserved
"""

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.signal import windows
from librosa.core import stft, istft
from librosa.display import specshow


# Separation algorithm
def separateHP(filename, winlength, alpha=0.5, gamma=0.4):
    # Open and read the file
    fs, y = read(filename)
    
    # Calculate the STFT i.e. transform to time-frequency domain
    winlen = int(float(winlength)/1000.0 * float(fs))
    hoplen = int(winlen/2)
    y = y / np.max(np.abs(y))
    F = stft(y, n_fft=winlen, hop_length=hoplen, window=windows.hann)
    
    # Calculate range-compressed version of power spectrogram
    W = np.power(np.abs(F), 2*gamma)
    
    # Initialize percussion and harmonic values
    H = W/2
    P = W/2
    
    # Length of transform timewise
    K = np.size(W,1)
    
    # Update P & H
    for i in range(0,K-1):
        # Calculate delta matrix which describes energy changes between frequencies
        delta = np.zeros([hoplen+1, 1])
        for h in range(1, hoplen):
            delta[h,0] = alpha * ( (H[h,i-1] - 2*H[h,i] + H[h,i+1]) /4 ) - (1-alpha) * ( (P[h-1,i] - 2*P[h,i] + P[h+1,i]) /4 )
        
        H[:,i+1] = np.minimum(np.maximum(np.add(H[:,i], delta[:,0]), 0), W[:,i+1])
        P[:,i+1] = np.subtract(W[:,i+1], H[:,i+1])
    
    # Binarize the separation
    for i in range(0, K-1):
        for h in range(0,hoplen):
            if H[h,i-1] < P[h,i-1]:
                H[h,i] = 0
                P[h,i] = W[h,i]
            else:
                H[h,i] = W[h,i]
                P[h,i] = 0
    
    # Visualize P & H
    plt.figure(figsize=[15,5])
    plt.subplot(1,2,1)
    #specshow(20*np.log10(1e-10+np.abs(H)), y_axis='log', x_axis='time',sr=fs,hop_length=hoplen)
    specshow(H, y_axis='linear', x_axis='time',sr=fs,hop_length=hoplen)
    plt.title('H')
    #plt.pcolormesh(H)
    plt.subplot(1,2,2)
    #specshow(20*np.log10(1e-10+np.abs(P)), y_axis='log', x_axis='time',sr=fs,hop_length=hoplen)
    specshow(P,y_axis='linear', x_axis='time',sr=fs,hop_length=hoplen)
    plt.title('P')
    #plt.pcolormesh(P)
    plt.show()
    
    # Convert into waveform
    h = istft(np.power(H, 1/(2*gamma))*np.exp(1j*np.angle(F)), win_length=winlen, hop_length=hoplen, window=windows.hann)
    p = istft(np.power(P, 1/(2*gamma))*np.exp(1j*np.angle(F)), win_length=winlen, hop_length=hoplen, window=windows.hann)
    
    h = h / np.max(np.abs(h))
    p = p / np.max(np.abs(p))
    
    # Save harmonic and percussive audiofiles
    audio_name = '%s-harmonics.wav' % (filename[:-4])
    write(audio_name, fs, h)
    
    audio_name = '%s-percussive.wav' % (filename[:-4])
    write(audio_name, fs, p)

    SNR = 20 * np.log10(np.abs((np.sum(y))/(np.sum(y)-np.sum(p)-np.sum(h))))
    print("SNR for signal p:", SNR, "dB")

if __name__ == "__main__":
    FILENAME = "police03short.wav"
    WINLENGTH = 64
    
    separateHP(FILENAME, WINLENGTH, alpha=0.5, gamma=0.3)
    
