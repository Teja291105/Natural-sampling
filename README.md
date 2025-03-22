**EXP NO : 2 Natural-Sampling**

**Aim :**
To implement Natural Sampling using Python, visualize the sampled signal, and reconstruct it using a low-pass filter.

**Tools required :**
COLAB(python)

**Program :**
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 1000 
T = 1 
t = np.arange(0, T, 1/fs) 

fm = 5  
message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50 
pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i + pulse_width] = 1 

nat_signal = message_signal * pulse_train

sampled_signal = nat_signal[pulse_train == 1]

sample_times = t[pulse_train == 1]

reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    if index + pulse_width < len(t): 
        reconstructed_signal[index:index + pulse_width] = sampled_signal[i]
    else:
        reconstructed_signal[index:] = sampled_signal[i]

def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

**Output Waveform:**
![image](https://github.com/user-attachments/assets/fe3aeebf-ad67-4dc3-90ba-10575e70f668)

**Results:**
i) Natural Sampling is obtained successfully.

ii) The generated plots include:
    Original Message Signal (Sine wave).
    Pulse Train (Square wave train).
    Naturally Sampled Signal (Message signal multiplied with pulse train).
    Reconstructed Signal (Filtered output to approximate the original signal).
