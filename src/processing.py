from scipy.signal import stft
import numpy as np

def compute_band_power(signal, srate, bands):
    
    # STFT parameters
    window_sec = 0.5
    nperseg = int(window_sec * srate)
    noverlap = max(0, nperseg - 192)
    f, t, Zxx = stft(signal, fs=srate, nperseg=nperseg, noverlap=noverlap, boundary='even')
    
    # Calculating power from complex voltage
    power = np.abs(Zxx)**2
    
    # Averaging power into bands
    band_powers = []
    
    for band_name, (low, high) in bands.items():
        # Finding indices of band frequencies
        idx = np.where((f >= low) & (f <= high))[0]
        
        # Band power calculation
        avg_power = np.mean(power[idx, :], axis=0)
        band_powers.append(avg_power)
        
    return np.array(band_powers)[:, 2:-2]