from scipy.signal import stft
import numpy as np

def compute_band_power(signal, srate, bands):
    
    # STFT parameters
    nperseg = int(0.5 * srate) # 500ms window
    f, t, Zxx = stft(signal, fs=srate, nperseg=nperseg)
    
    # Calculating power from complex voltage
    power = np.abs(Zxx)**2
    
    # Average power into bands
    band_powers = []
    
    for band_name, (low, high) in bands.items():
        # Finding indices of band frequencies
        idx = np.where((f >= low) & (f <= high))[0]
        
        # Band power calculation
        avg_power = np.mean(power[idx, :], axis=0)
        band_powers.append(avg_power)
        
    return np.array(band_powers)