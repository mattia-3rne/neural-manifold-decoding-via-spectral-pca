import numpy as np

def generate_pink_noise(n_samples):

    # Defining frequency domain
    f = np.fft.rfftfreq(n_samples)
    
    # Scaling amplitudes by 1/f
    f[0] = 1e-10 # avoiding zero
    spectral_shape = 1 / np.sqrt(f)
    
    # Generating random phases
    white_noise = np.random.normal(0, 1, len(f)) + 1j * np.random.normal(0, 1, len(f))
    
    # Applying 1/f filter and inverse transform
    S_pink = white_noise * spectral_shape
    pink_noise = np.fft.irfft(S_pink, n=n_samples)
    
    # Normalizing
    return pink_noise - np.mean(pink_noise)

def get_motor_envelope(n_samples, srate, event_time=2.5):
    
    t = np.linspace(0, n_samples/srate, n_samples)
    
    # Resting state
    envelope = np.ones_like(t)
    
    # Event-Related Desynchronization
    center = event_time
    width = 0.5
    erd_depth = 0.6
    
    gaussian = np.exp(-((t - center)**2) / (2 * width**2))
    envelope = envelope - (erd_depth * gaussian)
    
    return envelope

def simulate_trial(duration, srate, trial_type, noise_level=5.0):
    
    n_samples = int(duration * srate)
    t = np.linspace(0, duration, n_samples)
    
    # Background noise
    noise_c3 = generate_pink_noise(n_samples) * noise_level
    noise_c4 = generate_pink_noise(n_samples) * noise_level
    
    # Motor rhythms
    raw_alpha = np.sin(2 * np.pi * 10 * t) # 10 Hz
    raw_beta  = np.sin(2 * np.pi * 20 * t) # 20 Hz
    
    # Lateralization logic
    base_env = get_motor_envelope(n_samples, srate)
    idle_env = np.ones(n_samples)
    
    if trial_type == 'Left':
        env_c3 = idle_env
        env_c4 = base_env
    else: # trial_type == 'Right':
        env_c3 = base_env
        env_c4 = idle_env
        
    # Signal Superposition
    signal_c3 = noise_c3 + env_c3 * (raw_alpha + raw_beta)
    signal_c4 = noise_c4 + env_c4 * (raw_alpha + raw_beta)
    
    return np.vstack([signal_c3, signal_c4])