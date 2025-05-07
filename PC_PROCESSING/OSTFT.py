import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import stft

from stft import oversampled_signals


def acquire_data(filepaths, start_time=0.07, end_time=0.2, fs=250000):
    """ Load data from given file paths. """
    start_sample, end_sample = int(start_time * fs), int(end_time * fs)
    signals = [np.loadtxt(path)[start_sample:end_sample] for path in filepaths]
    return signals, fs


def process_data(signals, fs, target_fs=1000000, target_freq=69000, window_length=64, overlap=56, threshold=0.02,
                 distances=[0.095, 0.095], oversampling = 1):
    """ Perform oversampling, STFT, TOA extraction, and DOA calculation. """

    if oversampling:
        oversampled_signals = []
        for signal in signals:
            t_original = np.arange(len(signal)) / fs
            t_target = np.linspace(0, len(signal) / fs, int(len(signal) * (target_fs / fs)))
            oversampled_signals.append(interp1d(t_original, signal, kind='cubic', fill_value="extrapolate")(t_target))
    else:
        oversampled_signals = signals
        target_fs = fs

    t, magnitude = None, []
    for signal in oversampled_signals:
        f, t_stft, Zxx = stft(signal, fs=target_fs, nperseg=window_length, noverlap=overlap)
        if t is None:
            t = t_stft
        target_idx = np.argmin(np.abs(f - target_freq))
        mag = np.abs(Zxx[target_idx, :])
        magnitude.append((mag - np.min(mag)) / (np.max(mag) - np.min(mag)))

    toas = [t[np.where(mag > threshold)[0][0]] if np.any(mag > threshold) else None for mag in magnitude]

    tdoa_ns, tdoa_ew = toas[2] - toas[0], toas[1] - toas[3]
    doa_ns = np.arcsin(np.clip((tdoa_ns * 1500) / distances[0], -1, 1))
    doa_ew = np.arcsin(np.clip((tdoa_ew * 1500) / distances[1], -1, 1))

    bearing = np.degrees(np.arctan2(doa_ew, doa_ns))
    return t, magnitude, toas, bearing


def plot_results(t, magnitude, toas, target_freq, bearing):
    """ Plot STFT results and bearing. """
    fig, axes = plt.subplots(nrows=4, figsize=(10, 12), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(t, magnitude[i], label=f"{target_freq} Hz")
        if toas[i] is not None:
            ax.axvline(toas[i], color='r', linestyle='--', label=f"TOA: {toas[i]:.6f}s")
        ax.legend(), ax.grid()

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N'), ax.set_theta_direction(-1)
    ax.plot([0, np.radians(bearing)], [0, 1], linestyle='-', linewidth=2, color='r')
    ax.set_title(f'Bearing: {bearing:.2f}°')
    plt.show()


# Execution
filepaths = ["usblch0.dat", "usblch1.dat", "usblch2.dat", "usblch3.dat"]
signals, fs_original = acquire_data(filepaths)
t, magnitude, toas, bearing = process_data(signals, fs_original)
plot_results(t, magnitude, toas, 69000, bearing)
