import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import stft


def load_data(filepaths, start_time=0.07, end_time=0.2, sampling_rate=250000):
    """
    Load data from given file paths, only including the time range from start_time to end_time.

    Args:
        filepaths (list): List of file paths to load data from.
        start_time (float): The start time in seconds (default is 0.07).
        end_time (float): The end time in seconds (default is 0.12).
        sampling_rate (int): The sampling rate in Hz (default is 250000).

    Returns:
        list: List of signals with data between start_time and end_time.
    """
    signals = []

    # Calculate start and end sample indices
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)

    for path in filepaths:
        # Load the entire signal
        signal = np.loadtxt(path)

        # Slice the signal to only include the time range of interest
        sliced_signal = signal[start_sample:end_sample]

        signals.append(sliced_signal)

    return signals


def oversample_signal(signals, original_rate, target_rate):
    """
    Oversample signals to a higher sampling rate using cubic interpolation.
    """
    oversampled_signals = []
    target_time = None
    # Interpolate each signal using cubic spline
    for signal in signals:
        original_time = np.arange(len(signal)) / original_rate
        target_time = np.linspace(0, len(signal) / original_rate, int(len(signal) * (target_rate / original_rate)))
        interpolator = interp1d(original_time, signal, kind='cubic', fill_value="extrapolate")
        oversampled_signal = interpolator(target_time)
        oversampled_signals.append(oversampled_signal)

    return oversampled_signals, target_time


def find_frequency_over_time(signals, fs, target_freq, window_length, overlap):
    """
    Find the frequency value for a given target frequency over time using the Short-Time Fourier Transform (STFT).

    Parameters:
    - signals: List of input signals.
    - fs: Sampling rate of the signal.
    - target_freq: The target frequency to extract.
    - window_length: Length of each FFT window.
    - overlap: Number of samples by which adjacent windows overlap.

    Returns:
    - t: Time points for each signal (aligned time bins).
    - target_freq_magnitude: Magnitude of the target frequency over time for each signal.
    """
    t = None
    target_freq_magnitude = []

    for signal in signals:
        # Perform STFT
        f, ti, Zxx = stft(signal, fs=fs, nperseg=window_length, noverlap=overlap)

        if t is None:
            t = ti  # Use the time bins from the first signal

        # Find the index of the target frequency
        target_idx = np.argmin(np.abs(f - target_freq))

        # Extract magnitude of the target frequency over time
        magnitude = np.abs(Zxx[target_idx, :])

        # Normalize the magnitude between 0 and 1
        magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

        target_freq_magnitude.append(magnitude_normalized)

    return t, target_freq_magnitude


def find_toa_from_magnitude(t, target_freq_magnitude, threshold=0.5):
    """
    Find the Time of Arrival (TOA) for a given target frequency magnitude using a simple thresholding method.

    Parameters:
    - t: Time bins of the STFT.
    - target_freq_magnitude: Magnitude of the target frequency over time.
    - threshold: The magnitude threshold above which the signal is considered to have arrived.

    Returns:
    - toa: The time when the target frequency first exceeds the threshold.
    """
    toas = []

    for magnitude in target_freq_magnitude:
        # Find the first time the magnitude exceeds the threshold
        above_threshold_indices = np.where(magnitude > threshold)[0]

        if len(above_threshold_indices) > 0:
            # TOA is the first time point when the target frequency exceeds the threshold
            toa = t[above_threshold_indices[0]]
        else:
            # If no time exceeds the threshold, return None
            toa = None

        toas.append(toa)

    return toas


def plot_frequency_over_time(t, magnitude, target_freq, toa=None, ax=None):
    """
    Plot the frequency magnitude over time for a given target frequency.

    Parameters:
    - t: Time points.
    - magnitude: Magnitude of the target frequency.
    - target_freq: The target frequency.
    - toa: The time of arrival (optional).
    - ax: Axis object to plot on (for subplots).
    """
    if ax is None:
        ax = plt.gca()  # Get current axis if not provided

    ax.plot(t, magnitude, label=f"Frequency: {target_freq} Hz")

    # If TOA is provided, plot it
    if toa is not None:
        ax.axvline(toa, color='r', linestyle='--', label=f"TOA: {round(toa,6)} s")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True)


def calculate_doa(tdoa, distance, speed_of_sound=1500):
    """
    Calculate the Direction of Arrival (DOA) from TDOA (Time Difference of Arrival).

    Args:
        tdoa (float): Time difference of arrival (in seconds).
        distance (float): The distance between the two hydrophones (in meters).
        speed_of_sound (float, optional): Speed of sound in water (in meters per second). Default is 1500 m/s.

    Returns:
        float: Direction of Arrival in radians (if valid).
        None: If the input is outside the valid range for arcsin.
    """
    try:
        # Calculate the argument for arcsin
        arg = (tdoa * speed_of_sound) / distance

        # Clamp the value to be within the valid range for arcsin [-1, 1]
        arg = np.clip(arg, -1.0, 1.0)

        # Calculate the DOA (Direction of Arrival) using arcsin
        doa = np.arcsin(arg)
        return doa
    except ValueError:
        # Return None if there is an issue with the calculation
        return None


def calculate_doa_for_pairs(toas, distances, speed_of_sound=1500):
    """
    Calculate the Direction of Arrival (DOA) for two pairs of hydrophones.
    """
    # Calculate TDOA for North-South (0, 2) and East-West (1, 3)
    tdoa_ns = toas[2] - toas[0]  # TDOA between hydrophone 0 and 2
    tdoa_ew = toas[1] - toas[3]  # TDOA between hydrophone 1 and 3

    # Calculate DOA for both pairs
    doa_ns = calculate_doa(tdoa_ns, distances[0], speed_of_sound)
    doa_ew = calculate_doa(tdoa_ew, distances[1], speed_of_sound)

    # Calculate bearing (azimuth) using atan2
    azimuth_rad = np.arctan2(doa_ew, doa_ns)  # atan2(y, x)
    azimuth_deg = np.degrees(azimuth_rad)  # Convert radians to degrees

    return azimuth_deg

def plot_bearing(bearing):
    """
    Plot the azimuth (bearing) on a polar plot as a line from the center,
    and label the NSEW directions with corresponding hydrophone channels.
    The angular ticks are removed, leaving only the lines for cardinal and 45-degree increments.
    """
    # Convert bearing to radians
    bearing_rad = np.radians(bearing)

    # Create polar plot
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')  # Set 0° at North (default is East)
    ax.set_theta_direction(-1)  # Set clockwise direction for azimuth angles

    # Plot the bearing as a line extending from the center (0, 0) to the bearing angle
    ax.plot([0, bearing_rad], [0, 1], linestyle='-', linewidth=2, color='r')  # Red line
    ax.set_ylim(0, 1)  # Set radius limit

    # Add NSEW labels with channel numbers (adjusted positions to avoid overlap)
    directions = ["0","","3","","2","","1",""]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    label_distance = 1.05  # Moved slightly further away to avoid overlap

    for angle, direction in zip(angles, directions):
        ax.text(angle, label_distance, direction, fontsize=12, ha='center', va='center')

    # Remove angular ticks (i.e., 0°, 90°, 180°, 270° labels)
    ax.set_xticks([])  # Remove angular ticks
    ax.set_rticks([])  # No radial ticks

    # Draw lines at the NSEW directions and 45-degree increments
    for angle in angles:
        ax.plot([angle, angle], [0, 1], linestyle='--', color='gray', linewidth=1)

    # Title and show plot
    ax.set_title(f'Bearing: {bearing:.2f}°', pad=20)

#########################################
# VARIABLES
#########################################

filepaths = ["usblch0.dat", "usblch1.dat", "usblch2.dat", "usblch3.dat"]

fs_original = 250000        # 250ks/s
fs_target = 1000000         # 1Ms/s
target_freq = 69000         # 69 kHz

#####
# Distances between pairs (1,3 and 0,2)
#####
distances = [0.095, 0.095]

#####
#STFT params
#####

window_length = 64          # Length of the FFT window
overlap = 56                # Number of samples by which adjacent windows overlap ( 1 STFT every 8 samples)
threshold = 0.025            # Example threshold (normalized)

#########################################
# LOAD DATA
#########################################

signals = load_data(filepaths)

#########################################
# OVERSAMPLE
#########################################

oversampled_signals, time_axis = oversample_signal(signals, fs_original, fs_target)

#########################################
# STFT
#########################################

t, magnitude = find_frequency_over_time(oversampled_signals, fs_target, target_freq, window_length, overlap)

#########################################
# TOA (Simple Threshold)
#########################################

toas = find_toa_from_magnitude(t, magnitude, threshold)
print("TOAs:", toas)

#########################################
# DOA
#########################################

bearing = calculate_doa_for_pairs(toas,distances)

#########################################
# PLOTS
#########################################

# Create a figure and a set of subplots with shared x and y axes
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True, sharey=True)

# Plot each signal in a separate subplot
for i in range(len(oversampled_signals)):
    plot_frequency_over_time(t, magnitude[i], target_freq, toa=toas[i], ax=axes[i])

plot_bearing(bearing)

plt.tight_layout()
plt.show()
