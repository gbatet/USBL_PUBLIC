import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Define parameters
fs = 250000  # Sampling frequency in Hz
duration = 1024 / fs  # Duration of chirp in seconds (1024 samples)
f0 = 67700  # Start frequency in Hz
f1 = 68300  # End frequency in Hz
amplitude = 1  # Amplitude (0 to 1)
filename = "4ms_68khz.bin"  # Output filename

# Generate time vector
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate chirp signal
y = amplitude * signal.chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

# Scale to 16-bit integer range
y_int16 = np.int16(y * 32767)

# Save as 16-bit little-endian binary file
y_int16.tofile(filename)

# Plot the chirp signal
plt.figure(figsize=(10, 4))
plt.plot(t[:1000], y_int16[:1000])  # Plot only the first 1000 samples for clarity
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Chirp Signal")
plt.grid()
plt.show()
