import paramiko
import os

import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import spectrogram



def download_data():
    # Configuración de la conexión SFTP

    hostname = '147.83.159.131'
    port = 25422


    username = 'root'
    password = 'evologics'
    remote_dir = "/usr/local/bin/sdmsh/files_tank/"
    local_dir = "C:/Users/GERARD/Desktop/programes/ProgPycharm/USBL"  # Directorio local para almacenar los archivos descargados

    # Crear el directorio local si no existe
    os.makedirs(local_dir, exist_ok=True)

    try:
        # Establecer conexión SSH y SFTP
        transport = paramiko.Transport((hostname, port))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Listar archivos en el directorio remoto y descargar los .dat
        for filename in sftp.listdir(remote_dir):
            if filename.endswith('.dat'):
                remote_path = os.path.join(remote_dir, filename)
                local_path = os.path.join(local_dir, filename)
                print(f"Descargando {filename} ...")
                sftp.get(remote_path, local_path)
                print(f"{filename} descargado con éxito.")

        # Cerrar la conexión SFTP
        sftp.close()
        transport.close()
        print("Descarga completada.")
    except Exception as e:
        print(f"Error: {e}")
    return None

def plot_signals():
    # Sampling rate (250 kHz)
    fs_original = 250000  # 250 kHz

    # Load all four hydrophone signals
    files = ['usblch0.dat', 'usblch1.dat', 'usblch2.dat', 'usblch3.dat', 'usblch4.dat']
    signals = [np.loadtxt(file) for file in files]

    # Create a time axis based on the first signal length
    t_total_original = len(signals[0]) / fs_original  # Duration in seconds
    t_original = np.linspace(0, t_total_original, len(signals[0]))  # Time vector

    # Set up the figure with subplots
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex = True)  # 4 rows (one for each signal), 2 columns (time plot & spectrogram)

    for i, (signal, file) in enumerate(zip(signals, files)):
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs_original, nperseg=512, noverlap=256, scaling='density')

        # Time-domain plot
        axes[i, 0].plot(t_original, signal, label=file, color='b')
        axes[i, 0].set_xlabel("Time (seconds)")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_title(f"Time-Domain Signal: {file}")
        axes[i, 0].legend()
        axes[i, 0].grid(True)

        # Spectrogram plot
        pcm = axes[i, 1].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')  # Convert to dB scale
        axes[i, 1].set_xlabel("Time (seconds)")
        axes[i, 1].set_ylabel("Frequency (Hz)")
        axes[i, 1].set_title(f"Spectrogram: {file}")
        fig.colorbar(pcm, ax=axes[i, 1], label="Power (dB)")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
    return None



if __name__ == "__main__":

    download_data()
    plot_signals()
    os.system('python stft.py')

