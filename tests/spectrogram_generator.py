import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile

# NFFT: The "resolution" of the spectrogram. Higher values give more detail
# in the frequencies (vertical axis) but less detail in time (horizontal axis).
# A power of 2 is recommended (e.g., 1024, 2048, 4096).
NFFT_VALUE = 2048

# Y_SCALE: The scale of the frequency axis. 'log' is usually best for music
# as it mimics how our ears perceive pitch. Can also be 'linear'.
Y_AXIS_SCALE = 'log'

def get_normalized_audio_data(file_path):
    """
    Reads an MP3 or WAV file and returns a normalized mono audio signal
    and the sample rate.
    """
    temp_wav_path = None
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.mp3':
        sound = AudioSegment.from_mp3(file_path)
        temp_wav_path = "temp_spectrogram_audio.wav"
        sound.export(temp_wav_path, format="wav")
        read_path = temp_wav_path
    elif file_extension == '.wav':
        read_path = file_path
    else:
        raise ValueError(f"Unsupported file format '{file_extension}'")

    sample_rate, data = wavfile.read(read_path)

    if temp_wav_path:
        os.remove(temp_wav_path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
    normalized_data = data.astype(np.float32) / max_val
    
    return normalized_data, sample_rate


def plot_spectrogram(file_path):
    """
    Generates and displays a spectrogram for the given audio file.
    """
    signal, sample_rate = get_normalized_audio_data(file_path)

    plt.figure(figsize=(15, 6))

    Pxx, freqs, bins, im = plt.specgram(
        signal,
        NFFT=NFFT_VALUE, # Number of data points used in each block for the FFT, in essence this is the "resolution" of the spectrogram, "resolution" being the detail in the frequency axis.
        Fs=sample_rate, # Sampling frequency
        noverlap=NFFT_VALUE // 2, # 50% overlap is common
        cmap='inferno', # You can change this to 'viridis', 'plasma', etc.
        scale='dB'  # Set scale to 'dB' for intensity because it is common in audio analysis
    )
    
    # Apply the logarithmic scale to the Y-axis
    # This is because the human ear perceives sound intensity logarithmically.
    if Y_AXIS_SCALE == 'log':
        plt.yscale('log')

    plt.title(f'Spectrogram of {os.path.basename(file_path)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    cbar = plt.colorbar(im)
    cbar.set_label('Intensity (dB)')
    
    # Set a practical frequency limit for Y-axis if needed
    if Y_AXIS_SCALE == 'log':
            plt.ylim(20, sample_rate / 2)
    
    print("Displaying the plot. Close the plot window to end the script.")
    plt.show()


if __name__ == '__main__':
    audio_file_path = "inputs/dwtd_vocals.mp3_clean.wav"
    plot_spectrogram(audio_file_path)