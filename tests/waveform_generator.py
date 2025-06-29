import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile

# --- EASY SETTINGS YOU CAN CHANGE ---

# 1. Increase the peaks: Try values like 1.5, 2.0, etc.
#    Be careful, if you make it too high, the peaks will get flattened ("clipped").
AMPLIFICATION_FACTOR = 1.5 

# 2. Make the plot wider: The first number is the width in inches. 
#    A standard monitor is ~15-20 inches wide. Let's make it extra wide.
PLOT_WIDTH = 25
PLOT_HEIGHT = 6

# --- END OF SETTINGS ---


def plot_simple_waveform(file_path):
    """
    Plots a waveform with simple amplification and a wide figure, as requested.
    """
    try:
        # --- Step 1: Read the audio file (same as before) ---
        temp_wav_path = None
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.mp3':
            sound = AudioSegment.from_mp3(file_path)
            temp_wav_path = "temp_waveform_audio.wav"
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

        # --- Step 2: Normalize the data to a standard range first ---
        # (This is still a good idea so amplification is consistent)
        max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
        normalized_data = data.astype(np.float32) / max_val
        
        # --- Step 3: APPLY YOUR SUGGESTIONS ---

        # 1. "Increase the peaks"
        amplified_data = normalized_data * AMPLIFICATION_FACTOR
        
        # Create the time axis
        n_samples = len(amplified_data)
        time = np.linspace(0, n_samples / sample_rate, num=n_samples)

        # 2. "Assign more width to the plot window"
        plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
        
        plt.plot(time, amplified_data)
        plt.title(f'Waveform of blessthefall - Exodus vocals')
        plt.ylabel(f'Amplified Amplitude (by {AMPLIFICATION_FACTOR}x)')
        plt.xlabel('Time (s)')
        plt.grid(True)
        
        print("Displaying the plot. Close the plot window to end the script.")
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    audio_file_path = "inputs/btf_exodus_vocals.mp3"
    plot_simple_waveform(audio_file_path)