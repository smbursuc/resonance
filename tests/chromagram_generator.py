import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Script utilized to generate a chromagram from an audio file.
# Useful for visualizing the pitch content of the audio.

def plot_chromagram(file_path):
    """
    Generates and displays a chromagram for the given audio file using librosa.
    """
    y, sr = librosa.load(file_path)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    plt.figure(figsize=(15, 6))
    
    # librosa.display.specshow is specifically designed for this kind of plot.
    # y_axis='chroma' automatically labels the rows with note names.
    # x_axis='time' correctly formats the time axis.
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, cmap='magma') # You can change 'cmap' to 'viridis', 'plasma', etc.
    
    plt.title(f'Chromagram of {os.path.basename(file_path)}')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Class (Note)')
    plt.colorbar(label='Intensity')
    plt.show()

if __name__ == '__main__':
    audio_file_path = "inputs/fa-vocals.mp3"
    plot_chromagram(audio_file_path)