import numpy as np
import librosa
import sys
import os
import constants

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pitch_extraction_new import get_f0_for_analysis

def extract_vocal_timbre_features(y, sr, n_fft, n_mfcc):
    """
    Extracts a rich set of timbre-related features from a vocal audio signal.
    This returns features per frame, not averaged.
    Args:
        y (np.ndarray): The audio signal from which to extract features.
        sr (int): Sampling rate of the audio.
        n_fft (int): Length of the FFT window.
        n_mfcc (int): Number of MFCCs to extract.
    """
    if y is None or len(y) == 0:
        return None
    

    hop_length = constants.HOP_LENGTH
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    features = np.vstack([
        mfccs,
        spectral_centroid,
        spectral_rolloff,
        spectral_contrast
    ]).T

    return features[~np.isnan(features).any(axis=1)]

if __name__ == "__main__":
    f0, audio = get_f0_for_analysis("inputs/telescope_vocals.mp3")
    sr = constants.SAMPLING_RATE

    # Should probably make these constants but for now will remain flexible.
    n_fft = 400
    n_mfcc = 13

    features = extract_vocal_timbre_features(audio, sr, n_fft, n_mfcc)
    if features is not None:
        print(f"Extracted {features.shape[0]} frames of vocal timbre features.")
        print("First 5 frames of features:")
        print(features[:5])
    else:
        print("No features extracted, check the audio input.")

