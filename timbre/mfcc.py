import numpy as np
import librosa
import constants
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pitch_extraction_new import get_f0_for_analysis

def extract_mfcc(audio, sr, n_mfcc, n_fft, hop_length=constants.HOP_LENGTH):
    """
    Extracts MFCC features from an audio signal. Uses librosa to compute the MFCCs.
    Args:
        audio (np.ndarray): The audio signal from which to extract MFCCs.
        sr (int): Sampling rate of the audio.
        n_mfcc (int): Number of MFCCs to extract.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.
    Returns:
        np.ndarray: The extracted MFCCs, shape (n_mfcc, num_frames).
    """
    if audio is None or len(audio) == 0:
        return None
    
    mfccs = librosa.feature.mfcc(y=audio, 
                                 sr=sr, 
                                 n_mfcc=n_mfcc, 
                                 n_fft=n_fft,
                                 hop_length=hop_length, 
                                 win_length=n_fft)
    return mfccs

if __name__ == "__main__":
    f0, audio = get_f0_for_analysis("inputs/btf_exodus_guitar.mp3")
    sr = constants.SAMPLING_RATE

    # Should probably make these constants but for now will remain flexible.
    n_fft = 400
    n_mfcc = 13

    mfccs = extract_mfcc(audio=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    if mfccs is not None:
        print(f"Extracted MFCCs with shape: {mfccs.shape}")
        print("First 5 frames of MFCCs:")
        print(mfccs[:, :5])
    else:
        print("No MFCCs extracted, check the audio input.")