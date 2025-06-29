import os
import sys

# --- Path setup to import RMVPE from the sibling RMVPE folder ---

# Get the directory where the current script (in 'resonance') is located.
# e.g., /path/to/your_project_parent_directory/resonance/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory.
# e.g., /path/to/your_project_parent_directory/
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Define the name of the folder containing 'rmvpe.py'
RMVPE_FOLDER_NAME = "RMVPE"

# Construct the full path to the RMVPE directory.
# e.g., /path/to/your_project_parent_directory/RMVPE/
RMVPE_DIR = os.path.join(PARENT_DIR, RMVPE_FOLDER_NAME)

# Add the RMVPE directory to Python's system path so it can find the module.
# We insert it at the beginning to give it priority, just in case of name conflicts.
if RMVPE_DIR not in sys.path:
    sys.path.insert(0, RMVPE_DIR)

import librosa
import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

import constants

try:
    from rmvpe import RMVPE  # Python will now look for rmvpe.py in RMVPE_DIR
    print(f"Successfully imported RMVPE from: {os.path.join(RMVPE_DIR, 'rmvpe.py')}")
except ModuleNotFoundError:
    print(f"Error: Could not find the 'rmvpe' module in the directory: {RMVPE_DIR}")
    print("Please ensure that:")
    print(f"1. The directory '{RMVPE_DIR}' exists.")
    print(f"2. It contains a file named 'rmvpe.py' (or an '__init__.py' if 'rmvpe' is a package).")
    print(f"3. The directory structure is as expected (RMVPE and resonance are siblings).")
    raise
except ImportError as e:
    print(f"Error importing 'RMVPE' from the 'rmvpe' module: {e}")
    print(f"This might indicate an issue within 'rmvpe.py' itself or its own internal dependencies.")
    raise

def get_f0(audio_path):
    """
    Extracts the raw pitch (F0) from an audio file using RMVPE.
    This method is the simplest way to get the raw pitch contour.
    """

    audio, sr = librosa.load(audio_path, sr=constants.SAMPLING_RATE, mono=True)
    print(f"Audio shape: {audio.shape}, duration: {len(audio) / sr:.2f} seconds")

    rmvpe = RMVPE(f"{RMVPE_DIR}/rmvpe.pt", is_half=False, device="cuda" if torch.cuda.is_available() else "cpu")

    f0 = rmvpe.infer_from_audio(audio, thred=0.06)  # suppress weak salience

    return f0, audio

def get_f0_for_analysis(audio_path):
    """
    Extracts and processes the F0 contour for analysis, applying various filters
    and cleaning steps to prepare the pitch contour for further analysis or synthesis.
    """

    audio, sr = librosa.load(audio_path, sr=constants.SAMPLING_RATE, mono=True)
    print(f"Audio shape: {audio.shape}, duration: {len(audio) / sr:.2f} seconds")

    rmvpe = RMVPE(f"{RMVPE_DIR}/rmvpe.pt", is_half=False, device="cuda" if torch.cuda.is_available() else "cpu")

    f0 = rmvpe.infer_from_audio(audio, thred=0.06)  # suppress weak salience

    # Silence gating based on frame energy ---
    hop_length = constants.HOP_LENGTH
    frame_count = len(f0)
    frame_energy = np.array([
        np.mean(audio[i * hop_length : (i + 1) * hop_length] ** 2)
        for i in range(frame_count)
    ])
    norm_energy = frame_energy / np.max(frame_energy)
    energy_threshold = 0.02
    f0[norm_energy < energy_threshold] = 0  # mute low-energy frames

    # Median filter to smooth out spikes
    f0 = median_filter(f0, size=3)

    # Outlier removal: suppress pitch jumps from local median.
    # Apply a median filter again to smooth the F0 contour
    # and remove outliers based on a deviation threshold.
    f0_median = median_filter(f0, size=5)
    diff = np.abs(f0 - f0_median)
    f0[diff > (f0_median * 0.5)] = 0  # 50% deviation cutoff

    # Hybrid spike detection: high values or big jumps.
    # 900 Hz is a hard limit for F0 in vocals.
    # This is a heuristic to catch hard spikes that are likely artifacts.
    f0_upper_limit = 900  # Hz
    hard_spike_mask = f0 > f0_upper_limit

    # Also catch sudden jumps both forward and backward
    # by checking the difference between adjacent frames.
    # This helps catch abrupt changes that are not typical in vocal pitch.
    delta_forward = np.abs(np.roll(f0, -1) - f0)
    delta_backward = np.abs(f0 - np.roll(f0, 1))
    delta_combined = np.maximum(delta_forward, delta_backward)

    # Define a threshold for what constitutes a "jump"
    # This threshold can be adjusted based on the expected pitch range.
    jump_mask = delta_combined > 400
    combined_mask = hard_spike_mask | jump_mask

    # Apply the combined mask to set these frames to zero
    f0[combined_mask] = 0
    
    return f0, audio

if __name__ == "__main__":
    audio_path = "inputs/fa-vocals.mp3"
    f0, audio = get_f0_for_analysis(audio_path)
    sr = constants.SAMPLING_RATE

    debug_f0 = False

    if debug_f0:
        # Use this pattern to investigate pitch spikes
        # The formual for calculating the timestamp is:
        # timestamp = frame_index * hop_length / sr
        # So the example below shows the pitch spikes between 60s and 65s
        print("\n🔍 Inspecting pitch spikes in frames 6000–6500:")
        for i in range(6000, 6500):
            print(f"Frame {i} | F0: {f0[i]:.2f} Hz")

        print("\n🔍 Max pitch in frames 6000–6500:")
        max_idx = np.argmax(f0[6000:6500]) + 6000
        print(f"Frame {max_idx} | F0: {f0[max_idx]:.2f} Hz")

        plt.figure(figsize=(12, 4))
        plt.plot(f0[6000:6500], label="F0 (Hz)")
        plt.plot(f0, label="F0 (Hz)")
        plt.title("Pitch (F0) of Breaking Benjamin - Dance with the Devil")
        plt.xlabel("Frame")
        plt.ylabel("Frequency (Hz)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Save voiced mask before interpolation
    voiced_mask = f0 > 0

    # Interpolate over gaps for smoother synthesis
    if np.any(voiced_mask):
        f0_interp = np.interp(np.arange(len(f0)), np.where(voiced_mask)[0], f0[voiced_mask])
    else:
        f0_interp = np.zeros_like(f0)

    # Resample f0 to match audio sample length
    # In practice, this means that the F0 contour will be resampled
    # to match the length of the audio signal, allowing for a direct
    # correspondence between the audio frames and the pitch contour.
    # This is important for synthesis, as it ensures that the pitch contour
    # aligns correctly with the audio signal.
    f0_resampled = np.interp(
        np.linspace(0, len(f0_interp), len(audio)),
        np.arange(len(f0_interp)),
        f0_interp
    )

    # Reapply voiced mask to prevent interpolation from reviving silence
    # Explanation: We resample the voiced mask to match the audio length
    # and then apply it to the resampled f0 to ensure that only voiced parts
    # are retained in the final output.
    # This prevents any interpolation artifacts from affecting unvoiced parts.
    # In practice, this means that if a frame was unvoiced,
    # it will remain unvoiced in the final output.
    voiced_mask_resampled = np.interp(
        np.linspace(0, len(voiced_mask), len(audio)),
        np.arange(len(voiced_mask)),
        voiced_mask.astype(float)
    ) > 0.5
    f0_resampled[~voiced_mask_resampled] = 0

    # Generate sine wave only in voiced parts
    # Explanation: We generate a sine wave based on the resampled F0 contour.
    # The sine wave is generated by integrating the frequency over time,
    # which gives us the phase of the sine wave.
    synth = np.zeros_like(f0_resampled)
    phase = 0.0
    for i in range(1, len(f0_resampled)):
        if f0_resampled[i] > 0:
            phase += 2 * np.pi * f0_resampled[i] / sr
            synth[i] = 0.5 * np.sin(phase)
        else:
            phase = 0  # Cut cleanly

    # Apply fade at waveform level.
    # This function applies a fade-in and fade-out effect
    # to the waveform based on the voiced mask.
    # This prevents abrupt starts and stops in the audio,
    # which can create clicks or pops in the synthesized audio.
    def apply_waveform_fade(signal, mask, sr, fade_ms=10):
        fade_len = int(sr * fade_ms / 1000)
        signal = signal.copy()

        diff = np.diff(mask.astype(int), prepend=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for s in starts:
            end = min(s + fade_len, len(signal))
            signal[s:end] *= np.linspace(0, 1, end - s)

        for e in ends:
            start = max(e - fade_len, 0)
            signal[start:e] *= np.linspace(1, 0, e - start)

        return signal

    synth = apply_waveform_fade(synth, voiced_mask_resampled, sr)

    # Save output
    output_path = f"{audio_path}_clean.wav"
    sf.write(output_path, synth, sr)
    print(f"Done. Output saved to {output_path}")

