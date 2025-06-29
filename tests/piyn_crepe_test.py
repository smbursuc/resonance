import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import median_filter
import crepe

# Simple hybrid pitch tracking using pyin and CREPE
# This script loads an audio file, extracts pitch using both pyin and CREPE,
# fuses the results, and synthesizes a sine wave based on the fused pitch.

audio_path = 'inputs/dwtd_vocals.mp3'
sr_crepe = 16000
frame_length = 2048
hop_length = 256
fmin = 75
fmax = 600

y_orig, sr = librosa.load(audio_path, sr=None, mono=True)
y = y_orig / np.max(np.abs(y_orig))

# Resample for CREPE ---
# CREPE works best at 16kHz, so we resample the audio
y_crepe = librosa.resample(y, orig_sr=sr, target_sr=sr_crepe)

f0_pyin, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=fmin,
    fmax=fmax,
    sr=sr,
    frame_length=frame_length,
    hop_length=hop_length
)
times_pyin = librosa.times_like(f0_pyin, sr=sr, hop_length=hop_length)
f0_pyin = np.where((voiced_prob > 0.9) & ~np.isnan(f0_pyin), f0_pyin, 0)

# CREPE is a neural network model that requires a specific input format
# It expects a mono audio signal and outputs pitch in Hz.
# It predicts pitch and confidence scores based on the audio signal.
time_c, freq_c, conf_c, _ = crepe.predict(y_crepe, sr=sr_crepe, viterbi=True, step_size=100)
# Align time base
f0_crepe = np.where(conf_c > 0.85, freq_c, 0)

# Align both to common time axis (pyin's)
# CREPE outputs are at a different sampling rate, so we need to interpolate
f0_crepe_interp = np.interp(times_pyin, time_c, f0_crepe)

# Fusing the two pitch estimates
# We will use pyin's pitch where it is confident, otherwise use CREPE's pitch
f0_fused = np.where(f0_pyin > 0, f0_pyin, f0_crepe_interp)
f0_fused = median_filter(f0_fused, size=3)

# Interpolate to fill gaps in the fused pitch
duration = len(y) / sr
t_full = np.linspace(0, duration, len(y))
f0_interp = np.interp(t_full, times_pyin, f0_fused)

# Synthesize sine wave
phase = 2 * np.pi * np.cumsum(f0_interp) / sr
synth = 0.5 * np.sin(phase)

# Save output
output_path = "outputs/result_crepe_piyn.wav"
sf.write(output_path, synth, sr)
print(f"Synthesized hybrid pitch audio saved to {output_path}")

# Plot for debugging
plt.figure(figsize=(12, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Waveform')
plt.plot(times_pyin, f0_fused, color='r', label='Hybrid Fused Pitch (Hz)')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude / Frequency (Hz)")
plt.title("Hybrid pyin + CREPE Pitch Tracking")
plt.legend()
plt.tight_layout()
plt.show()
