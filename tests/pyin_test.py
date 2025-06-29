import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Basic piyn test.

audio_path = 'inputs/dwtd_vocals.mp3'
y, sr = librosa.load(audio_path)

f0, voiced_flag, voiced_prob = librosa.pyin(
    y, 
    fmin=75, 
    fmax=600, 
    sr=sr, 
    frame_length=1024, 
    hop_length=256
)

f0[~voiced_flag] = 0.0
f0_clean = np.where(np.isnan(f0), 0, f0)

f0_smooth = median_filter(f0_clean, size=3)

times = librosa.times_like(f0, sr=sr, hop_length=256)
plt.figure(figsize=(10, 4))
plt.plot(times, f0_smooth, label="Smoothed Pitch (Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Continuous Pitch Contour from Vocal Track")
plt.legend()
plt.tight_layout()
plt.show()

duration = len(y) / sr
t_full = np.linspace(0, duration, len(y))
f0_interp = np.interp(t_full, times, f0_smooth)

phase = 2 * np.pi * np.cumsum(f0_interp) / sr

synthesized_audio = 0.5 * np.sin(phase)

output_file = 'outputs/result_pyin.wav'
sf.write(output_file, synthesized_audio, sr)
print(f"Synthesized audio saved to {output_file}")