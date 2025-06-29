import numpy as np
import soundfile as sf

# Script to generate audio files for all major and minor scales in all keys.
# This script synthesizes sine waves for each note in the scale and saves them as .wav files.
# Used for testing and demonstration purposes and to audibly understand musical scales.

sample_rate = 44100  
note_duration = 0.5
note_gap = 0.05  # seconds of silence between notes

note_names = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 
              5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}

major_intervals = [0, 2, 4, 5, 7, 9, 11]

# Use natural minor intervals for simplicity
minor_intervals = [0, 2, 3, 5, 7, 8, 10, 11]

def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12))

def generate_sine_wave(freq, duration, sr, fade_duration=0.01):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)

    # Apply fade in/out to avoid clicks
    fade_samples = int(fade_duration * sr)
    envelope = np.ones_like(wave)

    # Fade-in
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)

    # Fade-out
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    return wave * envelope

def synthesize_scale(root, intervals, base_midi=60):
    # Calculate tonic MIDI note: 
    # For instance, for C (root=0) we keep it as 60 (C4). For D (root=2), we move up to 62 (D4).
    tonic = base_midi + ((root - (base_midi % 12)) % 12)

    # Create the scale: add each interval to the tonic.
    midi_notes = [tonic + offset for offset in intervals]
    return midi_notes


for root in range(12):
    for scale_type, intervals in zip(["Major", "Minor"], [major_intervals, minor_intervals]):
        scale_name = f"{note_names[root]} {scale_type}"
        midi_notes = synthesize_scale(root, intervals)

        print(midi_notes)
        
        silence = np.zeros(int(note_gap * sample_rate)) # audio silence for gap between notes
        scale_audio = np.concatenate([
            np.concatenate([generate_sine_wave(midi_to_freq(midi), note_duration, sample_rate), silence])
            for midi in midi_notes
        ])

        filename = f"scales/{scale_name.replace(' ', '_')}.wav"
        sf.write(filename, scale_audio, sample_rate)
        print(f"Saved {filename} with MIDI notes: {midi_notes}")
