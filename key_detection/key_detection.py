import numpy as np
import librosa
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pitch_extraction_new import get_f0_for_analysis

import constants

def detect_key_from_f0(f0):
    """
    Detects the musical key from an F0 contour.

    Args:
        f0 (np.array): Processed F0 contour (Hz).

    Returns:
        str: The estimated key (e.g., "C Major") or "Unknown" if detection fails.
    """
    if f0 is None or len(f0) == 0:
        return None
    
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) == 0:
        print("No voiced F0 found for key detection.")
        return None

    midi_vals = librosa.hz_to_midi(voiced_f0)

    # Filter out potential NaNs from hz_to_midi if input was weird
    midi_vals_clean = midi_vals[~np.isnan(midi_vals)]

    quantized_midi = np.rint(midi_vals_clean).astype(int)
    pitch_classes = (quantized_midi % 12).tolist()

    # Define all keys and scales ---
    note_names = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E',
                  5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}

    def major_scale(root):
        return {(root + i) % 12 for i in [0, 2, 4, 5, 7, 9, 11]}

    # Note that the minor scale is defined with a natural minor pattern
    # which is different from the harmonic minor or melodic minor.
    def minor_scale(root):
        return {(root + i) % 12 for i in [0, 2, 3, 5, 7, 8, 10, 11]}

    key_scales = {}
    for root in range(12):
        key_scales[f"{note_names[root]} Major"] = major_scale(root)
        key_scales[f"{note_names[root]} Minor"] = minor_scale(root)

    total_notes = len(pitch_classes)
    results = {}

    for key, scale in key_scales.items():
        tonic_name = key.split()[0]

        # Get pitch class number for tonic
        tonic_pc = list(note_names.keys())[list(note_names.values()).index(tonic_name)]

        # Count how many melody notes are in this scale
        count_in_scale = sum(1 for note in pitch_classes if note in scale)

        # Count how many times the tonic appears in the melody
        count_tonic_hits = sum(1 for note in pitch_classes if note == tonic_pc)

        # Compute ratios
        scale_ratio = count_in_scale / total_notes if total_notes > 0 else 0
        tonic_ratio = count_tonic_hits / total_notes if total_notes > 0 else 0

        # Weighted score combining scale match and tonic emphasis
        score = (0.6 * scale_ratio) + (0.4 * tonic_ratio)
        results[key] = score

    # print(results)

    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    best_key, best_score = sorted_results[0]
    # print(f"Detected key: {best_key} (Score: {best_score:.3f})")
    return best_key

def detect_key_from_audio(y, sr):
    """
    Detects the musical key of an audio file using a chromagram and the
    Krumhansl-Schmuckler key-finding algorithm.

    This method analyzes the harmonic content of the entire track, making it
    more robust than analyzing a single melodic line (like F0).

    Args:
        audio_path (str): The file path to the audio file (e.g., .wav, .mp3).

    Returns:
        str: The estimated key (e.g., "C Major", "F# Minor") or None if an
             error occurs.
    """

    # These are empirically derived profiles representing the perceived
    # stability of each pitch class within a major or minor key.
    # The order is C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    
    # Major key profile
    krumhansl_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

    # Minor key profile
    krumhansl_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.78, 3.98, 2.69, 3.34, 3.17])

    # Generate all 24 key profiles by rotating the base profiles
    key_profiles = {}
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for i in range(12):
        key_profiles[f"{note_names[i]} Major"] = np.roll(krumhansl_major, i)
        key_profiles[f"{note_names[i]} Minor"] = np.roll(krumhansl_minor, i)

    # Use a harmonic-percussive source separation to focus on harmonic content
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # Compute the chromagram from the harmonic component
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    # Aggregate chroma features across time to get a single vector
    # This represents the total energy of each pitch class in the song
    chroma_vector = np.sum(chromagram, axis=1)
    
    # Normalize the chroma vector to be comparable with key profiles
    chroma_vector = chroma_vector / np.sum(chroma_vector)

    scores = {}
    for key_name, key_profile in key_profiles.items():
        # Calculate the Pearson correlation coefficient
        # This measures how well the song's pitch content matches the key profile
        correlation = np.corrcoef(chroma_vector, key_profile)[0, 1]
        scores[key_name] = correlation

    # print(scores)

    # Find the key with the highest correlation score
    best_key = max(scores, key=scores.get)
    return best_key

if __name__ == "__main__":
    f0, audio = get_f0_for_analysis("inputs/our_mirage.mp3")
    detected_key = detect_key_from_audio(audio, constants.SAMPLING_RATE)
    print(f"Detected Key: {detected_key}")