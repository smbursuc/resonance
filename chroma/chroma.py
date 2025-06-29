import numpy as np
import librosa
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pitch_extraction_new import get_f0_for_analysis

from intervals import interval_analysis
import constants

def calculate_chroma_from_f0(f0, sr, hop_length):
    """
    Calculates a chroma vector by using the F0 contour directly.
    """
    if f0 is None or len(f0) == 0: 
        return None
    
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) == 0: 
        return None
    
    midi_notes = librosa.hz_to_midi(voiced_f0)
    valid_midi_mask = ~np.isnan(midi_notes) & ~np.isinf(midi_notes)

    if not np.any(valid_midi_mask): 
        return None
    
    return calculate_chroma(midi_notes[valid_midi_mask])

def calculate_chroma_from_f0_segmentation(f0, 
                                          sr, 
                                          hop_length,
                                          min_note_duration_sec,
                                          pitch_stability_threshold_semitones,
                                          note_stability_reference_lookback_frames):
    """
    Calculates a chroma vector by first segmenting the F0 contour into distinct notes.
    """

    note_pitches_midi, _, _ = interval_analysis.segment_notes(f0,
                                                              sr,
                                                              hop_length,
                                                              min_note_duration_sec,
                                                              pitch_stability_threshold_semitones,
                                                              note_stability_reference_lookback_frames)

    if note_pitches_midi is None or len(note_pitches_midi) == 0:
        return None

    return calculate_chroma(note_pitches_midi)
    
def calculate_chroma(midi_notes):
    """
    Given MIDI notes, calculates the chroma vector (pitch class distribution).
    Returns a 12-element vector where each element corresponds to the count of notes in that pitch class.
    """
    pitch_classes = np.mod(np.round(midi_notes), 12).astype(int)
    chroma_counts = np.zeros(12)
    unique_pcs, counts = np.unique(pitch_classes, return_counts=True)
    chroma_counts[unique_pcs] = counts
    total_counts = np.sum(chroma_counts)
    return chroma_counts / total_counts if total_counts > 0 else None

if __name__ == "__main__":
    f0, audio = get_f0_for_analysis("inputs/btf_exodus_vocals.mp3")

    # Play with those values...
    min_note_duration_sec = 0.06
    pitch_stability_threshold_semitones = 1.0
    note_stability_reference_lookback_frames = 3

    sr = constants.SAMPLING_RATE
    hop_length = constants.HOP_LENGTH

    chroma_vector = calculate_chroma_from_f0_segmentation(f0,
                                                          sr,
                                                          hop_length,
                                                          min_note_duration_sec,
                                                          pitch_stability_threshold_semitones,
                                                          note_stability_reference_lookback_frames)

    # chroma_vector = calculate_chroma_from_f0(f0, sr, hop_length)
    note_names = [
            'C', 'C# / Db', 'D', 'D# / Eb', 'E', 'F',
            'F# / Gb', 'G', 'G# / Ab', 'A', 'A# / Bb', 'B'
        ]

    for i, value in enumerate(chroma_vector):
        note_name = note_names[i]
        print(f"{note_name:<10} | {value:.8f}")