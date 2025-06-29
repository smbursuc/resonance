import numpy as np
import librosa
import sys
import os
import math
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pitch_extraction_new import get_f0_for_analysis

import constants

def calculate_sustainability_score(f0, frame_period, min_duration_sec):
    sustained_notes = find_sustained_notes_by_midi(f0, frame_period, min_duration_sec)
    return calculate_sustainability_score_impl(sustained_notes, f0, frame_period)

def find_sustained_notes_by_midi(f0, frame_period, min_duration_sec):
    """
    Identifies sustained notes by converting F0 to MIDI, quantizing, and finding
    runs of the same note. This version corrects the critical bug related to
    handling of unvoiced (f0=0) frames.

    Args:
        f0 (np.ndarray): The F0 contour in Hz.
        frame_period (float): The duration of one frame in seconds.
        min_duration_sec (float): The minimum duration for a note to be considered sustained.

    Returns:
        list: A list of tuples, each being (start_time, end_time, average_f0).
    """
    if f0 is None or len(f0) == 0:
        return []

    # Prevent divide-by-zero warnings by creating a copy and handling zeros manually
    f0_safe = f0.copy()
    f0_safe[f0_safe == 0] = np.nan # Replace 0s with NaN for hz_to_midi
    midi_notes = librosa.hz_to_midi(f0_safe)

    # Now, midi_notes contains NaNs where f0 was 0.
    # Replace these NaNs with a placeholder *before* rounding or casting.
    nan_placeholder = -1
    midi_notes[np.isnan(midi_notes)] = nan_placeholder

    # Now that all NaNs are gone cast to integer.
    quantized_midi_notes = np.round(midi_notes).astype(int)

    # Use a sentinel value to simplify loop logic.
    # This robust pattern ensures the final run of notes is always processed.
    final_sentinel = nan_placeholder - 1
    notes_with_sentinel = np.append(quantized_midi_notes, [final_sentinel])

    # Loop through and find runs of identical notes ---
    sustained_notes = []
    min_frames = int(np.ceil(min_duration_sec / frame_period))
    
    run_start_idx = 0
    for i in range(1, len(notes_with_sentinel)):

        # A run ends when the current note is different from the note that started the run.
        if notes_with_sentinel[i] != notes_with_sentinel[run_start_idx]:

            # The run that just ended was from 'run_start_idx' to 'i-1'.
            end_idx = i - 1
            run_len = (end_idx - run_start_idx) + 1
            
            note_of_run = notes_with_sentinel[run_start_idx]

            # If the run was a voiced note (not our placeholder) and was long enough, record it.
            if note_of_run != nan_placeholder and run_len >= min_frames:
                start_time = run_start_idx * frame_period
                end_time = (end_idx + 1) * frame_period
                
                # Get the true average frequency from the original F0 data for precision.
                avg_f0 = np.mean(f0[run_start_idx : end_idx + 1])
                sustained_notes.append((start_time, end_time, avg_f0))

            # The next run starts at the current index 'i'.
            run_start_idx = i
            
    return sustained_notes

def calculate_sustainability_score_impl(sustained_notes, f0, frame_period):
    """
    Calculates a more meaningful sustainability score based on duration.
    Score = (Total duration of sustained notes / Total duration of all voiced parts) * 100
    """
    total_voiced_duration = np.count_nonzero(f0) * frame_period
    
    if total_voiced_duration == 0:
        return 0.0

    total_sustained_duration = sum([(end - start) for start, end, f0_val in sustained_notes])
    
    score = (total_sustained_duration / total_voiced_duration) * 100
    return score

if __name__ == "__main__":
    f0, audio = get_f0_for_analysis("inputs/antigravity_intro.wav")
    hop_length = constants.HOP_LENGTH
    sr = constants.SAMPLING_RATE
    frame_period = constants.FRAME_PERIOD

    min_duration_sec = 0.2
    
    sustained_notes = find_sustained_notes_by_midi(f0,
                                                   frame_period,
                                                   min_duration_sec)

    if not sustained_notes:
        print("No sustained notes found meeting the final criteria.")
    else:
        print(f"{'Start Time (s)':<15} {'End Time (s)':<15} {'Duration (s)':<15} {'Avg F0 (Hz)':<15}")
        print("-" * 65)
        for start, end, avg_f0 in sustained_notes:
            duration = end - start
            print(f"{start:<15.3f} {end:<15.3f} {duration:<15.3f} {avg_f0:<15.2f}")

    print("\n--- Final Score ---")
    score = calculate_sustainability_score_impl(sustained_notes, f0, frame_period)
    print(f"Sustainability Score: {score:.2f}%")
    print("(Percentage of total singing time spent holding a sustained note)")
