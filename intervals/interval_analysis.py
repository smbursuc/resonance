import numpy as np
import librosa
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pitch_extraction_new import get_f0_for_analysis

import constants

def segment_notes(f0_hz, sr, hop_length,
                  min_note_duration_sec,
                  pitch_stability_threshold_semitones,
                  note_stability_reference_lookback_frames):
    """
    Segments the F0 contour into distinct notes based on stability and duration.
    Args:
        f0_hz (np.ndarray): The F0 contour in Hz.
        sr (int): Sampling rate of the audio.
        hop_length (int): Hop length in samples.
        min_note_duration_sec (float): Minimum duration for a note to be considered valid.
        pitch_stability_threshold_semitones (float): Threshold for pitch stability in semitones.
        note_stability_reference_lookback_frames (int): Number of frames to look back for stability reference.
    
    Returns:
        note_pitches_midi (np.ndarray): Array of MIDI pitches for the segmented notes.
        percentage_voiced_frames (float): Percentage of frames that are voiced.
        num_raw_pitch_segments (int): Number of raw pitch segments detected.
    """ 
    
    if f0_hz is None or len(f0_hz) == 0: 
        return
    frame_period_sec = float(hop_length) / sr
    min_frames_for_note = max(1, int(np.ceil(min_note_duration_sec / frame_period_sec)))
    f0_midi = np.full_like(f0_hz, np.nan, dtype=float)
    valid_f0_mask = (f0_hz > 0) & np.isfinite(f0_hz)
    f0_midi[valid_f0_mask] = 69 + 12 * np.log2(f0_hz[valid_f0_mask] / 440.0) # equivalent to librosa.hz_to_midi(f0_hz[valid_f0_mask])
    num_total_frames = len(f0_midi); 
    num_voiced_frames = np.sum(~np.isnan(f0_midi))
    percentage_voiced_frames = (num_voiced_frames / num_total_frames) * 100 if num_total_frames > 0 else 0.0
    notes_midi_list = []; 
    current_note_candidate_pitches = []
    num_raw_pitch_segments = 0
    in_raw_segment = False

    # The idea is to iterate through the F0 contour and segment it into notes based on stability.
    # We will use a simple heuristic: if the pitch remains within a certain threshold for a minimum duration, we consider it a note.
    # If the pitch deviates beyond the threshold, we consider it a new note.
    # We will use a list to collect candidate pitches for the current note and apply stability checks.
    for i in range(num_total_frames):
        pitch_val_midi = f0_midi[i]
        if not np.isnan(pitch_val_midi):
            if not in_raw_segment: num_raw_pitch_segments += 1; 
            in_raw_segment = True
        else: 
            in_raw_segment = False
        if not np.isnan(pitch_val_midi):
            if not current_note_candidate_pitches: 
                current_note_candidate_pitches.append(pitch_val_midi)
            else:
                ref_len = min(len(current_note_candidate_pitches), note_stability_reference_lookback_frames)
                reference_pitch_midi = np.median(current_note_candidate_pitches[:ref_len]) if ref_len > 0 else current_note_candidate_pitches[0]
                if abs(pitch_val_midi - reference_pitch_midi) <= pitch_stability_threshold_semitones:
                    current_note_candidate_pitches.append(pitch_val_midi)
                else:
                    if len(current_note_candidate_pitches) >= min_frames_for_note:
                        notes_midi_list.append(np.median(current_note_candidate_pitches))
                    current_note_candidate_pitches = [pitch_val_midi]
        else:
            if len(current_note_candidate_pitches) >= min_frames_for_note:
                notes_midi_list.append(np.median(current_note_candidate_pitches))
            current_note_candidate_pitches = []
    if len(current_note_candidate_pitches) >= min_frames_for_note:
        notes_midi_list.append(np.median(current_note_candidate_pitches))
    note_pitches_midi = np.array(notes_midi_list); 

    return note_pitches_midi, percentage_voiced_frames, num_raw_pitch_segments

def analyze_melodic_intervals_v2(f0_hz, sr, hop_length,
                                 min_note_duration_sec,
                                 pitch_stability_threshold_semitones,
                                 note_stability_reference_lookback_frames):
    
    """
    Analyzes melodic intervals in the F0 contour and returns various statistics.
    """
    
    note_pitches_midi, percentage_voiced_frames, num_raw_pitch_segments = segment_notes(
        f0_hz=f0_hz, 
        sr=sr, 
        hop_length=hop_length,
        min_note_duration_sec=min_note_duration_sec,
        pitch_stability_threshold_semitones=pitch_stability_threshold_semitones,
        note_stability_reference_lookback_frames=note_stability_reference_lookback_frames
    )
    
    num_notes = len(note_pitches_midi)
    notes_per_segment_ratio = num_notes / num_raw_pitch_segments if num_raw_pitch_segments > 0 else np.nan
    if num_notes < 2:
        return
    intervals_semitones = np.diff(note_pitches_midi) 
    num_intervals = len(intervals_semitones)
    abs_intervals = np.abs(intervals_semitones)
    return {'num_notes': num_notes, 
            'num_intervals': num_intervals,
            'avg_interval_abs_semitones': np.mean(abs_intervals), 
            'std_interval_abs_semitones': np.std(abs_intervals),
            'median_interval_abs_semitones': np.median(abs_intervals), 
            'min_interval_abs_semitones': np.min(abs_intervals),
            'max_interval_abs_semitones': np.max(abs_intervals),
            'proportion_stepwise': (np.sum(abs_intervals <= 2) / num_intervals),
            'proportion_small_leaps': (np.sum((abs_intervals > 2) & (abs_intervals < 7)) / num_intervals),
            'proportion_large_leaps': (np.sum(abs_intervals >= 7) / num_intervals),
            'percentage_voiced_frames': percentage_voiced_frames, 
            'num_raw_pitch_segments': num_raw_pitch_segments,
            'notes_per_segment_ratio': notes_per_segment_ratio}

if __name__ == '__main__':
    audio_path = "inputs/starting_over_vocals.mp3"
    f0, audio = get_f0_for_analysis(audio_path)

    sr = constants.SAMPLING_RATE
    hop_length = constants.HOP_LENGTH 

    print("\nAnalyzing f0_test_hz (more leaps):")
    interval_features1 = analyze_melodic_intervals_v2(
        f0_hz=f0, 
        sr=sr, 
        hop_length=hop_length,
        min_note_duration_sec=0.05,
        pitch_stability_threshold_semitones=0.8,
        note_stability_reference_lookback_frames=3
    )
    
    for key, value in interval_features1.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

