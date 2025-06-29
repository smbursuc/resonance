import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import librosa
import re
import warnings
from adjustText import adjust_text

# Resonance specific imports
from pitch_extraction_new import get_f0_for_analysis
from sustained_pitches import sustained_pitches
from key_detection import key_detection
from intervals import interval_analysis
from chroma import chroma
from timbre import vocal_timbre
from timbre import mfcc
import constants

# --- Global Parameters ---
AUDIO_FOLDER = "separated_audio_output" # Default folder to scan for audio files.

# Clustering K values (tunable)
K_CHROMA_CLUSTERS = 5
K_SUSTAINABILITY_CLUSTERS = 3
K_RATIONAL_CLUSTERS = 4
K_MFCC_CLUSTERS = 4 # For instrumental MFCCs
K_VOCAL_TIMBRE_CLUSTERS = 5 

# Vocal Analysis Parameters
STABILITY_THRESHOLD_CV = 0.1
MIN_SUSTAINED_DURATION_SEC = 0.2
MIN_NOTE_DURATION_SEC_INTERVALS = 0.06
PITCH_STABILITY_SEMITONES_INTERVALS = 1.0
NOTE_STABILITY_LOOKBACK_INTERVALS = 3

# MFCC Parameters (shared where applicable)
N_MFCC = 13 # Number of MFCC coefficients to extract, standard number
N_FFT_MFCC = 400 # FFT window for MFCC (25ms at 16kHz)

# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

def create_error_entry_vocal(current_result):
    current_result['chroma_vector'] = None
    current_result['sustainability_score'] = np.nan 
    current_result['key'] = "Error"
    current_result['vocal_timbre_features_raw'] = None

def clusterize(file_paths):
    """
    Starts the clustering process for audio files, analyzing vocal and instrumental tracks.
    It will search for audio files in the specified AUDIO_FOLDER if no file paths are provided.
    Will use PCA for dimensionality reduction before clustering.
    Uses KMeans for clustering the chroma, sustainability, rationality, MFCC, and vocal timbre features.
    Saves the clustering results to a CSV file  in the current directory.
    Generates a plot of the clustering results with PCA-reduced features.
    """
    audio_extensions = ["wav", "mp3", "flac"]
    if file_paths is None:
        # If no file paths are provided, search the default audio folder which includes all saved audio files.
        audio_folder_main = AUDIO_FOLDER
        supported_glob_formats = [f"*.{ext}" for ext in audio_extensions]

        print(f"Searching for audio files in: {audio_folder_main}")
        all_audio_files_paths = []
        for fmt in supported_glob_formats:
            all_audio_files_paths.extend(glob.glob(os.path.join(audio_folder_main, "**", fmt), recursive=True))
        all_audio_files_paths = sorted(list(set(all_audio_files_paths)))

        if not all_audio_files_paths: print(f"No audio files found in '{audio_folder_main}'."); exit()
        print(f"Found {len(all_audio_files_paths)} audio files initially.")
    else:
        all_audio_files_paths = file_paths

    # Processing is different for vocals and instrumentals, so create regex patterns to match filenames.
    vocal_pattern_str = r"^(?P<basename>.+?)_vocals\.(" + "|".join(audio_extensions) + r")$"
    instrumental_pattern_str = r"^(?P<basename>.+?)_(instrumentals|instrumental|accompaniment|instrum)\.(" + "|".join(audio_extensions) + r")$"
    vocal_regex = re.compile(vocal_pattern_str, re.IGNORECASE)
    instrumental_regex = re.compile(instrumental_pattern_str, re.IGNORECASE)

    song_files_map = {}
    for path in all_audio_files_paths:
        filename = os.path.basename(path)
        v_match = vocal_regex.match(filename)
        i_match = instrumental_regex.match(filename)
        base_name = None
        file_type = None
        if v_match: 
            base_name = v_match.group('basename') 
            file_type = 'vocals'
        elif i_match: 
            base_name = i_match.group('basename') 
            file_type = 'instrumental'
        if base_name:
            if base_name not in song_files_map: song_files_map[base_name] = {}
            song_files_map[base_name][file_type] = path
            song_files_map[base_name][f'filename_{file_type}'] = filename

    results_list = []
    for base_name, files_dict in song_files_map.items():
        print("-" * 30 + f"\nProcessing song: {base_name}")
        current_result = {'base_songname': base_name}

        sr = constants.SAMPLING_RATE
        
        if 'vocals' in files_dict:
            vocal_path = files_dict['vocals']
            current_result['filename'] = files_dict.get('filename_vocals', os.path.basename(vocal_path))
            print(f"Processing vocal track: {current_result['filename']}")
            try:
                f0, audio_vocal = get_f0_for_analysis(vocal_path)
                hop_length = constants.HOP_LENGTH
                frame_period = constants.FRAME_PERIOD

                current_result['chroma_vector'] = chroma.calculate_chroma_from_f0(f0, sr, hop_length)

                # Uncomment the following if you want to use the segmentation-based chroma calculation.
                # For some reason, despite this option being more accurate theoretically, yields similar if not worse results than just
                # analyzing the F0 directly. This should be investigated further.

                # current_result['chroma_vector'] = chroma.calculate_note_chroma_from_f0_segmentation(
                #     f0=f0,
                #     sr=sr,
                #     hop_length=hop_length,
                #     min_note_duration_sec=MIN_NOTE_DURATION_SEC_INTERVALS,
                #     pitch_stability_threshold_semitones=PITCH_STABILITY_SEMITONES_INTERVALS,
                #     note_stability_reference_lookback_frames=NOTE_STABILITY_LOOKBACK_INTERVALS
                # )

                current_result['sustainability_score'] = sustained_pitches.calculate_sustainability_score(f0, frame_period, MIN_SUSTAINED_DURATION_SEC)

                melodic_features = interval_analysis.analyze_melodic_intervals_v2(f0, 
                                                                                  sr, 
                                                                                  hop_length, 
                                                                                  MIN_NOTE_DURATION_SEC_INTERVALS, 
                                                                                  PITCH_STABILITY_SEMITONES_INTERVALS, 
                                                                                  NOTE_STABILITY_LOOKBACK_INTERVALS)
                current_result.update(melodic_features)

                current_result['key'] = key_detection.detect_key_from_audio(audio_vocal, sr)

                # Deprecated: If you want to use the F0-based key detection, uncomment the following line.
                # This is not recommended as it may yield less accurate results than using the audio directly.
                # current_result['key'] = key_detection.detect_key_from_f0(f0)
                
                current_result['vocal_timbre_features_raw'] = vocal_timbre.extract_vocal_timbre_features(audio_vocal, sr, N_FFT_MFCC, N_MFCC)
            except Exception as e:
                # If an error only occurs for a single track then it is better to solve it gracefully rather than crashing the entire process.
                print(f"Error processing vocal track {current_result['filename']}: {e}")
                create_error_entry_vocal(current_result)
        else:
            # In case the file somehow gets deleted.
            print(f"No vocal track found for {base_name}, skipping vocal analysis.")
            create_error_entry_vocal(current_result)

        current_result['mean_mfccs_coeffs'] = None
        current_result['filename_instrumental'] = files_dict.get('filename_instrumental', None)
        if 'instrumental' in files_dict:
            instrumental_path = files_dict['instrumental']
            print(f"Processing instrumental track: {current_result['filename_instrumental']}")
            try:
                audio_instr, sr_instr = librosa.load(instrumental_path, sr, mono=True)
                mfccs = mfcc.extract_mfcc(audio_instr, sr_instr, N_MFCC, N_FFT_MFCC, constants.HOP_LENGTH)
                current_result['mean_mfccs_coeffs'] = np.mean(mfccs, axis=1)
            except Exception as e: 
                print(f"Error processing instrumental track {current_result['filename_instrumental']}: {e}")
        else: 
            print(f"No instrumental track found for {base_name}.")

        results_list.append(current_result)

    df = pd.DataFrame(results_list)

    # Feature Preparation for clustering
    all_scalar_features_list = ['sustainability_score', 
                                'avg_interval_abs_semitones', 
                                'std_interval_abs_semitones', 
                                'median_interval_abs_semitones', 
                                'min_interval_abs_semitones', 
                                'max_interval_abs_semitones', 
                                'proportion_stepwise', 
                                'proportion_small_leaps', 
                                'proportion_large_leaps', 
                                'percentage_voiced_frames', 
                                'num_raw_pitch_segments', 
                                'notes_per_segment_ratio']
    
    # Because we gracefully handle missing features, we need to ensure all scalar features are present.
    for col in all_scalar_features_list:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
        else: df[col] = 0

    # Unpack MFCC coefficients into separate columns
    mfcc_feature_names = []
    if 'mean_mfccs_coeffs' in df.columns and not df['mean_mfccs_coeffs'].dropna().empty:
        mfcc_data_list = df['mean_mfccs_coeffs'].dropna().tolist()
        num_mfcc_coeffs = len(mfcc_data_list[0])
        mfcc_feature_names = [f'mean_mfcc_{i}' for i in range(num_mfcc_coeffs)]
        mfcc_df = pd.DataFrame(mfcc_data_list, columns=mfcc_feature_names, index=df['mean_mfccs_coeffs'].dropna().index)
        df = pd.concat([df, mfcc_df], axis=1)
        
    # Drop the original mean_mfccs_coeffs column as it is no longer needed
    df.drop(columns=['mean_mfccs_coeffs'], inplace=True, errors='ignore')
    for col in mfcc_feature_names:
        if col not in df.columns: df[col] = 0.0
        else: df[col].fillna(df[col].mean(), inplace=True)


    # == Clustering 1: Chroma (Vocals) ==
    print("\n--- Clustering 1: Tonal Distance (Chroma) ---")
    df_chroma_valid = df.dropna(subset=['chroma_vector']).copy()
    df['chroma_cluster'] = np.nan; 
    df['chroma_pca1_plot'] = np.nan 
    df['chroma_pca2_plot'] = np.nan
    if not df_chroma_valid.empty:
        X_chroma = np.vstack(df_chroma_valid['chroma_vector'].values)
        X_chroma_scaled = StandardScaler().fit_transform(X_chroma)
        k_c = min(K_CHROMA_CLUSTERS, len(df_chroma_valid))

        # If there are not enough valid chroma vectors, we cannot cluster.
        if k_c >= 2: 
            df_chroma_valid['chroma_cluster'] = KMeans(n_clusters=k_c, random_state=42).fit_predict(X_chroma_scaled)
        else: 
            df_chroma_valid['chroma_cluster'] = 0
        nc_pca_c = min(2, X_chroma_scaled.shape[1])
        pca_c_res = PCA(n_components=nc_pca_c).fit_transform(X_chroma_scaled)

        # If there are not enough valid chroma vectors, we cannot visualize PCA.
        # Therefore we only add PCA columns if there are enough components.
        if nc_pca_c > 0: 
            df_chroma_valid['chroma_pca1_plot'] = pca_c_res[:,0]
        if nc_pca_c > 1: 
            df_chroma_valid['chroma_pca2_plot'] = pca_c_res[:,1]

        df.update(df_chroma_valid[['chroma_cluster', 'chroma_pca1_plot', 'chroma_pca2_plot']])

    # == Clustering 2: Sustainability (Vocals) ==
    print("\n--- Clustering 2: Pitch Sustainability ---")
    df['sustainability_cluster'] = np.nan
    df_sustain_valid = df[['sustainability_score']].dropna()
    if not df_sustain_valid.empty:
        X_sustain_scaled = StandardScaler().fit_transform(df_sustain_valid)
        k_s = min(K_SUSTAINABILITY_CLUSTERS, len(df_sustain_valid))
        if k_s >= 2: 
            df.loc[df_sustain_valid.index, 'sustainability_cluster'] = KMeans(n_clusters=k_s, random_state=42).fit_predict(X_sustain_scaled)
        else: 
            df.loc[df_sustain_valid.index, 'sustainability_cluster'] = 0

    # == Clustering 3: "Rational" Combined (Vocals) ==
    print("\n--- Clustering 3: 'Rational' Combined Features (Vocals) ---")

    # These features are considered "rational" for clustering purposes.
    rational_scalar_features = ['median_interval_abs_semitones', 
                                'proportion_stepwise', 
                                'percentage_voiced_frames', 
                                'num_raw_pitch_segments', 
                                'notes_per_segment_ratio']
    features_for_rational_clustering = [f for f in rational_scalar_features if f in df.columns]
    df['combined_cluster_rational'] = np.nan 
    df['combined_pca1_rational_viz'] = np.nan 
    df['combined_pca2_rational_viz'] = np.nan

    if 'chroma_pca1_plot' in df.columns and 'chroma_pca2_plot' in df.columns and not df[['chroma_pca1_plot', 'chroma_pca2_plot']].isnull().all().all():
        features_for_rational_clustering.extend(['chroma_pca1_plot', 'chroma_pca2_plot'])

    df_rational_input = df[features_for_rational_clustering].copy().dropna(how='any')
    if not df_rational_input.empty:
        X_rational_scaled = StandardScaler().fit_transform(df_rational_input)
        k_r = min(K_RATIONAL_CLUSTERS, len(df_rational_input))
        if k_r >= 2: 
            df.loc[df_rational_input.index, 'combined_cluster_rational'] = KMeans(n_clusters=k_r, random_state=42).fit_predict(X_rational_scaled)
        else: 
            df.loc[df_rational_input.index, 'combined_cluster_rational'] = 0
        nc_pca_r_viz = min(2, X_rational_scaled.shape[1]); pca_r_res_viz = PCA(n_components=nc_pca_r_viz).fit_transform(X_rational_scaled)
        if nc_pca_r_viz > 0: 
            df.loc[df_rational_input.index, 'combined_pca1_rational_viz'] = pca_r_res_viz[:,0]
        if nc_pca_r_viz > 1: 
            df.loc[df_rational_input.index, 'combined_pca2_rational_viz'] = pca_r_res_viz[:,1]

    # == Clustering 4: Instrumental MFCCs ==
    print("\n--- Clustering 4: Instrumental Timbre (MFCCs) ---")
    df['mfcc_cluster'] = np.nan 
    df['mfcc_pca1_plot'] = np.nan
    df['mfcc_pca2_plot'] = np.nan
    if mfcc_feature_names:
        df_mfcc_valid = df.dropna(subset=mfcc_feature_names, how='any').copy()
        if not df_mfcc_valid.empty:
            X_mfcc = df_mfcc_valid[mfcc_feature_names].values; X_mfcc_scaled = StandardScaler().fit_transform(X_mfcc)
            k_m = min(K_MFCC_CLUSTERS, len(df_mfcc_valid))
            if k_m >= 2: 
                df.loc[df_mfcc_valid.index, 'mfcc_cluster'] = KMeans(n_clusters=k_m, random_state=42).fit_predict(X_mfcc_scaled)
            else: 
                df.loc[df_mfcc_valid.index, 'mfcc_cluster'] = 0
            nc_pca_m = min(2, X_mfcc_scaled.shape[1]); pca_m_res = PCA(n_components=nc_pca_m).fit_transform(X_mfcc_scaled)
            if nc_pca_m > 0: 
                df.loc[df_mfcc_valid.index, 'mfcc_pca1_plot'] = pca_m_res[:,0]
            if nc_pca_m > 1: 
                df.loc[df_mfcc_valid.index, 'mfcc_pca2_plot'] = pca_m_res[:,1]

    # == Clustering 5: Vocal Timbre Analysis ==
    print("\n--- Clustering 5: Vocal Timbre ---")
    df['vocal_timbre_cluster'] = np.nan
    df['vocal_timbre_pca1_centroid'] = np.nan
    df['vocal_timbre_pca2_centroid'] = np.nan
    
    df_vocal_timbre_valid = df.dropna(subset=['vocal_timbre_features_raw']).copy()

    if not df_vocal_timbre_valid.empty:
        all_vocal_features = []
        song_indices = []
        for index, row in df_vocal_timbre_valid.iterrows():
            features = row['vocal_timbre_features_raw']
            all_vocal_features.append(features)
            song_indices.extend([index] * len(features))
        
        master_feature_array = np.vstack(all_vocal_features)
        master_feature_array_scaled = StandardScaler().fit_transform(master_feature_array)
        
        pca_vocal_timbre = PCA(n_components=2)
        pca_results_all_frames = pca_vocal_timbre.fit_transform(master_feature_array_scaled)
        
        df_all_frames = pd.DataFrame({
            'song_index': song_indices,
            'pca1': pca_results_all_frames[:, 0],
            'pca2': pca_results_all_frames[:, 1]
        })
        
        song_centroids = df_all_frames.groupby('song_index')[['pca1', 'pca2']].mean()
        
        k_vt = min(K_VOCAL_TIMBRE_CLUSTERS, len(song_centroids))
        if k_vt >= 2:
            cluster_labels = KMeans(n_clusters=k_vt, random_state=42, n_init=10).fit_predict(song_centroids)
            song_centroids['cluster'] = cluster_labels
        else:
            song_centroids['cluster'] = 0

        df['vocal_timbre_cluster'] = song_centroids['cluster']
        df['vocal_timbre_pca1_centroid'] = song_centroids['pca1']
        df['vocal_timbre_pca2_centroid'] = song_centroids['pca2']

    # --- Plotting (excludes the vocal timbre section) --- #
    plot_configs = [
        {'name': 'Tonal Distance', 'cluster_col': 'chroma_cluster', 'pca1_col': 'chroma_pca1_plot', 'pca2_col': 'chroma_pca2_plot', 'palette': 'viridis'},
        {'name': 'Interval Analysis Combined', 'cluster_col': 'combined_cluster_rational', 'pca1_col': 'combined_pca1_rational_viz', 'pca2_col': 'combined_pca2_rational_viz', 'palette': 'coolwarm'},
        {'name': 'Instrumental MFCC', 'cluster_col': 'mfcc_cluster', 'pca1_col': 'mfcc_pca1_plot', 'pca2_col': 'mfcc_pca2_plot', 'palette': 'cubehelix'}
    ]

    for config in plot_configs:
        plt.figure(figsize=(14, 10))
        plot_df = df.dropna(subset=[config['cluster_col'], config['pca1_col'], config['pca2_col']]).copy()
        if not plot_df.empty:
            plot_df[config['cluster_col']] = plot_df[config['cluster_col']].astype(int)
            k_actual = plot_df[config['cluster_col']].nunique()
            sns.scatterplot(data=plot_df, x=config['pca1_col'], y=config['pca2_col'], hue=config['cluster_col'],
                            palette=sns.color_palette(config['palette'], n_colors=max(1, k_actual)),
                            s=200, alpha=0.9, edgecolor='k')
            texts = []
            for _, point_row in plot_df.iterrows():
                label = point_row.get('base_songname', 'Unknown')
                texts.append(
                            plt.text(point_row[config['pca1_col']], point_row[config['pca2_col']] + 0.02,
                            os.path.splitext(label)[0][:20], fontsize=9)
                            )
            
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            plt.title(f"{config['name']} Clusters (k={k_actual})", fontsize=16)
            plt.xlabel(f"{config['name']} PCA 1"); plt.ylabel(f"{config['name']} PCA 2")
            plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout(rect=[0,0,0.85,1]); plt.savefig(f"plot_{config['name'].lower().replace(' ', '_')}_clusters.png")
            print(f"Saved plot_{config['name'].lower().replace(' ', '_')}_clusters.png")
        else: print(f"No data for {config['name']} plot.")
        plt.close()

    # --- Plotting including vocal timbre --- #
    # This plot now shows one dot per song, representing its average timbre.
    df_centroids = df.dropna(subset=['vocal_timbre_pca1_centroid', 'vocal_timbre_pca2_centroid']).copy()
    if not df_centroids.empty:
        plt.figure(figsize=(14, 10))
        
        sns.scatterplot(data=df_centroids, x='vocal_timbre_pca1_centroid', y='vocal_timbre_pca2_centroid',
                        hue='vocal_timbre_cluster', palette='viridis', s=200, alpha=0.9, edgecolor='k')

        texts = []
        for index, row in df_centroids.iterrows():
            label = row.get('base_songname', 'Unknown')
            texts.append(
                plt.text(row['vocal_timbre_pca1_centroid'], row['vocal_timbre_pca2_centroid'],
                         os.path.splitext(label)[0][:30], fontsize=9)
            )
        
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
                     
        plt.title('Vocal Timbre Comparison', fontsize=16)
        plt.xlabel('Vocal Timbre PCA 1 (Timbre Characteristic)', fontsize=12)
        plt.ylabel('Vocal Timbre PCA 2 (Timbre Characteristic)', fontsize=12)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout(rect=[0,0,0.85,1])
        plt.savefig("plot_vocal_timbre_comparison.png")
        print("Saved plot_vocal_timbre_comparison.png")
        plt.close()
    else:
        print("No data for Vocal Timbre Comparison plot.")

    # --- Plotting the pitch sustainability --- #
    # The only plot that uses a bar chart to visualize the sustainability scores.
    plt.figure(figsize=(max(12, len(df) * 0.6), 8))
    plot_df_sustain = df.dropna(subset=['sustainability_cluster', 'sustainability_score']).copy()
    if not plot_df_sustain.empty:
        plot_df_sustain['sustainability_cluster'] = plot_df_sustain['sustainability_cluster'].astype(int)
        k_actual = plot_df_sustain['sustainability_cluster'].nunique()
        sns.barplot(data=plot_df_sustain.sort_values('sustainability_score'), x='filename', y='sustainability_score',
                    hue='sustainability_cluster', palette=sns.color_palette("magma", n_colors=max(1, k_actual)), dodge=False)
        plt.title(f'Sustainability Score by File (Clustered, k={k_actual})', fontsize=16)
        plt.xlabel('Audio File (Vocals)'); plt.ylabel('Sustainability Score (%)')
        plt.xticks(rotation=70, ha='right', fontsize=8)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(); plt.savefig("plot_sustainability_clusters.png"); print("Saved plot_sustainability_clusters.png")
    else: print("No data for Sustainability plot.")
    plt.close()

    # --- Final Output ---
    print("\n--- Final DataFrame ---")

    # Choose what is relevant to display/save.
    console_display_cols = ['base_songname', 'sustainability_score', 'median_interval_abs_semitones',
                            'notes_per_segment_ratio', 'chroma_cluster', 'sustainability_cluster',
                            'combined_cluster_rational', 'mfcc_cluster', 'vocal_timbre_cluster', 'key']
    existing_console_display_cols = [col for col in console_display_cols if col in df.columns]
    print(df[existing_console_display_cols].round(3).to_string())

    csv_columns_to_save = ['base_songname', 'filename', 'filename_instrumental', 'key',
                           'sustainability_score', 'avg_interval_abs_semitones', 'std_interval_abs_semitones',
                           'median_interval_abs_semitones', 'min_interval_abs_semitones', 'max_interval_abs_semitones',
                           'proportion_stepwise', 'proportion_small_leaps', 'proportion_large_leaps',
                           'percentage_voiced_frames', 'num_raw_pitch_segments', 'notes_per_segment_ratio',
                           'chroma_cluster', 'sustainability_cluster', 'combined_cluster_rational',
                           'mfcc_cluster', 'vocal_timbre_cluster', 
                           'chroma_pca1_plot',  'chroma_pca2_plot',
                           'combined_pca1_rational_viz', 'combined_pca2_rational_viz',
                           'mfcc_pca1_plot', 'mfcc_pca2_plot',
                           'vocal_timbre_pca1_centroid', 'vocal_timbre_pca2_centroid'] # Added centroid coordinates
    existing_csv_columns_to_save = [col for col in csv_columns_to_save if col in df.columns]

    output_csv_filename = "audio_analysis_results_summary.csv"
    try:
        df_to_save = df[existing_csv_columns_to_save].copy()
        for col in ['chroma_cluster', 'sustainability_cluster', 'combined_cluster_rational', 'mfcc_cluster', 'vocal_timbre_cluster']:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].astype('Int64')
        df_to_save.round(3).to_csv(output_csv_filename, index=False)
        print(f"\nSuccessfully saved a summary DataFrame to: {output_csv_filename} 📄")
    except Exception as e: print(f"\nError saving summary DataFrame to CSV: {e} ❌")
    
    print("\n--- Script Finished ---")


if __name__ == '__main__':
    clusterize(AUDIO_FOLDER)
