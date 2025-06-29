import argparse
import os
import shutil
import sys
import yt_dlp

# Import the inference module from the Mel-Band-Roformer-Vocal-Model directory.
# In this case it is located one level up from the script's directory.

# Directory where this script is located (/path/to/resonance.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parent directory of the script's directory (/path/to/)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# Name of the directory containing inference.py
MEL_BAND_ROFORMER_FOLDER_NAME = "Mel-Band-Roformer-Vocal-Model"

# Full path to the Mel-Band-Roformer-Vocal-Model directory
MEL_BAND_ROFORMER_DIR = os.path.join(PARENT_DIR, MEL_BAND_ROFORMER_FOLDER_NAME)

# Import proc_folder by adding its directory to sys.path
try:
    # Add the directory containing inference.py to Python's search path
    # This allows 'inference.py' to import its own sibling modules (e.g., 'utils')
    if MEL_BAND_ROFORMER_DIR not in sys.path:
        sys.path.insert(0, MEL_BAND_ROFORMER_DIR) # Insert at the beginning for priority

    # Now, Python should be able to find 'inference' and any modules it imports
    # from its own directory (like 'utils')
    import inference # Attempt to import the inference module directly
    proc_folder = inference.proc_folder # Access the function
    
    print(f"Successfully imported 'proc_folder' from 'inference.py' located in: {MEL_BAND_ROFORMER_DIR}")

except ModuleNotFoundError as e:
    # This will catch if 'inference' itself is not found, or if 'inference'
    # fails to import one of its dependencies (like 'utils')
    print(f"Error importing 'proc_folder' or its dependencies (e.g., '{e.name}') from '{MEL_BAND_ROFORMER_DIR}': {e}")
    print("This might be due to 'inference.py' not found, a missing dependency like 'utils.py' within that directory,")
    print(f"or an issue within 'inference.py' or '{e.name}.py' itself.")
    print(f"Ensure '{MEL_BAND_ROFORMER_DIR}' is correctly structured and contains all necessary files.")
    print(f"Current sys.path includes: {MEL_BAND_ROFORMER_DIR} (should be listed if added)")
    sys.exit(1)
except ImportError as e:
    print(f"Error importing 'proc_folder' from '{MEL_BAND_ROFORMER_DIR}': {e}")
    print("This could be an issue within 'inference.py' or its dependencies during their import phase.")
    sys.exit(1)
except AttributeError:
    # This means inference.py was loaded, but proc_folder wasn't found in it.
    print(f"Error: The function 'proc_folder' was not found in 'inference.py' (from {MEL_BAND_ROFORMER_DIR}).")
    print("Please ensure 'inference.py' defines a function named 'proc_folder'.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while trying to import 'proc_folder' from '{MEL_BAND_ROFORMER_DIR}': {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

import clusterize

# --- Configuration ---
# These output folders will be created inside the SCRIPT_DIR (e.g., inside 'resonance/')
DOWNLOADS_BASE_FOLDER = os.path.join(SCRIPT_DIR, "downloaded_playlists")
TEMP_PROCESSING_INPUT_DIR = os.path.join(SCRIPT_DIR, "temp_processing_input") # Temporary input for proc_folder
SEPARATED_AUDIO_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "separated_audio_output") # Final output of separation

# Paths for the audio separation model and configuration
# IMPORTANT: These are now assumed to be inside the MEL_BAND_ROFORMER_DIR.
# If your .ckpt and config files are elsewhere (e.g. inside 'resonance/'), adjust these paths.
# The .ckpt and config files can be found in their Github repository.
MODEL_CKPT_PATH = os.path.join(MEL_BAND_ROFORMER_DIR, "MelBandRoformer.ckpt")
CONFIG_PATH = os.path.join(MEL_BAND_ROFORMER_DIR, "configs", "config_vocals_mel_band_roformer.yaml")

# Device for the separation model. "0" usually means the first GPU.
# If no compatible GPU then it will use CPU.
DEVICE_ID_FOR_SEPARATOR = "0"

def sanitize_filename(filename):
    """Removes or replaces characters that are problematic in filenames."""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def setup_directories():
    """Creates necessary directories if they don't exist."""
    os.makedirs(DOWNLOADS_BASE_FOLDER, exist_ok=True)

    # TEMP_PROCESSING_INPUT_DIR is created/cleaned per track in separate_audio_track
    os.makedirs(SEPARATED_AUDIO_OUTPUT_DIR, exist_ok=True)
    
    model_config_dir = os.path.dirname(CONFIG_PATH)
    if not os.path.isdir(model_config_dir):
        print(f"Warning: Expected model configuration directory not found: {model_config_dir}")
        print(f"Please ensure the path to the config file '{CONFIG_PATH}' is correct and the directory exists.")

    if not os.path.exists(MODEL_CKPT_PATH):
        print(f"Error: Model checkpoint file not found: {MODEL_CKPT_PATH}")
        sys.exit(1)
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Model configuration file not found: {CONFIG_PATH}")
        sys.exit(1)


def get_playlist_info(playlist_url):
    """Fetches playlist title and video information using yt-dlp."""

    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist', 
        'dump_single_json': True, 
        'rejecttitle': 'Private video|Deleted video' 
    }
    videos = []
    playlist_title_sanitized = "Unknown_Playlist"

    print(f"Fetching playlist information for: {playlist_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_dict = ydl.extract_info(playlist_url, download=False)
            
            if playlist_dict and 'title' in playlist_dict:
                playlist_title_sanitized = sanitize_filename(playlist_dict['title'])
            elif playlist_dict and 'id' in playlist_dict:
                 playlist_title_sanitized = f"playlist_{sanitize_filename(playlist_dict['id'])}"

            if playlist_dict and 'entries' in playlist_dict:
                for entry in playlist_dict['entries']:
                    if entry: 
                        video_id = entry.get('id')
                        video_title = entry.get('title', f"Untitled_Video_{video_id}")
                        if video_id and video_title and 'Private video' not in video_title and 'Deleted video' not in video_title:
                            videos.append({
                                'id': video_id,
                                'title': video_title,
                                'url': entry.get('url') 
                            })
                        elif video_title and ('Private video' in video_title or 'Deleted video' in video_title):
                            print(f"Skipping '{video_title}' as it is private or deleted.")
                        else:
                            print(f"Skipping entry due to missing id or title: {entry.get('id', 'N/A')}")
            else:
                print("Could not retrieve entries from the playlist.")
                return None, []
    except yt_dlp.utils.DownloadError as e:
        print(f"Error fetching playlist info: {e}")
        return None, []
    except Exception as e:
        print(f"An unexpected error occurred while fetching playlist info: {e}")
        return None, []
    
    if not videos:
        print("No downloadable videos found in the playlist.")
    
    return playlist_title_sanitized, videos

def download_track(video_info, playlist_download_folder):
    """
    Downloads a single track as MP3 if it doesn't already exist.
    Returns the path to the MP3 file, the base track name (without extension),
    and a boolean indicating if the track was newly downloaded.
    """

    video_title_sanitized = sanitize_filename(video_info['title'])
    base_filename_template = video_title_sanitized 
    expected_mp3_filename = f"{base_filename_template}.mp3"
    potential_mp3_path = os.path.join(playlist_download_folder, expected_mp3_filename)
    
    existing_mp3_path = None
    if os.path.exists(potential_mp3_path): 
        existing_mp3_path = potential_mp3_path
    else: 
        for f_name in os.listdir(playlist_download_folder):
            if f_name.startswith(video_title_sanitized) and f_name.lower().endswith(".mp3"):
                existing_mp3_path = os.path.join(playlist_download_folder, f_name)
                print(f"Found existing track as: {f_name}")
                break

    if existing_mp3_path:
        print(f"Track '{video_title_sanitized}' already downloaded: {existing_mp3_path}")
        return existing_mp3_path, os.path.splitext(os.path.basename(existing_mp3_path))[0], False

    print(f"Downloading '{video_info['title']}'...")
    
    ydl_opts_download = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': {'default': potential_mp3_path}, 
        'quiet': False,
        'noplaylist': True,
        'ffmpeg_location': shutil.which('ffmpeg')
    }
    
    if not ydl_opts_download['ffmpeg_location']:
        print("Error: ffmpeg not found in PATH. Please install ffmpeg and ensure it's accessible.")
        return None, None, False

    try:
        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
            video_url = video_info.get('url') or f"https://www.youtube.com/watch?v={video_info['id']}"
            ydl.download([video_url])
        
        if os.path.exists(potential_mp3_path):
            print(f"Successfully downloaded and converted: {potential_mp3_path}")
            return potential_mp3_path, base_filename_template, True
        else:
            print(f"Error: MP3 file not found at '{potential_mp3_path}' after download attempt for '{video_info['title']}'.")
            for f_name in os.listdir(playlist_download_folder):
                 if f_name.startswith(base_filename_template) and f_name.lower().endswith(".mp3"):
                    found_path = os.path.join(playlist_download_folder, f_name)
                    print(f"Downloaded file found as: {found_path}")
                    return found_path, os.path.splitext(f_name)[0], True
            return None, None, False

    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading track '{video_info['title']}': {e}")
        return None, None, False
    except Exception as e:
        print(f"An unexpected error occurred during download of '{video_info['title']}': {e}")
        return None, None, False

def separate_audio_track(mp3_file_path, base_track_name):
    """
    Processes a single MP3 file using the proc_folder function.
    Returns True on success, False on failure.
    """
    if not os.path.exists(mp3_file_path):
        print(f"Error: MP3 file for separation does not exist: {mp3_file_path}")
        return False

    print(f"Preparing to separate audio for: {base_track_name}")

    if os.path.exists(TEMP_PROCESSING_INPUT_DIR):
        try:
            shutil.rmtree(TEMP_PROCESSING_INPUT_DIR) 
        except Exception as e:
            print(f"Warning: Could not remove old temp directory {TEMP_PROCESSING_INPUT_DIR}: {e}")
    try:
        os.makedirs(TEMP_PROCESSING_INPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create temp directory {TEMP_PROCESSING_INPUT_DIR}: {e}")
        return False

    temp_input_file_name = os.path.basename(mp3_file_path) 
    temp_input_file_path = os.path.join(TEMP_PROCESSING_INPUT_DIR, temp_input_file_name)
    
    try:
        shutil.copy(mp3_file_path, temp_input_file_path)
        print(f"Copied '{temp_input_file_name}' to temporary processing folder: {TEMP_PROCESSING_INPUT_DIR}")
    except Exception as e:
        print(f"Error copying file '{mp3_file_path}' to temporary folder '{TEMP_PROCESSING_INPUT_DIR}': {e}")
        return False

    # For more details check the Mel-Band-Roformer-Vocal-Model repository, there are many configurations available.
    proc_args = [
        "--model_type", "mel_band_roformer",
        "--config_path", CONFIG_PATH,
        "--model_path", MODEL_CKPT_PATH,
        "--input_folder", TEMP_PROCESSING_INPUT_DIR, 
        "--store_dir", SEPARATED_AUDIO_OUTPUT_DIR,   
        "--device_ids", DEVICE_ID_FOR_SEPARATOR
    ]

    print(f"Starting audio separation for '{base_track_name}'...")
    print(f"  Model: {MODEL_CKPT_PATH}")
    print(f"  Config: {CONFIG_PATH}")
    print(f"  Input Folder (for proc_folder): {TEMP_PROCESSING_INPUT_DIR}")
    print(f"  Output Folder (for proc_folder): {SEPARATED_AUDIO_OUTPUT_DIR}")
    print(f"  Device: {DEVICE_ID_FOR_SEPARATOR}")
    
    success = False
    try:        
        proc_folder(proc_args)
        
        # os.chdir(original_cwd) # Restore CWD if changed
        print(f"Audio separation process completed for '{base_track_name}'. Outputs should be in '{SEPARATED_AUDIO_OUTPUT_DIR}'.")
        success = True
    except Exception as e:
        print(f"Error during audio separation for '{base_track_name}': {e}")
        print("Please check if Mel-Band-Roformer-Vocal-Model works on a smaller testcase first.")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        if os.path.exists(TEMP_PROCESSING_INPUT_DIR):
            try:
                shutil.rmtree(TEMP_PROCESSING_INPUT_DIR)
            except Exception as e:
                print(f"Warning: Could not clean up temporary folder {TEMP_PROCESSING_INPUT_DIR}: {e}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Download YouTube playlist tracks and process them for audio separation.")
    parser.add_argument("playlist_url", help="The URL of the YouTube playlist.")
    args = parser.parse_args()

    print("--- YouTube Playlist Audio Processor ---")
    try:
        setup_directories()
    except Exception as e:
        print(f"Failed to set up directories: {e}")
        sys.exit(1)

    playlist_title, videos_info = get_playlist_info(args.playlist_url)

    if playlist_title is None or not videos_info:
        print("Could not retrieve playlist information or no videos found. Exiting.")
        print("Please check the playlist URL and ensure it is accessible AKA not private or deleted.")
        sys.exit(1)

    playlist_download_folder = os.path.join(DOWNLOADS_BASE_FOLDER, playlist_title)
    os.makedirs(playlist_download_folder, exist_ok=True)
    print(f"MP3 downloads will be saved in: {playlist_download_folder}")
    print(f"Separated audio will be saved in: {SEPARATED_AUDIO_OUTPUT_DIR}")

    newly_downloaded_count = 0
    processed_successfully_count = 0
    skipped_existing_download_count = 0
    skipped_existing_separated_count = 0
    failed_download_count = 0
    failed_processing_count = 0
    file_paths = []

    for i, video_info in enumerate(videos_info):
        print(f"\n--- Processing track {i+1}/{len(videos_info)}: {video_info['title']} ---")
        
        mp3_file_path, base_track_name, newly_downloaded = download_track(video_info, playlist_download_folder)

        if not mp3_file_path or not base_track_name:
            print(f"Skipping processing for '{video_info['title']}' due to download failure.")
            failed_download_count +=1
            continue 

        if newly_downloaded:
            newly_downloaded_count +=1
        else:
            skipped_existing_download_count +=1
            
        expected_vocals_path = os.path.join(SEPARATED_AUDIO_OUTPUT_DIR, f"{base_track_name}_vocals.wav")
        expected_instrumental_path = os.path.join(SEPARATED_AUDIO_OUTPUT_DIR, f"{base_track_name}_instrumental.wav")
        file_paths.append(expected_vocals_path)
        file_paths.append(expected_instrumental_path) 

        if os.path.exists(expected_vocals_path) or os.path.exists(expected_instrumental_path): 
            print(f"Separated files for '{base_track_name}' (e.g., '{expected_vocals_path}') already exist. Skipping separation.")
            skipped_existing_separated_count +=1
        else:
            if separate_audio_track(mp3_file_path, base_track_name):
                processed_successfully_count += 1
            else:
                failed_processing_count +=1

    clusterize.clusterize(file_paths)
        
    print("\n--- Processing Summary ---")
    print(f"Total videos in playlist: {len(videos_info)}")
    print(f"Tracks newly downloaded: {newly_downloaded_count}")
    print(f"Tracks already downloaded (skipped download): {skipped_existing_download_count}")
    print(f"Tracks successfully processed for separation: {processed_successfully_count}")
    print(f"Tracks already separated (skipped separation): {skipped_existing_separated_count}")
    print(f"Tracks failed to download: {failed_download_count}")
    print(f"Tracks failed during separation processing: {failed_processing_count}")
    print(f"Original MP3s are in subfolders of: {DOWNLOADS_BASE_FOLDER}")
    print(f"Output for separated audio is in: {SEPARATED_AUDIO_OUTPUT_DIR}")
    print("Processing complete.")

if __name__ == "__main__":
    main()
