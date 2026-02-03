import os
import librosa
import numpy as np
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_audio_from_video(video_path, temp_audio_path="temp_audio.wav"):
    """
    Extracts audio from a video file and saves it as a temporary WAV file.
    """
    try:
        print(f"[Acoustic] Extracting audio from {video_path}...")
        # Load video and export audio
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio_path, logger=None)
        # Close the video file to release the file handle
        video.close()
        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def get_high_energy_timestamps(video_path, chunk_duration=5, threshold_ratio=0.5):
    """
    Analyzes the video's audio and returns a list of (start, end) timestamps
    where the crowd/commentary energy is high.
    
    Args:
        video_path (str): Path to the MP4 file.
        chunk_duration (int): Length of each analysis segment in seconds.
        threshold_ratio (float): Sensitivity (0.0 to 1.0). 
                                 0.5 means keep audio that is at least 50% of the peak volume.
    
    Returns:
        List of tuples: [(start_time, end_time), ...]
    """
    
    # 1. Extract Audio to Temp File (Faster than direct loading)
    temp_audio = extract_audio_from_video(video_path)
    if not temp_audio:
        return []

    print("[Acoustic] Loading audio data...")
    # Load audio with librosa (sr=16000 is standard for speech processing)
    try:
        x, sr = librosa.load(temp_audio, sr=16000)
    except Exception as e:
        print(f"[Error] Librosa could not load audio: {e}")
        return []
    finally:
        # Clean up the temp file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    # 2. Calculate Short-Time Energy
    print("[Acoustic] Calculating energy profile...")
    window_length = int(chunk_duration * sr)
    
    # List comprehension to calculate energy for every chunk
    energy_profile = np.array([
        sum(abs(x[i : i + window_length]**2)) 
        for i in range(0, len(x), window_length)
    ])

    if len(energy_profile) == 0:
        return []

    # 3. Dynamic Thresholding (The "Honors" Logic)
    # Instead of a fixed number, we calculate a threshold based on this specific video.
    # We use the max energy found to normalize expectations.
    peak_energy = np.max(energy_profile)
    dynamic_threshold = peak_energy * threshold_ratio
    
    print(f"[Acoustic] Peak Energy: {peak_energy:.2f} | Threshold: {dynamic_threshold:.2f}")

    # 4. Filter and Group Timestamps
    df = pd.DataFrame(columns=['energy', 'start', 'end'])
    row_index = 0

    for i, energy_val in enumerate(energy_profile):
        if energy_val >= dynamic_threshold:
            start_t = i * chunk_duration
            end_t = (i + 1) * chunk_duration
            # Add small padding (optional, ensures we don't cut words off)
            df.loc[row_index] = [energy_val, start_t, end_t]
            row_index += 1

    if df.empty:
        print("[Acoustic] No high-energy moments found.")
        return []

    # 5. Merge Consecutive Clips
    # This logic combines [0-5s, 5-10s] into [0-10s]
    merged_clips = []
    df = df.sort_values('start')
    
    current_start = df.iloc[0]['start']
    current_end = df.iloc[0]['end']

    for i in range(1, len(df)):
        row = df.iloc[i]
        # If the next clip starts right where (or before) the current one ends
        if row['start'] <= current_end:
            current_end = max(current_end, row['end'])
        else:
            merged_clips.append((current_start, current_end))
            current_start = row['start']
            current_end = row['end']
    
    # Append the final clip
    merged_clips.append((current_start, current_end))

    print(f"[Acoustic] Found {len(merged_clips)} candidate segments.")
    return merged_clips