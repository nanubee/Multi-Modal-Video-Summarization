import sys
import os

# --- HONORS PROJECT: IMPORT CUSTOM ACOUSTIC LOGIC ---
from acoustic_processor import get_high_energy_timestamps 
from moviepy.video.io.VideoFileClip import VideoFileClip
# ----------------------------------------------------

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
  # We keep create_clips to cut the video
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import re

def create_clips(video_path, timestamps, output_folder, pad_start=0.0, pad_end=0.0):
    """
    Cuts the video based on timestamps and saves them to the output folder.
    """
    clip_paths = []
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load video once
    try:
        video = VideoFileClip(video_path)
    except Exception as e:
        print(f"Error loading video: {e}")
        return []

    for i, (start, end) in enumerate(timestamps):
        # Apply padding and ensure we don't go past the video end
        start_time = max(0, start - pad_start)
        end_time = min(video.duration, end + pad_end)
        
        filename = f"highlight_{i+1}_{int(start_time)}-{int(end_time)}.mp4"
        output_path = os.path.join(output_folder, filename)
        
        try:
            print(f"Cutting clip {i+1}: {start_time} to {end_time}")
            
            # The robust MoviePy method (works on all versions)
            subclip = video.subclipped(start_time, end_time)
            
            # Write the file (codec='libx264' ensures it works in browsers)
            subclip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True,
                logger=None # Hides the noisy progress bars
            )
            
            clip_paths.append(output_path)
        except Exception as e:
            print(f"Error cutting clip {i+1}: {e}")

    # Close the video file to release memory
    video.close()
    return clip_paths

class Config:
    def __init__(self, config_file_path):
        config_default = {
            "use_gpu": False,
            "auto_load_model": False,
            "segment_length": 600,
            "minimum_clip_length": 5,
            "maximum_clip_length": 30,
            "pad_clip_start": 0.0, # Changed default to 0 for precision
            "pad_clip_end": 0.0,
            "number_of_clips": 5,  # Increased default
            "threshold": 0.5,      # Re-purposed as "Audio Energy Threshold"
            "leniency": 2
        }

        if not os.path.exists(config_file_path):
            with open(config_file_path, "w") as f:
                json.dump(config_default, f, indent="\t")

        with open(config_file_path, "r") as f:
            config = json.load(f)

        # We keep these variables so the Frontend doesn't break, 
        # but we might not use all of them.
        self.use_gpu = config.get("use_gpu", False)
        self.auto_load_model = config.get("auto_load_model", False)
        self.segment_length = config.get("segment_length", 600)

        self.minimum_clip_length = config.get("minimum_clip_length", 5)
        self.maximum_clip_length = config.get("maximum_clip_length", 30)
        self.pad_clip_start = config.get("pad_clip_start", 0.0)
        self.pad_clip_end = config.get("pad_clip_end", 0.0)
        self.number_of_clips = config.get("number_of_clips", 2)

        self.threshold = config.get("threshold", 0.7)
        self.leniency = config.get("leniency", 2)

    def get_device(self):
        return "cuda" if self.use_gpu else "cpu"


def get_folder_name():
    return datetime.now().strftime("%Y-%m-%d")


def numerical_sort(value):
    numbers = re.findall(r"\d+", value)
    return int(numbers[0]) if numbers else 0


def get_files(clip_folder):
    folder_list = os.listdir(clip_folder)
    all_files = {}

    for folder in folder_list:
        folder_path = os.path.join(clip_folder, folder)
        files = sorted(os.listdir(folder_path), key=numerical_sort)
        all_files[folder] = files

    return all_files


app = Flask(__name__)
video_folder = os.path.abspath("./static/uploads")
clip_folder = os.path.abspath("./static/clips")
static_folder = os.path.abspath("./static")
output_folder = os.path.abspath(os.path.join(clip_folder, get_folder_name()))

os.makedirs(video_folder, exist_ok=True)
os.makedirs(clip_folder, exist_ok=True)

config_file_path = os.path.abspath("./config.json")
config = Config(config_file_path)

# NOTE: We removed the 'load_model' call here because we are using Acoustic Logic now.

@app.route("/", methods=["GET", "POST"])
def main():
    
    if request.method == "POST":
        if "video" in request.files:
            try:
                video = request.files["video"]
                if video:
                    print("Processing video...")

                    filename = secure_filename(video.filename)
                    video_path = os.path.join(video_folder, filename)
                    video.save(video_path)

                    # --- HONORS PROJECT: ACOUSTIC LOGIC START ---
                    print("--- Starting Acoustic Analysis ---")
                    
                    # We utilize the 'threshold' slider from the UI to control sensitivity
                    # If user sets 0.7 on UI, we look for top 30% loudest moments (1 - 0.7 approx)
                    # Or we can just map it directly. Let's map it directly.
                    ui_threshold = config.threshold 
                    
                    # Call our custom processor
                    # video_path: Where the user uploaded the file
                    # chunk_duration: 5s (standard for cricket)
                    # threshold_ratio: The sensitivity from the config
                    timestamps = get_high_energy_timestamps(
                        video_path, 
                        chunk_duration=5, 
                        threshold_ratio=ui_threshold
                    )
                    
                    print(f"Acoustic Processor found {len(timestamps)} clips.")
                    
                    # Convert format for the video cutter
                    # The cutter likely expects specific formatting, but create_clips usually takes [(start, end)]
                    # Our timestamps are already [(start, end)]
                    
                    print("Creating clips from Acoustic Timestamps...")
                    
                    # We pass our acoustic timestamps directly to the existing clipper
                    clip_paths = create_clips(
                        video_path, 
                        timestamps, 
                        output_folder, 
                        config.pad_clip_start, 
                        config.pad_clip_end
                    )
                    
                    clip_urls = [os.path.relpath(clip_path, static_folder).replace("\\", "/") for clip_path in clip_paths]

                    print("Done!")
                    # --- HONORS PROJECT: ACOUSTIC LOGIC END ---

                    return render_template("index.html", config=config, clips=clip_urls, folders=get_files(clip_folder))

            except Exception as e:
               print(f"ERROR: {e}")

            finally:
                # Clean up the uploaded video to save space
                if len(os.listdir(video_folder)) != 0:
                    for path in os.listdir(video_folder):
                        try:
                            os.remove(os.path.join(video_folder, path))
                        except:
                            pass

    return render_template("index.html", config=config, folders=get_files(clip_folder))


@app.route("/get-config", methods=["POST"])
def get_config():
    try:
        # We removed 'global model' logic because we don't need to reload a heavy AI model anymore.
        
        config.use_gpu = request.form.get("use-gpu") == "on"
        config.auto_load_model = request.form.get("auto-load-model") == "on"
        config.segment_length = int(request.form.get("segment-length"))

        config.minimum_clip_length = int(request.form.get("minimum-clip-length"))
        config.maximum_clip_length = int(request.form.get("maximum-clip-length"))
        config.pad_clip_start = float(request.form.get("pad-clip-start"))
        config.pad_clip_end = float(request.form.get("pad-clip-end"))
        config.number_of_clips = int(request.form.get("number-of-clips"))

        config.threshold = float(request.form.get("threshold"))
        config.leniency = int(request.form.get("leniency"))

    except ValueError as e:
        print(e)

    finally:
        return jsonify({"status": "success", "message": "Settings succesfully updated"})


@app.route("/save-config", methods=["POST"])
def save_config():
    with open(config_file_path, "w") as f:
        json.dump(config.__dict__, f, indent="\t")
    
    return jsonify({"status": "success", "message": "Settings succesfully updated"})


if __name__ == "__main__":
    app.run(port=5000)