import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip, concatenate_videoclips
from preprocessing import audio_to_spectrogram, SAMPLE_RATE, DURATION

# Settings
MODEL_PATH = "../cricket_model.h5"
VIDEO_PATH = "../input_video/match.mp4"
OUTPUT_PATH = "../output/final_highlights.mp4"

def generate_video():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Run cnn.py first!")
        return

    model = load_model(MODEL_PATH)
    
    print(f"Scanning Video: {VIDEO_PATH}")
    # Load entire video audio to scan it fast
    y, sr = librosa.load(VIDEO_PATH, sr=SAMPLE_RATE)
    
    step = int(SAMPLE_RATE * DURATION) 
    timestamps = []
    
    total_chunks = len(y) // step
    print(f"Analyzing {total_chunks} audio segments")

    for i in range(0, len(y) - step, step):
        chunk = y[i : i + step]
        # Convert to Spectrogram
        spec = audio_to_spectrogram(audio_array=chunk, sr=sr)
        
        # Predict (Reshape to 1, 128, 130, 1)
        score = model.predict(np.array([spec]), verbose=0)[0][0]
    
        if score > 0.8:
            seconds = i / SAMPLE_RATE
            timestamps.append(seconds)

    print(f"Found {len(timestamps)} exciting moments. Merging")
    
    final_clips = []
    if timestamps:
        video = VideoFileClip(VIDEO_PATH)
        
        #  If spikes are within 10s, merge them
        merged_intervals = []
        if len(timestamps) > 0:
            start = timestamps[0]
            end = timestamps[0]
            
            for t in timestamps[1:]:
                if t - end < 10: 
                    end = t
                else:
                    merged_intervals.append((start, end))
                    start = t
                    end = t
            merged_intervals.append((start, end))
        
        # Cut the video
        for start, end in merged_intervals:
            # Add buffer (-2s start, +5s end)
            clip_start = max(0, start - 2)
            clip_end = min(video.duration, end + 5)
            final_clips.append(video.subclip(clip_start, clip_end))

        # Stitch
        print(" Rendering Final Video")
        final_video = concatenate_videoclips(final_clips)
        final_video.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")
        print(f"Done! Saved to {OUTPUT_PATH}")
    else:
        print("No excitement found. Your AI thinks this match was boring.")

if __name__ == "__main__":
    generate_video()