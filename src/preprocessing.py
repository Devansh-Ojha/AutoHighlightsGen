import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3.0       # analyze 3-second chunks
N_MELS = 128         
MAX_TIME_STEPS = 130 

def audio_to_spectrogram(file_path=None, audio_array=None, sr=SAMPLE_RATE):
    """
    Converts audio into a Spectrogram.
    Accepts EITHER a file_path (for training) OR an audio_array (for inference).
    """
    try:
        if file_path:
            y, sr = librosa.load(file_path, sr=sr, duration=DURATION)
        else:
            y = audio_array
        target_length = int(sr * DURATION)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    
        spec_db = librosa.power_to_db(spec, ref=np.max)

        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())
        if spec_db.shape[1] < MAX_TIME_STEPS:
            spec_db = np.pad(spec_db, ((0, 0), (0, MAX_TIME_STEPS - spec_db.shape[1])))
        else:
            spec_db = spec_db[:, :MAX_TIME_STEPS]
        return spec_db[..., np.newaxis]

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None