try:
    import librosa
    import numpy as np
except Exception:
    librosa = None


def detect_audio_beats(video_path, sr=22050):
    """
    Extracts audio and detects beat timestamps using librosa, if available.
    Returns a list of beat times in seconds. If librosa not installed, returns [].
    """
    if librosa is None:
        # librosa not available; return empty and caller can handle.
        return []
    try:
        import soundfile as sf
        # Load audio from video using librosa (ffmpeg-backed)
        y, _ = librosa.load(video_path, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return beat_times.tolist()
    except Exception:
        return []
