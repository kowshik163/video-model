import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def hist_diff(frame_a, frame_b, bins=32):
    """Compute histogram difference between two frames (grayscale).
    Returns a normalized L1 distance between histograms.
    """
    a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    ha = cv2.calcHist([a], [0], None, [bins], [0, 256])
    hb = cv2.calcHist([b], [0], None, [bins], [0, 256])
    ha = ha.flatten() / (ha.sum() + 1e-8)
    hb = hb.flatten() / (hb.sum() + 1e-8)
    return float(np.abs(ha - hb).sum())


def ssim_score(frame_a, frame_b):
    """Return SSIM score (1.0 identical)."""
    a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    try:
        val = ssim(a, b)
    except Exception:
        val = 1.0
    return float(val)


def detect_cut(prev_frame, frame, hist_thresh=0.6, ssim_thresh=0.5):
    """
    Heuristic cut detector: returns True when a hard cut is detected.
    - Large histogram difference OR low SSIM indicates a cut.
    """
    hdiff = hist_diff(prev_frame, frame)
    s = ssim_score(prev_frame, frame)
    # Normalize hist diff (bins dependent); empirically 0.5-1.0 indicates big change
    if hdiff > hist_thresh or s < ssim_thresh:
        return True, {"hist_diff": hdiff, "ssim": s}
    return False, {"hist_diff": hdiff, "ssim": s}


def motion_spike(prev_frame, frame, flow_magnitude_thresh=20.0):
    """
    Detect large motion spikes by optical flow magnitude between two frames.
    Expect frames in BGR uint8.
    """
    # Convert to grayscale
    a = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    median_mag = float(np.median(mag))
    return (median_mag > flow_magnitude_thresh), {"median_flow": median_mag}


def detect_transition_sequence(frames, window=3):
    """
    Given a short list of frames, detect whether a gradual transition (fade, wipe)
    occurs centered at the last frame. Returns boolean and diagnostics.
    """
    # Simple heuristic: look for monotonic SSIM decrease then increase
    if len(frames) < window:
        return False, {}
    scores = []
    for i in range(1, len(frames)):
        _, info = detect_cut(frames[i-1], frames[i])
        scores.append(info.get('ssim', 1.0))
    # If ssim drops below 0.8 and shows a trough, flag
    if min(scores) < 0.8 and (scores[-1] > scores[0]):
        return True, {"ssim_trend": scores}
    return False, {"ssim_trend": scores}
