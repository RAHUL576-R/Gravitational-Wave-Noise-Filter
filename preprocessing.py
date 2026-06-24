

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.signal.windows import hann
from gwpy.timeseries import TimeSeries

VALID_DETECTORS = {"H1", "L1", "V1"}
SAMPLE_RATE     = 4096
WINDOW_LENGTH   = 4096          # 1 second at 4096 Hz
HOP_LENGTH      = WINDOW_LENGTH // 2   # 50% overlap
GLITCH_THRESH   = 8.0           # σ — discard segment if |peak| exceeds this


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_strain(detector: str, start: int, end: int) -> np.ndarray:
    
    if detector not in VALID_DETECTORS:
        raise ValueError(
            f"Invalid detector '{detector}'. Choose from {sorted(VALID_DETECTORS)}."
        )
    try:
        ts = TimeSeries.fetch_open_data(
            detector, start, end,
            sample_rate=SAMPLE_RATE,
            timeout=120,
        )
    except Exception as exc:
        raise RuntimeError(
            f"GWOSC fetch failed for {detector} [{start}–{end}]: {exc}"
        ) from exc

    data = np.asarray(ts.value, dtype=np.float64)
    if data.size == 0:
        raise RuntimeError(f"Empty data returned for {detector} [{start}–{end}].")
    if not np.isfinite(data).all():
        # Replace NaN/Inf from data gaps with linear interpolation
        mask = ~np.isfinite(data)
        idx  = np.arange(len(data))
        data[mask] = np.interp(idx[mask], idx[~mask], data[~mask])
    return data

def bandpass(
    strain: np.ndarray,
    f_low:  float = 20.0,
    f_high: float = 500.0,
    order:  int   = 8,
    fs:     int   = SAMPLE_RATE,
) -> np.ndarray:
    
    nyq = 0.5 * fs
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, strain)


def asd_whiten(strain: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    
    N   = len(strain)
    # Estimate PSD using Welch method on the whole segment
    freqs, psd = welch(strain, fs=fs, nperseg=min(N, WINDOW_LENGTH * 4))
    # Interpolate PSD to FFT frequency grid
    fft_freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd_interp = np.interp(fft_freqs, freqs, psd)
    asd_interp = np.sqrt(psd_interp)
    asd_interp = np.where(asd_interp < 1e-30, 1e-30, asd_interp)   # avoid /0

    # Whiten in frequency domain
    strain_fft      = np.fft.rfft(strain)
    strain_fft_white = strain_fft / asd_interp
    whitened        = np.fft.irfft(strain_fft_white, n=N)

    # Re-normalise to unit RMS so segments are on comparable scales
    rms = np.std(whitened)
    if rms > 0:
        whitened /= rms
    return whitened


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def make_segments(
    strain:        np.ndarray,
    window_length: int = WINDOW_LENGTH,
    hop_length:    int = HOP_LENGTH,
    glitch_thresh: float = GLITCH_THRESH,
) -> np.ndarray:
    
    win  = hann(window_length, sym=False)
    segs = []
    n    = len(strain)
    i    = 0
    n_glitch = 0
    while i + window_length <= n:
        seg = strain[i : i + window_length] * win
        # Glitch gate
        peak = np.max(np.abs(seg))
        if peak > glitch_thresh:
            n_glitch += 1
        else:
            segs.append(seg.astype(np.float32))
        i += hop_length

    if n_glitch:
        print(f"    ⚡  {n_glitch} glitch segment(s) discarded")
    if not segs:
        raise ValueError(
            "All segments were flagged as glitches or the strain is too short."
        )
    return np.array(segs)          # (N, window_length)



def preprocess(
    detector: str,
    start:    int,
    end:      int,
    f_low:    float = 20.0,
    f_high:   float = 500.0,
) -> np.ndarray:
    
    strain = fetch_strain(detector, start, end)
    strain = bandpass(strain, f_low=f_low, f_high=f_high)
    strain = asd_whiten(strain)
    segs   = make_segments(strain)
    return segs
