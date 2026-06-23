"""
preprocessing.py — GW strain preprocessing pipeline
=====================================================

Pipeline order (physics-motivated):
  fetch → quality-check → bandpass → PSD whiten → segment (with overlap)

Key improvements over original
-------------------------------
* PSD-based spectral whitening replaces naive z-score normalisation.
  Real LIGO noise is coloured (1/f at low freq, shot noise at high freq).
  Dividing by the ASD flattens the noise floor so the model sees a
  stationary white-noise background — the standard LIGO analysis approach.
* Bandpass BEFORE whitening (avoids spectral leakage at band edges).
* 50% overlapping windows (Hann-windowed) doubles effective dataset size
  and prevents boundary artefacts from corrupting segment edges.
* Glitch / data-quality gate: segments with |peak| > 8σ after whitening
  are flagged as likely glitches and discarded.
* Pipeline order is now: bandpass → ASD whiten → segment → glitch-gate.
"""

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
    """
    Fetch open strain data from GWOSC.

    Parameters
    ----------
    detector : 'H1' | 'L1' | 'V1'
    start, end : GPS seconds

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Signal conditioning
# ─────────────────────────────────────────────────────────────────────────────

def bandpass(
    strain: np.ndarray,
    f_low:  float = 20.0,
    f_high: float = 500.0,
    order:  int   = 8,
    fs:     int   = SAMPLE_RATE,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass.
    8th-order gives steeper roll-off than the original 4th-order,
    better suppressing the low-freq seismic wall and high-freq shot noise.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    return filtfilt(b, a, strain)


def asd_whiten(strain: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Spectral (ASD) whitening — the standard LIGO pre-processing step.

    Divides the strain FFT by the estimated one-sided amplitude spectral
    density (ASD = sqrt(PSD)), then inverse-FFTs back to time domain.
    This flattens the coloured LIGO noise floor to approximately white,
    so every frequency bin contributes equally to the loss.

    Why this matters
    ----------------
    z-score normalisation (original code) only removes the mean and scales
    by total RMS — it does nothing about the spectral shape.  A 60 Hz power-
    line artefact still dominates after z-score but is properly suppressed
    after ASD whitening.
    """
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
    """
    Hann-windowed overlapping segments with glitch gating.

    Parameters
    ----------
    strain        : 1-D whitened strain array
    window_length : samples per segment (default 4096 = 1 s)
    hop_length    : step between windows (default 2048 = 50% overlap)
    glitch_thresh : discard segment if |peak| > this many σ

    Returns
    -------
    np.ndarray, shape (N_good, window_length)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    detector: str,
    start:    int,
    end:      int,
    f_low:    float = 20.0,
    f_high:   float = 500.0,
) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    fetch → bandpass → ASD-whiten → Hann-segment (50% overlap) → glitch-gate

    Returns
    -------
    np.ndarray, shape (N_segments, WINDOW_LENGTH)
    """
    strain = fetch_strain(detector, start, end)
    strain = bandpass(strain, f_low=f_low, f_high=f_high)
    strain = asd_whiten(strain)
    segs   = make_segments(strain)
    return segs
