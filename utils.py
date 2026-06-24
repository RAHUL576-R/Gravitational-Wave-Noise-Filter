
import matplotlib
matplotlib.use("Agg")   # headless / script-safe
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
   
    p_sig  = np.mean(signal ** 2)
    p_noise = np.mean(noise  ** 2)
    if p_noise < 1e-30:
        return float("inf")
    return 10.0 * np.log10(p_sig / p_noise)


def compute_pearson(a: np.ndarray, b: np.ndarray) -> float:
   
    a  = a - np.mean(a)
    b  = b - np.mean(b)
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-30
    return float(num / den)


def compute_spectral_mse(
    signal: np.ndarray,
    recon:  np.ndarray,
    fs:     int = 4096,
) -> float:
   
    n    = len(signal)
    s_mag = np.abs(np.fft.rfft(signal)) / (n / 2)
    r_mag = np.abs(np.fft.rfft(recon))  / (n / 2)
    return float(np.mean((s_mag - r_mag) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_signals(
    noisy:         np.ndarray,
    reconstructed: np.ndarray,
    title:         str  = "GW Denoiser Output",
    save_path:     str  = None,
    fs:            int  = 4096,
) -> None:
    
    if save_path is None:
        save_path = title.replace(" ", "_").replace("/", "-") + ".png"

    t = np.linspace(0, len(noisy) / fs, len(noisy), endpoint=False)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # ── Panel 1: time domain ─────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, noisy,         color="#4C8FD6", lw=0.6, alpha=0.8,
            label="Noisy input")
    ax.plot(t, reconstructed, color="#E85B4A", lw=0.8,
            label="Reconstructed (denoised)")
    ax.set_ylabel("Strain (whitened)")
    ax.set_title("Time Domain")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(t[0], t[-1])

    # ── Panel 2: residual ────────────────────────────────────────────────────
    residual = noisy - reconstructed
    ax = axes[1]
    ax.plot(t, residual, color="#6AAF6A", lw=0.6)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_ylabel("Residual")
    ax.set_title("Noise Residual  (noisy − reconstructed)")
    ax.set_xlim(t[0], t[-1])

    # ── Panel 3: FFT magnitude ───────────────────────────────────────────────
    ax     = axes[2]
    freqs  = np.fft.rfftfreq(len(noisy), d=1.0 / fs)
    s_mag  = np.abs(np.fft.rfft(noisy))         / (len(noisy) / 2)
    r_mag  = np.abs(np.fft.rfft(reconstructed)) / (len(noisy) / 2)
    ax.semilogy(freqs, s_mag, color="#4C8FD6", lw=0.6, alpha=0.8,
                label="Noisy")
    ax.semilogy(freqs, r_mag, color="#E85B4A", lw=0.8, label="Reconstructed")
    ax.set_xlim(20, fs / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT|")
    ax.set_title("Frequency Domain")
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel 4: spectrogram ─────────────────────────────────────────────────
    ax = axes[3]
    f_s, t_s, Sxx = scipy_spectrogram(
        reconstructed, fs=fs,
        nperseg=256, noverlap=224,
        scaling="density",
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-30)
    im = ax.pcolormesh(
        t_s, f_s, Sxx_db,
        vmin=np.percentile(Sxx_db, 5),
        vmax=np.percentile(Sxx_db, 99),
        shading="gouraud", cmap="inferno",
    )
    ax.set_ylim(20, 500)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Reconstructed Signal — Spectrogram")
    fig.colorbar(im, ax=ax, label="Power (dB)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {save_path}")
