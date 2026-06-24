"""

"""

import numpy as np
import torch
import streamlit as st

from preprocessing import preprocess, SAMPLE_RATE, WINDOW_LENGTH
from utils import compute_snr, compute_pearson, compute_spectral_mse, plot_signals
from model import Autoencoder

TARGET_SNR      =  3.0
TARGET_MSE      =  0.05
TARGET_NR       = 40.0
TARGET_PEARSON  =  0.85
TARGET_SPEC_MSE =  0.10
EPS             =  1e-12


def evaluate(noisy, recon):
    residual = noisy - recon
    sig_pow  = np.mean(noisy ** 2)
    return dict(
        snr      = compute_snr(noisy, residual),
        mse      = np.mean((noisy - recon) ** 2) / (sig_pow + EPS),
        nr       = (1 - np.mean(residual ** 2) / (sig_pow + EPS)) * 100,
        pearson  = compute_pearson(noisy, recon),
        spec_mse = compute_spectral_mse(noisy, recon),
    )

def passes(m):
    return dict(
        snr      = m["snr"]      >= TARGET_SNR,
        mse      = m["mse"]       < TARGET_MSE,
        nr       = m["nr"]       >= TARGET_NR,
        pearson  = m["pearson"]  >= TARGET_PEARSON,
        spec_mse = m["spec_mse"]  < TARGET_SPEC_MSE,
    )


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Autoencoder().to(device)
    ckpt = torch.load("denoiser.pth", map_location=device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, device


@st.cache_data(show_spinner=False)
def cached_preprocess(detector, gps_start, gps_end):
    """Full preprocess — returns ALL segments exactly like TEST.PY."""
    segs = preprocess(detector, gps_start, gps_end)   # (N, WINDOW_LENGTH)
    return segs.astype(np.float32)


@st.cache_data(show_spinner=False)
def cached_run_and_plot(segments_bytes, n_segs, title):
    """
    Run ALL segments as one batch — identical to TEST.PY.
    Evaluate and plot only segment[0].
    """
    segments = np.frombuffer(segments_bytes, dtype=np.float32).copy().reshape(n_segs, WINDOW_LENGTH)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model()

    # Batch inference — identical to TEST.PY
    inp_t = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        recon_t = model(inp_t)

    # Only segment[0] — identical to TEST.PY
    noisy_np = inp_t[0].squeeze().cpu().numpy()
    recon_np = recon_t[0].squeeze().cpu().numpy()

    m  = evaluate(noisy_np, recon_np)
    pf = passes(m)
    ok = all(pf.values())

    save_path = "/tmp/plot_result.png"
    plot_signals(noisy_np, recon_np, title=title, save_path=save_path, fs=SAMPLE_RATE)
    with open(save_path, "rb") as f:
        png_bytes = f.read()

    return m, pf, ok, png_bytes


# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GW Denoiser", page_icon="🌊", layout="wide")
st.title("🌊 Gravitational Wave Noise Filter")

try:
    model, device = load_model()
    st.sidebar.success(f"Model loaded ✅ | Device: **{device}**")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Pass Thresholds")
    st.markdown(f"""
| Metric | Threshold |
|---|---|
| SNR Improvement | ≥ {TARGET_SNR} dB |
| Norm. MSE | < {TARGET_MSE} |
| Noise Reduction | ≥ {TARGET_NR} % |
| Pearson Correlation | ≥ {TARGET_PEARSON} |
| Spectral MSE | < {TARGET_SPEC_MSE} |
""")

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.subheader("Enter Event Parameters")
col1, col2, col3, col4 = st.columns([2, 3, 1, 3])
detector  = col1.radio("Detector", ["H1", "L1", "V1"], horizontal=True)
gps_start = col2.number_input("GPS Start", value=1268903510, step=1, format="%d")
col3.markdown("<br><br>to", unsafe_allow_html=True)
gps_end   = col4.number_input("GPS End", value=1268903520, step=1, format="%d")

if gps_end <= gps_start:
    st.error("GPS End must be greater than GPS Start.")
    st.stop()

run = st.button("▶  Run Denoiser", type="primary", use_container_width=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result   = None
if "last_run" not in st.session_state:
    st.session_state.last_run = None

# ── Run ────────────────────────────────────────────────────────────────────────
if run:
    st.session_state.result   = None
    st.session_state.last_run = (detector, int(gps_start), int(gps_end))

    with st.spinner(f"Fetching {detector} [{gps_start}–{gps_end}] from GWOSC…"):
        try:
            segments = cached_preprocess(detector, int(gps_start), int(gps_end))
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    with st.spinner("Running denoiser…"):
        title    = f"{detector} | GPS {gps_start}–{gps_end}"
        n_segs   = len(segments)
        m, pf, ok, png_bytes = cached_run_and_plot(
            segments.tobytes(), n_segs, title
        )

    st.session_state.result = (m, pf, ok, png_bytes)

# ── Display ────────────────────────────────────────────────────────────────────
if st.session_state.result is not None:
    m, pf, ok, png_bytes = st.session_state.result
    det, gs, ge = st.session_state.last_run

    st.divider()
    badge = "✅ PASS" if ok else "❌ FAIL"
    st.subheader(f"Result — {det} | GPS {gs}–{ge}  {badge}")

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, name, key, fmt in zip(
        [col1, col2, col3, col4, col5],
        ["SNR (dB)", "Norm. MSE", "Noise Red. %", "Pearson", "Spectral MSE"],
        ["snr", "mse", "nr", "pearson", "spec_mse"],
        [".2f", ".4f", ".2f", ".4f", ".4f"],
    ):
        icon = "✅" if pf[key] else "❌"
        col.metric(f"{icon} {name}", f"{m[key]:{fmt}}")

    st.image(png_bytes, use_column_width=True)

    if ok:
        st.success("All metrics passed.")
    else:
        failed = [k for k, v in pf.items() if not v]
        st.warning(f"Failed metrics: {', '.join(failed)}")