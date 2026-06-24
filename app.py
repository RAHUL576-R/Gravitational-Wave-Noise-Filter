"""
app.py — Streamlit deployment of the GW denoiser
"""

import io
import numpy as np
import torch
import streamlit as st

from preprocessing import preprocess, SAMPLE_RATE
from utils import compute_snr, compute_pearson, compute_spectral_mse, plot_signals
from model import Autoencoder

# ── Thresholds ─────────────────────────────────────────────────────────────────
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


# ── Cached functions ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_preprocess(detector, gps_start, gps_end):
    return preprocess(detector, gps_start, gps_end)


@st.cache_data(show_spinner=False)
def cached_run_segment(segment_bytes, seg_idx):
    noisy_np = np.frombuffer(segment_bytes, dtype=np.float32).copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model()
    inp_t = torch.tensor(noisy_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon_np = model(inp_t).squeeze().cpu().numpy()
    m  = evaluate(noisy_np, recon_np)
    pf = passes(m)
    ok = all(pf.values())
    return noisy_np, recon_np, m, pf, ok


@st.cache_data(show_spinner=False)
def cached_plot(noisy_bytes, recon_bytes, title, seg_idx):
    noisy_np = np.frombuffer(noisy_bytes, dtype=np.float32).copy()
    recon_np = np.frombuffer(recon_bytes, dtype=np.float32).copy()
    save_path = f"/tmp/plot_seg{seg_idx}.png"
    plot_signals(noisy_np, recon_np, title=title, save_path=save_path, fs=SAMPLE_RATE)
    with open(save_path, "rb") as f:
        return f.read()


# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GW Denoiser", page_icon="🌊", layout="wide")
st.title("🌊 Gravitational Wave Noise Filter")

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Autoencoder().to(device)
    ckpt = torch.load("denoiser.pth", map_location=device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, device

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

# ── Session state — results persist across reruns ──────────────────────────────
if "results" not in st.session_state:
    st.session_state.results     = None
    st.session_state.last_run    = None   # (detector, gps_start, gps_end)

if run:
    # Clear old results if inputs changed
    st.session_state.results  = None
    st.session_state.last_run = (detector, int(gps_start), int(gps_end))

    # Fetch + preprocess
    with st.spinner(f"Fetching {detector} strain [{gps_start} – {gps_end}] from GWOSC…"):
        try:
            segments = cached_preprocess(detector, int(gps_start), int(gps_end))
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    n_segs = len(segments)
    st.write(f"**Evaluating all {n_segs} segments**")

    # Inference
    all_results = []
    progress = st.progress(0, text="Running denoiser…")
    for i in range(n_segs):
        seg_bytes = segments[i].astype(np.float32).tobytes()
        noisy_np, recon_np, m, pf, ok = cached_run_segment(seg_bytes, i)
        all_results.append((i, noisy_np, recon_np, m, pf, ok))
        progress.progress((i + 1) / n_segs, text=f"Segment {i+1}/{n_segs}")
    progress.empty()

    # Store in session state so results survive any rerun
    st.session_state.results = all_results

# ── Display results if available ───────────────────────────────────────────────
if st.session_state.results is not None:
    all_results = st.session_state.results
    det, gs, ge = st.session_state.last_run

    st.divider()
    st.subheader("Per-Segment Results")

    for i, noisy_np, recon_np, m, pf, ok in all_results:
        with st.expander(f"Segment {i+1}  —  {'✅ PASS' if ok else '❌ FAIL'}", expanded=(i == 0)):
            col1, col2 = st.columns([1, 2])

            with col1:
                def row(name, val, fmt, passed):
                    return f"| {name} | {val:{fmt}} | {'✅' if passed else '❌'} |"
                st.markdown("\n".join([
                    "| Metric | Value | |",
                    "|---|---|---|",
                    row("SNR (dB)",     m["snr"],      ".2f", pf["snr"]),
                    row("Norm. MSE",    m["mse"],      ".4f", pf["mse"]),
                    row("Noise Red. %", m["nr"],       ".2f", pf["nr"]),
                    row("Pearson",      m["pearson"],  ".4f", pf["pearson"]),
                    row("Spectral MSE", m["spec_mse"], ".4f", pf["spec_mse"]),
                ]))

            with col2:
                title = f"{det} | GPS {gs}–{ge} | Segment {i+1}"
                png_bytes = cached_plot(
                    noisy_np.astype(np.float32).tobytes(),
                    recon_np.astype(np.float32).tobytes(),
                    title, i
                )
                st.image(png_bytes, use_column_width=True)

    # Summary
    st.divider()
    st.subheader("Summary")

    header = f"{'Segment':<10} {'SNR':>7} {'MSE':>7} {'NR%':>7} {'Pearson':>8} {'SpecMSE':>8} {'Pass':>6}"
    rows_txt = [header, "-" * 55]
    for i, _, _, m, pf, ok in all_results:
        rows_txt.append(
            f"{'Seg '+str(i+1):<10} {m['snr']:7.2f} {m['mse']:7.4f} {m['nr']:7.2f} "
            f"{m['pearson']:8.4f} {m['spec_mse']:8.4f} {'PASS' if ok else 'FAIL':>6}"
        )
    st.code("\n".join(rows_txt))

    n_pass = sum(r[5] for r in all_results)
    st.metric("Segments passed", f"{n_pass} / {len(all_results)}")
    if n_pass == len(all_results):
        st.success("All evaluated segments passed every threshold.")
    else:
        st.warning(f"{len(all_results) - n_pass} segment(s) failed one or more thresholds.")