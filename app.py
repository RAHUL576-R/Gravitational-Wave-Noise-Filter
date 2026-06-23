"""
app.py — Streamlit deployment of the GW denoiser
Uses preprocessing.py and utils.py directly — identical to TEST.PY.

Pipeline: HDF5 upload → bandpass → ASD-whiten → Hann-segment → glitch-gate
          → Autoencoder → 5 metrics (SNR, MSE, NR%, Pearson, Spectral MSE)
"""

import io
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Same imports as TEST.PY ────────────────────────────────────────────────────
from preprocessing import bandpass, asd_whiten, make_segments, SAMPLE_RATE, WINDOW_LENGTH
from utils import compute_snr, compute_pearson, compute_spectral_mse, plot_signals
from model import Autoencoder

# ── Thresholds (identical to TEST.PY) ─────────────────────────────────────────
TARGET_SNR      =  3.0
TARGET_MSE      =  0.05
TARGET_NR       = 40.0
TARGET_PEARSON  =  0.85
TARGET_SPEC_MSE =  0.10
EPS             =  1e-12


# ── HDF5 loader ────────────────────────────────────────────────────────────────

def load_hdf5_strain(file_obj) -> tuple[np.ndarray, float]:
    """
    Read strain + sample-rate from an HDF5 file.

    Strategy (in order):
      1. Try known GWOSC paths: /strain/Strain, /strain, /data
      2. Walk ALL nodes with visititems (always yields str keys, never tuples)
         and pick the largest 1-D Dataset — most likely to be the strain.
      3. Raise with a full structure dump so the user knows what's in the file.
    """
    with h5py.File(file_obj, "r") as f:

        # ── 1. Known paths ────────────────────────────────────────────────────
        for path in ("/strain/Strain", "/strain", "/data"):
            if path in f and isinstance(f[path], h5py.Dataset):
                strain = f[path][:].flatten().astype(np.float64)
                # Sample rate from metadata attrs or groups
                fs = _read_fs(f)
                return strain, fs

        # ── 2. Walk all nodes, collect every 1-D Dataset ─────────────────────
        candidates = {}   # path -> size
        def _collect(name, obj):
            # name is always a str here — h5py guarantee
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
                candidates[name] = obj.size

        f.visititems(_collect)

        if candidates:
            # Pick the dataset with the most samples (the strain)
            best = max(candidates, key=candidates.__getitem__)
            strain = f[best][:].flatten().astype(np.float64)
            fs     = _read_fs(f)
            return strain, fs

        # ── 3. Nothing found — dump structure for debugging ───────────────────
        structure = []
        def _dump(name, obj):
            kind = "Dataset" if isinstance(obj, h5py.Dataset) else "Group"
            shape = obj.shape if hasattr(obj, "shape") else ""
            structure.append(f"  {kind}: /{name}  {shape}")
        f.visititems(_dump)
        raise ValueError(
            "Could not locate a strain Dataset.\n"
            "File structure:\n" + "\n".join(structure)
        )


def _read_fs(f: h5py.File) -> float:
    """Try common locations for the sample rate; default to SAMPLE_RATE."""
    # Check top-level attributes
    for attr in ("sample_rate", "SampleRate", "fs", "Fs"):
        if attr in f.attrs:
            return float(f.attrs[attr])
    # Check meta / metadata groups
    for group in ("meta", "metadata"):
        if group in f:
            g = f[group]
            for attr in ("SampleRate", "sample_rate", "fs"):
                if attr in g.attrs:
                    return float(g.attrs[attr])
                if attr in g and isinstance(g[attr], h5py.Dataset):
                    val = g[attr][()]
                    return float(val.flat[0] if hasattr(val, "flat") else val)
    return float(SAMPLE_RATE)


# ── Metric helpers (same logic as TEST.PY) ─────────────────────────────────────

def evaluate(noisy: np.ndarray, recon: np.ndarray) -> dict:
    residual = noisy - recon
    sig_pow  = np.mean(noisy ** 2)
    snr      = compute_snr(noisy, residual)
    mse      = np.mean((noisy - recon) ** 2) / (sig_pow + EPS)
    nr       = (1 - np.mean(residual ** 2) / (sig_pow + EPS)) * 100
    pearson  = compute_pearson(noisy, recon)
    spec_mse = compute_spectral_mse(noisy, recon)
    return dict(snr=snr, mse=mse, nr=nr, pearson=pearson, spec_mse=spec_mse)


def passes(m: dict) -> dict:
    return dict(
        snr      = m["snr"]      >= TARGET_SNR,
        mse      = m["mse"]       < TARGET_MSE,
        nr       = m["nr"]       >= TARGET_NR,
        pearson  = m["pearson"]  >= TARGET_PEARSON,
        spec_mse = m["spec_mse"]  < TARGET_SPEC_MSE,
    )


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="GW Denoiser", page_icon="🌊", layout="wide")
st.title("🌊 Gravitational Wave Noise Filter")
st.markdown(
    "Upload an HDF5 strain file (`.hdf5` / `.h5`). "
    "The app runs the **exact same pipeline and metrics as the test suite**."
)

# ── Load model once ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m      = Autoencoder().to(device)
    ckpt   = torch.load("denoiser.pth", map_location=device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, device

try:
    model, device = load_model()
    st.sidebar.success(f"Model loaded ✅  |  Device: **{device}**")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
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
    max_segs = st.slider("Max segments to evaluate", 1, 20, 5)

# ── File upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload HDF5 strain file", type=["hdf5", "h5"])
if uploaded is None:
    st.info("Waiting for a file…")
    st.stop()

# ── Load strain ────────────────────────────────────────────────────────────────
with st.spinner("Reading HDF5 file…"):
    try:
        strain, fs = load_hdf5_strain(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

st.write(
    f"**Samples:** {len(strain):,}  |  **Sample rate:** {fs:.0f} Hz  "
    f"|  **Duration:** {len(strain)/fs:.2f} s"
)

# ── Preprocessing — identical pipeline to preprocessing.preprocess() ───────────
with st.spinner("Bandpass filtering (20–500 Hz, 8th-order Butterworth)…"):
    try:
        strain = bandpass(strain, fs=int(fs))
    except Exception as e:
        st.warning(f"Bandpass failed ({e}); using raw strain.")

with st.spinner("ASD whitening…"):
    strain = asd_whiten(strain, fs=int(fs))

with st.spinner("Segmenting (Hann window, 50% overlap, glitch-gating)…"):
    try:
        segments = make_segments(strain)           # (N, WINDOW_LENGTH)
    except ValueError as e:
        st.error(str(e))
        st.stop()

n_segs = min(len(segments), max_segs)
st.write(f"**Segments available:** {len(segments)}  |  **Evaluating:** {n_segs}")

# ── Inference + metrics ────────────────────────────────────────────────────────
all_results = []
progress = st.progress(0, text="Running denoiser…")

for i in range(n_segs):
    noisy_np = segments[i]                                              # (WINDOW_LENGTH,)
    inp_t    = torch.tensor(noisy_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,N)

    with torch.no_grad():
        recon_t = model(inp_t)

    recon_np = recon_t.squeeze().cpu().numpy()
    m        = evaluate(noisy_np, recon_np)
    pf       = passes(m)
    ok       = all(pf.values())
    all_results.append((i, noisy_np, recon_np, m, pf, ok))
    progress.progress((i + 1) / n_segs, text=f"Segment {i+1}/{n_segs}")

progress.empty()

# ── Per-segment display ────────────────────────────────────────────────────────
st.divider()
st.subheader("Per-Segment Results")

for i, noisy_np, recon_np, m, pf, ok in all_results:
    badge = "✅ PASS" if ok else "❌ FAIL"
    with st.expander(f"Segment {i+1}  —  {badge}", expanded=(i == 0)):
        col1, col2 = st.columns([1, 2])

        with col1:
            def row(name, val, fmt, passed):
                icon = "✅" if passed else "❌"
                return f"| {name} | {val:{fmt}} | {icon} |"

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
            # Use plot_signals from utils.py — save to buffer then display
            buf = io.BytesIO()
            save_path = f"/tmp/plot_seg{i+1}.png"
            plot_signals(
                noisy_np, recon_np,
                title=f"Segment {i+1}",
                save_path=save_path,
                fs=SAMPLE_RATE,
            )
            with open(save_path, "rb") as f:
                st.image(f.read(), use_container_width=True)

# ── Summary table — mirrors TEST.PY console output ─────────────────────────────
st.divider()
st.subheader("Summary")

import pandas as pd
rows = []
for i, _, _, m, pf, ok in all_results:
    rows.append({
        "Segment":      f"Seg {i+1}",
        "SNR (dB)":     round(m["snr"],      2),
        "Norm. MSE":    round(m["mse"],      4),
        "Noise Red %":  round(m["nr"],       2),
        "Pearson":      round(m["pearson"],  4),
        "Spectral MSE": round(m["spec_mse"], 4),
        "Overall":      "PASS" if ok else "FAIL",
    })

df = pd.DataFrame(rows)

def highlight(val):
    if val == "PASS": return "background-color:#1a4a2e; color:#2ECC71; font-weight:bold"
    if val == "FAIL": return "background-color:#4a1a1a; color:#E74C3C; font-weight:bold"
    return ""

st.dataframe(
    df.style.applymap(highlight, subset=["Overall"]),
    use_container_width=True, hide_index=True
)

n_pass = sum(r[5] for r in all_results)
st.metric("Segments passed", f"{n_pass} / {n_segs}")

if n_pass == n_segs:
    st.success("All evaluated segments passed every threshold.")
else:
    st.warning(f"{n_segs - n_pass} segment(s) failed one or more thresholds.")