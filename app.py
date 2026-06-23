import streamlit as st

st.set_page_config(page_title="GW Filter", layout="wide")
import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import torch
import numpy as np
from model import Autoencoder

st.title("Gravitational Wave Noise Filter")

# ── Load model ─────────────────────────────
@st.cache_resource
def load_model():
    model = Autoencoder()
    checkpoint = torch.load("denoiser.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

model = load_model()
st.success("Model loaded successfully")

# ── File upload ─────────────────────────────
uploaded_file = st.file_uploader("Upload noisy signal (.npy)")

if uploaded_file is not None:
    signal = np.load(uploaded_file)

    # ── Convert to correct shape (B, 1, L) ──
    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    if signal_tensor.ndim == 1:
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
    elif signal_tensor.ndim == 2:
        signal_tensor = signal_tensor.unsqueeze(1)

    # ── Inference ─────────────────────────────
    with torch.no_grad():
        output = model(signal_tensor)

    output = output.squeeze().cpu().numpy()

    st.write("Denoised Output:")
    st.line_chart(output)
