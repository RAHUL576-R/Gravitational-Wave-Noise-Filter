import streamlit as st
import torch
import numpy as np
from model import Autoencoder
import h5py
st.title("Gravitational Wave Noise Filter")

@st.cache_resource
def load_model():
    model = Autoencoder()
    checkpoint = torch.load("denoiser.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

model = load_model()
st.success("Model loaded successfully")

uploaded_file = st.file_uploader("Upload noisy signal", type=["npy", "hdf", "h5"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".npy"):
        signal = np.load(uploaded_file)
    elif uploaded_file.name.endswith((".hdf", ".h5")):
        with h5py.File(uploaded_file, "r") as f:
            signal = f[list(f.keys())[0]][()]

    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    if signal_tensor.ndim == 1:
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
    elif signal_tensor.ndim == 2:
        signal_tensor = signal_tensor.unsqueeze(1)

    with torch.no_grad():
        output = model(signal_tensor)

    output = output.squeeze().cpu().numpy()

    st.line_chart(output)