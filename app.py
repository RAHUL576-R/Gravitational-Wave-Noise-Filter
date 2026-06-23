import streamlit as st
import torch
from model import Denoiser
import numpy as np

st.title("Gravitational Wave Noise Filter")

# Load model
model = Denoiser()
model.load_state_dict(torch.load("denoiser.pth", map_location=torch.device('cpu')))
model.eval()

# Input
uploaded_file = st.file_uploader("Upload noisy signal (.npy)")

if uploaded_file is not None:
    signal = np.load(uploaded_file)

    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    with torch.no_grad():
        output = model(signal_tensor).numpy()

    st.write("Denoised Output:")
    st.line_chart(output)
