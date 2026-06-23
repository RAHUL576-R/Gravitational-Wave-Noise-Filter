import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Autoencoder

# ── Model loading ────────────────────────────

def load_model():
    model = Autoencoder()
    checkpoint = torch.load("denoiser.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

model = load_model()


def denoise_npy_file(noisy_file):
    signal = np.load(noisy_file)
    signal_tensor = torch.tensor(signal, dtype=torch.float32)

    if signal_tensor.ndim == 1:
        signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)
    elif signal_tensor.ndim == 2:
        signal_tensor = signal_tensor.unsqueeze(1)

    with torch.no_grad():
        output = model(signal_tensor)

    output = output.squeeze().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(output)
    ax.set_title("Denoised Output")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


demo = gr.Interface(
    fn=denoise_npy_file,
    inputs=gr.File(label="Upload noisy signal (.npy)"),
    outputs=gr.Plot(label="Denoised Signal"),
    title="Gravitational Wave Noise Filter",
    description="Upload a noisy gravitational-wave signal saved as a NumPy .npy file and view the denoised waveform.",
)

if __name__ == "__main__":
    demo.launch()
