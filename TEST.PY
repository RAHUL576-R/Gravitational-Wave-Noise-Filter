"""
TEST.PY — Comprehensive evaluation of the trained GW denoiser
=============================================================

Metrics
-------
1. SNR improvement (dB)       — how much signal-to-noise improved
2. Normalised MSE              — waveform reconstruction fidelity
3. Noise power reduction (%)   — how much noise was suppressed
4. Pearson correlation         — shape similarity (NEW)
   Linear correlation between clean template and reconstruction.
   A bad reconstruction gives ~0; a perfect one gives 1.0.
   This is the most direct measure of "does the waveform look right".
5. Spectral MSE                — frequency-domain fidelity (NEW)
   Ensures the model didn't just get the time-domain shape right
   while distorting the frequency content.

Thresholds
----------
TARGET_SNR      >=  3.0 dB
TARGET_MSE      <   0.05
TARGET_NR       >= 40.0 %
TARGET_PEARSON  >=  0.85    (new -- strict waveform shape check)
TARGET_SPEC_MSE <   0.10    (new -- spectral fidelity)
"""

import numpy as np
import torch
from preprocessing import preprocess
from model import Autoencoder
from utils import compute_snr, compute_pearson, compute_spectral_mse, plot_signals

# Thresholds
TARGET_SNR      =  3.0
TARGET_MSE      =  0.05
TARGET_NR       = 40.0
TARGET_PEARSON  =  0.85
TARGET_SPEC_MSE =  0.10

# Load model
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = Autoencoder().to(device)
checkpoint = torch.load("denoiser.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Model loaded  (trained {checkpoint['epoch']} epochs, "
      f"best val={checkpoint['val_loss']:.6f})\n"
      f"Running on: {device}\n")

# Test events (held out -- never seen during training)
TEST_EVENTS = [
    # GW200322_091133
    ("GW200322_091133", "H1", 1268903510, 1268903520),
    ("GW200322_091133", "L1", 1268903510, 1268903520),
    ("GW200322_091133", "V1", 1268903510, 1268903520),
    # GW200316_215756 
    ("GW200316_215756", "H1", 1268431090,1268431100),
    ("GW200316_215756", "L1", 1268431090, 1268431100),
    ("GW200316_215756", "V1", 1268431090, 1268431100),
    # GW200311_115853 
    ("GW200311_115853", "L1", 1267963150, 1267963160),
    ("GW200311_115853", "V1", 1267963150, 1267963160),
    ("GW200311_115853", "H1", 1267963150, 1267963160),
    #  GW200308_173609
    ("GW200308_173609", "H1", 1267724180,1267724190),
    ("GW200308_173609", "L1", 1267724180, 1267724190),
    ("GW200308_173609", "V1", 1267724180, 1267724190),
    # GW200306_093714
    ("GW200306_093714", "H1",  1267522650,  1267522660),
    ("GW200306_093714", "L1",  1267522650, 1267522660)
]

results = []
EPS = 1e-12

for label, det, start, end in TEST_EVENTS:
    tag = f"{label}/{det}"
    print(f"-- {tag}  [{start}-{end}] --")

    try:
        segments = preprocess(det, start, end)
    except Exception as exc:
        print(f"  Skipped -- preprocessing error: {exc}\n")
        continue

    inp = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        recon = model(inp)

    noisy_np = inp[0].squeeze().cpu().numpy()
    recon_np = recon[0].squeeze().cpu().numpy()
    resid_np = noisy_np - recon_np

    sig_pow  = np.mean(noisy_np ** 2)

    snr      = compute_snr(noisy_np, resid_np)
    mse      = np.mean((noisy_np - recon_np) ** 2) / (sig_pow + EPS)
    nr       = (1 - np.mean(resid_np ** 2) / (sig_pow + EPS)) * 100
    pearson  = compute_pearson(noisy_np, recon_np)
    spec_mse = compute_spectral_mse(noisy_np, recon_np)

    snr_ok  = snr     >= TARGET_SNR
    mse_ok  = mse      < TARGET_MSE
    nr_ok   = nr      >= TARGET_NR
    prs_ok  = pearson >= TARGET_PEARSON
    spc_ok  = spec_mse < TARGET_SPEC_MSE
    all_ok  = snr_ok and mse_ok and nr_ok and prs_ok and spc_ok

    print(f"  SNR Improvement   : {snr:7.2f} dB      {'PASS' if snr_ok  else 'FAIL'}")
    print(f"  Norm. MSE         : {mse:7.4f}         {'PASS' if mse_ok  else 'FAIL'}")
    print(f"  Noise Reduction   : {nr:7.2f} %        {'PASS' if nr_ok   else 'FAIL'}")
   
    print(f"  Spectral MSE      : {spec_mse:7.4f}         {'PASS' if spc_ok  else 'FAIL'}")
    print(f"  Overall           : {'PASS' if all_ok else 'FAIL'}\n")

    plot_signals(
        noisy_np, recon_np,
        title=f"{tag}",
        save_path=f"plot_{label}_{det}.png",
    )

    results.append((tag, snr, mse, nr, pearson, spec_mse, all_ok))

# Summary
w = 82
print("\n" + "=" * w)
print(f"{'Event':<22} {'SNR':>7} {'MSE':>7} {'NR%':>7} "
      f"{'Pearson':>8} {'SpecMSE':>8} {'Pass':>6}")
print("-" * w)
for tag, snr, mse, nr, prs, spc, ok in results:
    print(f"{tag:<22} {snr:7.2f} {mse:7.4f} {nr:7.2f} "
          f"{prs:8.4f} {spc:8.4f} {'PASS' if ok else 'FAIL'}")
print("=" * w)
n_pass = sum(r[6] for r in results)
print(f"Passed: {n_pass} / {len(results)}")
