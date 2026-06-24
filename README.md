# GW-Denoiser — WaveUNet for Gravitational-Wave Strain Denoising

A deep-learning pipeline that removes LIGO detector noise from gravitational-wave strain data using a WaveUNet autoencoder trained on GWTC events.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Noise Model](#noise-model)
3. [Dataset Guide](#dataset-guide)
4. [Evaluation Methodology](#evaluation-methodology)
5. [Quickstart](#quickstart)
6. [File Reference](#file-reference)

---

## Architecture

The model is a 1-D WaveUNet autoencoder that operates directly on whitened strain time series at 1024 Hz.

```
Input  (1, 4096)   ← 4 s window at 1024 Hz, whitened, unit-std normalised
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENCODER  (strided Conv1d blocks — each halves the time axis)       │
│                                                                     │
│  Block 1 │ Conv1d(1   → 16,  k=15, s=2) │ BN │ LeakyReLU  → 2048  │
│  Block 2 │ Conv1d(16  → 32,  k=15, s=2) │ BN │ LeakyReLU  → 1024  │
│  Block 3 │ Conv1d(32  → 64,  k=15, s=2) │ BN │ LeakyReLU  →  512  │
│  Block 4 │ Conv1d(64  → 128, k=15, s=2) │ BN │ LeakyReLU  →  256  │
│  Block 5 │ Conv1d(128 → 256, k=15, s=2) │ BN │ LeakyReLU  →  128  │
└────────────────────────────┬────────────────────────────────────────┘
                             │  skip connections (concatenated)
┌────────────────────────────▼────────────────────────────────────────┐
│  BOTTLENECK                                                         │
│  Conv1d(256 → 512, k=15, s=2) │ BN │ LeakyReLU        →   64      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  DECODER  (ConvTranspose1d blocks — each doubles the time axis)     │
│                                                                     │
│  Block 5 │ ConvT(512 → 256, k=15, s=2) │ BN │ ReLU  + skip → 128  │
│  Block 4 │ ConvT(512 → 128, k=15, s=2) │ BN │ ReLU  + skip → 256  │
│  Block 3 │ ConvT(256 →  64, k=15, s=2) │ BN │ ReLU  + skip → 512  │
│  Block 2 │ ConvT(128 →  32, k=15, s=2) │ BN │ ReLU  + skip → 1024 │
│  Block 1 │ ConvT( 64 →  16, k=15, s=2) │ BN │ ReLU  + skip → 2048 │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  OUTPUT HEAD                                                        │
│  ConvT(32 → 1, k=1) │ Tanh                              →  4096    │
└─────────────────────────────────────────────────────────────────────┘
Output (1, 4096)   ← reconstructed clean strain
```

### Loss Function

Training minimises a combined time-domain + frequency-domain loss:

```
L_total = 0.7 × MSE(pred, clean) + 0.3 × MSE(|FFT(pred)|, |FFT(clean)|)
```

The spectral term forces the model to match the chirp's frequency-domain shape, preventing the over-smoothed outputs that MSE alone produces.

### Training Schedule

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Batch size | 32 |
| Max epochs | 80 |
| Early stopping patience | 10 epochs |
| Gradient clip | max_norm = 1.0 |

---

## Noise Model

Real LIGO noise is not Gaussian. Training injects a physics-motivated mixture of three noise types:

### Mixture Composition

| Component | Probability | Amplitude | Description |
|---|---|---|---|
| Coloured noise | 100 % | 0.05 – 0.20 | Pink (α = 0.5) spectrum, 20–400 Hz band-limited with Tukey taper; mimics LIGO's low-frequency wall |
| Power-line tones | 70 % | 0.02 – 0.10 | 60 Hz fundamental + 120 Hz harmonic; models US mains interference |
| Burst glitches | 10 % | 0.30 – 1.00 | Short Gaussian blobs (width 10–100 samples) centred in the middle half of the window |

### Coloured Noise Generation

The `make_coloured_noise` function generates band-limited noise with a controlled power spectral density:

1. Compute the one-sided frequency axis via `rfftfreq`
2. Shape power as `f^(−α)` with `α = 0.5` (pink noise)
3. Apply a Tukey-tapered band mask (20–400 Hz) — smooth edges prevent Gibbs ringing
4. Normalise the power spectrum to preserve spectral shape
5. Multiply independent real/imaginary Gaussian draws by `√power` in the frequency domain
6. IFFT back to time domain; normalise to unit standard deviation

---

## Dataset Guide

### Training Data — GWTC Events

All segments are fetched live from [GWOSC](https://gwosc.org) via the `preprocessing.py` module. Each event contributes one or more 4-second whitened windows per detector.

*("H1", 1126259462, 1126259472), ("L1", 1126259462, 1126259472),  
    ("H1", 1128678900, 1128678910), ("L1", 1128678900, 1128678910),   
    ("H1", 1135136350, 1135136360), ("L1", 1135136350, 1135136360),   
    ("H1", 1167559936, 1167559946), ("L1", 1167559936, 1167559946),   
    ("H1", 1180922494, 1180922504), ("L1", 1180922494, 1180922504),   
    ("H1", 1185389807, 1185389817), ("L1", 1185389807, 1185389817),   
    ("H1", 1186302519, 1186302529), ("L1", 1186302519, 1186302529),   
    ("H1", 1186741861, 1186741871),
    ("L1", 1186741861, 1186741871),
    ("V1", 1186741861, 1186741871),    
    ("H1", 1187008882, 1187008892),
    ("L1", 1187008882, 1187008892),
    ("V1", 1187008882, 1187008892),  
    ("H1", 1187529256, 1187529266), ("L1", 1187529256, 1187529266),
("H1",1238303730.2,1238303747.2),("L1",1238303730.2,1238303747.2),
("H1",1238782700,1238782710),("L1",1238782700,1238782710),("V1",1238782700,1238782710),
("H1",1239082260,1239082270),("L1",1239082260,1239082270),("V1",1239082260,1239082270),
("H1",1239168610,1239168620),("L1",1239168610,1239168620),
("H1",1239198200,1239198210),("L1",1239198200,1239198210),("V1",1239198200,1239198210),
("H1",1239917950,1239917960),("L1",1239917950,1239917960),
("V1",1240215500,1240215510),("L1",1240215500,1240215510),
("H1", 1240340815, 1240340825),("L1", 1240340815, 1240340825),
("H1",1240944860,1240944870),("L1",1240944860,1240944870),("V1",1240944860,1240944870),
("H1",1241719650,1241719660),("L1",1241719650,1241719660),("V1",1241719650,1241719660),
("H1",1241816080,1241816090),("L1",1241816080,1241816090),("V1",1241816080,1241816090),
("H1", 1241852070, 1241852080),("L1", 1241852070, 1241852080),
("H1",1242107475,1242107485),("L1",1242107475,1242107485),("V1",1242107475,1242107485),
("H1",1242315360,1242315370),("L1",1242315360,1242315370),("V1",1242315360,1242315370),
("H1",1242442960,1242442970),("L1",1242442960,1242442970),("V1",1242442960,1242442970),
("H1",1242459850,1242459860),("L1",1242459850,1242459860),
("H1", 1242984070, 1242984080),("L1", 1242984070, 1242984080),
("H1",1243533580,1243533590),("L1",1243533580,1243533590),("V1",1243533580,1243533590),
("V1",1245035070,1245035080),("L1",1245035070,1245035080),
("V1",1245955940,1245955950),("L1",1245955940,1245955950) ,
("H1", 1246048400, 1246048410),("L1", 1246048400, 1246048410),("V1", 1246048400, 1246048410),
("H1", 1246487210, 1246487220),("L1", 1246487210, 1246487220),("V1", 1246487210, 1246487220),
("H1",1246527220,1246527230),("L1",1246527220,1246527230),
("V1",1246663510,1246663520),("L1",1246663510,1246663520),
("H1",1247608530,1247608540),("L1",1247608530,1247608540),
("H1",1247616530,1247616540),("L1",1247616530,1247616540),("V1",1247616530,1247616540),
("H1",1248112060,1248112070),("L1",1248112060,1248112070),("V1",1248112060,1248112070),
("H1",1248242630,1248242640),("L1",1248242630,1248242640),("V1",1248242630,1248242640),
("H1", 1248331520, 1248331530),("L1", 1248331520, 1248331530),("V1", 1248331520, 1248331530),
("H1",1248617390,1248617400),("L1",1248617390,1248617400)
O3 contributes the majority of training segments (~80 GPS timestamps spanning April 2019 – March 2020). See `TRAIN_EVENTS` in `train.py` for the full list.

### Test / Held-Out Events

The following events are **strictly excluded** from training and used only for evaluation:

| Event | GPS Time | Notes |
|---|---|---|
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


### Preprocessing Steps

Each segment goes through the following pipeline before training or inference:

```
Raw strain (GWOSC)
    │
    ▼ Bandpass filter         20 – 400 Hz, 4th-order Butterworth
    │
    ▼ Whitening               divide by ASD estimated from neighbouring data
    │
    ▼ Window                  4-second Hann-windowed segment (4096 samples)
    │
    ▼ Normalise               subtract mean, divide by std → unit-std float32
    │
    ▼ Unsqueeze               (N, 4096) → (N, 1, 4096) for Conv1d input
```

### Fetching Data

```python
from preprocessing import preprocess, WINDOW_LENGTH, SAMPLE_RATE

# Returns np.ndarray of shape (n_segments, WINDOW_LENGTH)
segments = preprocess("H1", 1126259462, 1126259472)
```

Requirements: internet access to `https://gwosc.org`. Segments that fail to fetch (bad data quality, gaps) are silently skipped with a warning printed to stdout.

---

## Evaluation Methodology

### Metrics

Each held-out event is evaluated on three complementary metrics:

**1. Time-domain MSE**
```
MSE = mean((pred − clean)²)
```
Measures overall amplitude fidelity. Lower is better. Reported per-segment and averaged over the event.

**2. Matched-filter SNR recovery**
```
SNR_recovered / SNR_injected
```
The key GW-physics metric. A well-denoised signal should recover ≥ 90 % of the template-matched SNR. Values > 1.0 indicate the model is amplifying noise (a sign of failure — as seen in the current plot).

**3. Spectral Similarity (frequency-domain MSE)**
```
spec_loss = MSE(|FFT(pred)|, |FFT(clean)|)
```
Catches spectral hallucination: spurious peaks in the reconstructed spectrum that are absent from the noisy input.

### Diagnostic Plots

For each held-out event, the evaluation script produces a 4-panel figure:

| Panel | What to look for |
|---|---|
| **Time domain** | Reconstructed (red) should be quieter than noisy (blue), not larger |
| **Noise residual** | Should be near zero with no periodic structure |
| **Frequency domain** | Reconstructed spectrum should not introduce spikes absent from input |
| **Spectrogram** | Should show a chirp track, not horizontal stripes |

### Known Failure Modes

The plot `GW200306_093714_H1.png` illustrates a **collapsed model**. Symptoms and causes:

| Symptom | Likely cause |
|---|---|
| Reconstructed amplitude > input | Normalisation mismatch between train and inference |
| Positive-only residual | Model adding energy, not removing it |
| 250 Hz harmonic spikes | Model overtrained on fixed 60/120 Hz tones; generalisation failure |
| Horizontal spectrogram stripes | Model outputs fixed spectral template regardless of input |

**Remediation checklist:**

1. Confirm inference preprocessing matches training (whitening scale, normalisation range)
2. Randomise power-line tone frequency during training (`f₀ ~ Uniform(40, 80)` Hz)
3. Check training loss curves — plateau before epoch 10 indicates collapse
4. Validate on synthetic data with known ground truth before running on real events

### Running Evaluation

```bash
# Single event
python test.py --detector H1 --gps-start 1249852257 --gps-end 1249852267 \
    --checkpoint denoiser.pth --output plots/

# All held-out events
python test.py --all-test-events --checkpoint denoiser.pth --output plots/
```

---


                                                            " RESULTS "

-- GW200322_091133/H1  [1268903510-1268903520] --
  SNR Improvement   :   14.94 dB      PASS
  Norm. MSE         :  0.0320         PASS
  Noise Reduction   :   96.80 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200322_091133_H1.png
-- GW200322_091133/L1  [1268903510-1268903520] --
  SNR Improvement   :   -7.76 dB      FAIL
  Norm. MSE         :  5.9672         FAIL
  Noise Reduction   : -496.72 %        FAIL
  Spectral MSE      :  0.0001         PASS
  Overall           : FAIL

  Plot saved -> plot_GW200322_091133_L1.png
-- GW200322_091133/V1  [1268903510-1268903520] --
  SNR Improvement   :    2.22 dB      FAIL
  Norm. MSE         :  0.6001         FAIL
  Noise Reduction   :   39.99 %        FAIL
  Spectral MSE      :  0.0001         PASS
  Overall           : FAIL

  Plot saved -> plot_GW200322_091133_V1.png
-- GW200316_215756/H1  [1268431090-1268431100] --
  SNR Improvement   :   19.58 dB      PASS
  Norm. MSE         :  0.0110         PASS
  Noise Reduction   :   98.90 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200316_215756_H1.png
-- GW200316_215756/L1  [1268431090-1268431100] --
  SNR Improvement   :   23.57 dB      PASS
  Norm. MSE         :  0.0044         PASS
  Noise Reduction   :   99.56 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200316_215756_L1.png
-- GW200316_215756/V1  [1268431090-1268431100] --
  SNR Improvement   :   19.86 dB      PASS
  Norm. MSE         :  0.0103         PASS
  Noise Reduction   :   98.97 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200316_215756_V1.png
-- GW200311_115853/L1  [1267963150-1267963160] --
  SNR Improvement   :   13.95 dB      PASS
  Norm. MSE         :  0.0403         PASS
  Noise Reduction   :   95.97 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200311_115853_L1.png
-- GW200311_115853/V1  [1267963150-1267963160] --
  SNR Improvement   :   19.31 dB      PASS
  Norm. MSE         :  0.0117         PASS
  Noise Reduction   :   98.83 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200311_115853_V1.png
-- GW200311_115853/H1  [1267963150-1267963160] --
  SNR Improvement   :   14.29 dB      PASS
  Norm. MSE         :  0.0372         PASS
  Noise Reduction   :   96.28 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200311_115853_H1.png
-- GW200308_173609/H1  [1267724180-1267724190] --
  SNR Improvement   :   18.31 dB      PASS
  Norm. MSE         :  0.0147         PASS
  Noise Reduction   :   98.53 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200308_173609_H1.png
-- GW200308_173609/L1  [1267724180-1267724190] --
  SNR Improvement   :   18.89 dB      PASS
  Norm. MSE         :  0.0129         PASS
  Noise Reduction   :   98.71 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200308_173609_L1.png
-- GW200308_173609/V1  [1267724180-1267724190] --
  SNR Improvement   :    3.03 dB      PASS
  Norm. MSE         :  0.4973         FAIL
  Noise Reduction   :   50.27 %        PASS
  Spectral MSE      :  0.0001         PASS
  Overall           : FAIL

  Plot saved -> plot_GW200308_173609_V1.png
-- GW200306_093714/H1  [1267522650-1267522660] --
  SNR Improvement   :   21.02 dB      PASS
  Norm. MSE         :  0.0079         PASS
  Noise Reduction   :   99.21 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200306_093714_H1.png
-- GW200306_093714/L1  [1267522650-1267522660] --
  SNR Improvement   :   21.02 dB      PASS
  Norm. MSE         :  0.0079         PASS
  Noise Reduction   :   99.21 %        PASS
  Spectral MSE      :  0.0000         PASS
  Overall           : PASS

  Plot saved -> plot_GW200306_093714_L1.png

==================================================================================
Event                      SNR     MSE     NR%  Pearson  SpecMSE   Pass
----------------------------------------------------------------------------------
GW200322_091133/H1       14.94  0.0320   96.80   0.9862   0.0000 PASS
GW200322_091133/L1       -7.76  5.9672 -496.72  -0.6983   0.0001 FAIL
GW200322_091133/V1        2.22  0.6001   39.99   0.9182   0.0001 FAIL
GW200316_215756/H1       19.58  0.0110   98.90   0.9963   0.0000 PASS
GW200316_215756/L1       23.57  0.0044   99.56   0.9978   0.0000 PASS
GW200316_215756/V1       19.86  0.0103   98.97   0.9967   0.0000 PASS
GW200311_115853/L1       13.95  0.0403   95.97   0.9900   0.0000 PASS
GW200311_115853/V1       19.31  0.0117   98.83   0.9959   0.0000 PASS
GW200311_115853/H1       14.29  0.0372   96.28   0.9906   0.0000 PASS
GW200308_173609/H1       18.31  0.0147   98.53   0.9934   0.0000 PASS
GW200308_173609/L1       18.89  0.0129   98.71   0.9946   0.0000 PASS
GW200308_173609/V1        3.03  0.4973   50.27   0.9183   0.0001 FAIL
GW200306_093714/H1       21.02  0.0079   99.21   0.9979   0.0000 PASS
GW200306_093714/L1       21.02  0.0079   99.21   0.9979   0.0000 PASS
==================================================================================
Passed: 11 / 14

## Quickstart

```bash
# 1. Install dependencies
pip install torch numpy scipy gwpy matplotlib

# 2. Train (fetches data from GWOSC automatically)
python train.py

# 3. Evaluate on held-out events
python evaluate.py --all-test-events --checkpoint denoiser.pth

# 4. Denoise a custom segment
python train.py --detector H1 --gps-start <GPS> --gps-end <GPS+10> \
    --checkpoint denoiser.pth
```

---

## File Reference

| File | Purpose |
|---|---|
| `train.py` | Full training pipeline: data fetch, noise injection, train loop, checkpointing |
| `model.py` | WaveUNet `Autoencoder` class |
| `preprocessing.py` | GWOSC fetch, bandpass, whitening, windowing |
|'utils.py'|for plotting |
| `test.py` | evaluating events |
| `denoiser.pth` | Saved checkpoint (best validation loss) |

---

## References

- Abbott et al. (2021) — GWTC-2, [arXiv:2004.08342](https://arxiv.org/abs/2004.08342)
- Stowell & Plumbley (2019) — WaveUNet, [arXiv:1806.03185](https://arxiv.org/abs/1806.03185)
- GWOSC — [gwosc.org](https://gwosc.org)
- GWpy — [gwpy.github.io](https://gwpy.github.io)
