import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from preprocessing import preprocess, WINDOW_LENGTH, SAMPLE_RATE
from model import Autoencoder

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 80
LR            = 3e-4
WEIGHT_DECAY  = 1e-4   # FIX D: was 1e-5
VAL_FRACTION  = 0.15   # FIX D: was 0.10
PATIENCE      = 10      # early stopping
SAVE_PATH     = "denoiser.pth"
SPEC_LOSS_W   = 0.3     # weight of spectral loss in combined loss


TRAIN_EVENTS = [
   
    ("H1", 1126259462, 1126259472), ("L1", 1126259462, 1126259472),  
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
]


# ── Realistic noise injection ─────────────────────────────────────────────────
def make_coloured_noise(
    shape: tuple,
    alpha: float = 0.5,
    fs: float = 1024.0,
    f_low: float = 20.0,
    f_high: float = 400.0,
    normalize: bool = True,
    taper_alpha: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
   
    from scipy.signal.windows import tukey as tukey_window

    if rng is None:
        rng = np.random.default_rng()

    n     = shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # eps tied to f_low — prevents DC blow-up without hardcoding
    eps    = f_low / 10.0
    f_safe = np.maximum(freqs, eps)
    power  = f_safe ** (-alpha)

    # Tukey-tapered band mask — smooth edges kill Gibbs ringing in time domain
    # without mixing in white noise (which would corrupt the PSD)
    band_mask    = (freqs >= f_low) & (freqs <= f_high)
    band_indices = np.where(band_mask)[0]
    taper        = np.zeros(len(freqs))
    if len(band_indices) > 0:
        taper[band_indices] = tukey_window(len(band_indices), alpha=taper_alpha)
    power = power * taper

    # Normalise power instead of clip(0, 50) — preserves true spectral shape
    max_power = power.max()
    if max_power > 0:
        power = power / max_power

    # Proper complex noise using caller's rng
    n_freqs   = len(freqs)
    real_part = rng.standard_normal((*shape[:-1], n_freqs))
    imag_part = rng.standard_normal((*shape[:-1], n_freqs))
    spectrum  = (real_part + 1j * imag_part) * np.sqrt(power)

    noise = np.fft.irfft(spectrum, n=n)

    # Normalise over in-band signal only (not full spectrum)
    if normalize:
        std   = np.std(noise, axis=-1, keepdims=True)
        noise = noise / (std + 1e-8)

    return noise.astype(np.float32)


def inject_noise(
    clean: np.ndarray,
    rng:   np.random.Generator,
) -> np.ndarray:
   
    N, L  = clean.shape
    noise = np.zeros_like(clean)
    t     = np.linspace(0, L / SAMPLE_RATE, L, endpoint=False)

    # ① Coloured noise (always present) — rng passed through for reproducibility
    amp_c  = rng.uniform(0.05, 0.20, size=(N, 1))
    noise += amp_c * make_coloured_noise((N, L), rng=rng)

    # ② Narrowband tones — randomised fundamental (FIX C)
    mask_pl = rng.random(N) < 0.70
    if mask_pl.any():
        n_pl   = mask_pl.sum()
        amp_pl = rng.uniform(0.02, 0.10, size=(n_pl, 1))
        phase  = rng.uniform(0, 2 * np.pi, size=(n_pl, 1))
        # Random fundamental per sample instead of fixed 60 Hz
        f0     = rng.uniform(40, 80, size=(n_pl, 1))
        tones  = (np.sin(2 * np.pi * f0       * t + phase)
                + 0.5 * np.sin(2 * np.pi * 2 * f0 * t + phase))
        noise[mask_pl] += amp_pl * tones

    # ③ Burst glitches (short duration)
    mask_gl = rng.random(N) < 0.10
    if mask_gl.any():
        n_gl   = mask_gl.sum()
        centre = rng.integers(L // 4, 3 * L // 4, size=n_gl)
        width  = rng.integers(10, 100, size=n_gl)
        amp_gl = rng.uniform(0.3, 1.0, size=n_gl)
        for i, (c, w, a) in enumerate(zip(centre, width, amp_gl)):
            idx    = np.arange(L)
            glitch = a * np.exp(-0.5 * ((idx - c) / w) ** 2)
            noise[mask_gl][i] += glitch

    return (clean + noise).astype(np.float32)


# ── Loss function ─────────────────────────────────────────────────────────────

def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE on the FFT magnitude spectrum — penalises spectral shape errors."""
    pred_mag   = torch.abs(torch.fft.rfft(pred.squeeze(1),   norm="ortho"))
    target_mag = torch.abs(torch.fft.rfft(target.squeeze(1), norm="ortho"))
    return F.mse_loss(pred_mag, target_mag)


def combined_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    w_spec: float = SPEC_LOSS_W,
) -> torch.Tensor:
    mse  = F.mse_loss(pred, target)
    spec = spectral_loss(pred, target)
    return (1 - w_spec) * mse + w_spec * spec


# ── Dynamic dataset (FIX B) ───────────────────────────────────────────────────

class NoisyGWDataset(Dataset):
   

    def __init__(self, clean: np.ndarray, epoch_seed: int = 0):
        self.clean      = clean.astype(np.float32)   # (N, L)
        self.epoch_seed = epoch_seed

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int):
        # Unique seed per (epoch, sample) — deterministic but different each epoch
        rng   = np.random.default_rng(self.epoch_seed + idx)
        noisy = inject_noise(self.clean[idx : idx + 1], rng)[0]  # (L,)
        clean = self.clean[idx]                                    # (L,)
        return (
            torch.from_numpy(noisy).unsqueeze(0),   # (1, L)
            torch.from_numpy(clean).unsqueeze(0),   # (1, L)
        )


# ── Data collection ───────────────────────────────────────────────────────────

print(f"Fetching {len(TRAIN_EVENTS)} event segments …\n")

all_clean  = []
event_ids  = []   # tracks which event each segment came from

for event_idx, (det, start, end) in enumerate(TRAIN_EVENTS):
    try:
        segs = preprocess(det, start, end)
        all_clean.append(segs)
        event_ids.extend([event_idx] * len(segs))
        print(f"  {det}  [{start}–{end}]  →  {len(segs)} segment(s)")
    except Exception as exc:
        print(f"    {det}  [{start}–{end}]  skipped: {exc}")

if not all_clean:
    raise RuntimeError("No training data fetched. Check network / GWOSC availability.")

clean_np  = np.vstack(all_clean)          # (N, 4096)
event_ids = np.array(event_ids)           # (N,)
print(f"\nTotal clean segments: {len(clean_np)}")

unique_events = np.unique(event_ids)
n_val_events  = max(1, int(len(unique_events) * VAL_FRACTION))

rng_split    = np.random.default_rng(SEED)
val_events   = rng_split.choice(unique_events, size=n_val_events, replace=False)
val_mask     = np.isin(event_ids, val_events)
train_mask   = ~val_mask

clean_train  = clean_np[train_mask]
clean_val    = clean_np[val_mask]

n_train = len(clean_train)
n_val   = len(clean_val)
print(f"Events  — train: {len(unique_events) - n_val_events}  |  val: {n_val_events}")
print(f"Segments— train: {n_train}  |  val: {n_val}  |  Batch size: {BATCH_SIZE}\n")

train_ds = NoisyGWDataset(clean_train, epoch_seed=SEED * 10000)
val_ds   = NoisyGWDataset(clean_val,   epoch_seed=SEED * 20000)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

# ── Model / optimiser / scheduler ────────────────────────────────────────────

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = Autoencoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6,
)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")
print(f"Device: {device}\n")

# ── Training loop ─────────────────────────────────────────────────────────────

best_val  = float("inf")
patience  = 0
history   = {"train": [], "val": []}

for epoch in range(1, NUM_EPOCHS + 1):

    # ── refresh noise seeds (FIX B) ───────────────────────────────────────────
    # Advance epoch_seed so each epoch sees different noise realisations.
    # val seed is also advanced so val noise isn't static either.
    train_ds.epoch_seed = SEED * 10000 + epoch * 100000
    val_ds.epoch_seed   = SEED * 20000 + epoch * 100000

    # ── train ─────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for noisy_b, clean_b in train_loader:
        noisy_b = noisy_b.to(device)
        clean_b = clean_b.to(device)

        optimizer.zero_grad()
        pred = model(noisy_b)
        loss = combined_loss(pred, clean_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * noisy_b.size(0)

    train_loss /= n_train
    scheduler.step(epoch)

    # ── validate ──────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy_b, clean_b in val_loader:
            noisy_b = noisy_b.to(device)
            clean_b = clean_b.to(device)
            val_loss += combined_loss(model(noisy_b), clean_b).item() * noisy_b.size(0)
    val_loss /= n_val

    history["train"].append(train_loss)
    history["val"].append(val_loss)

    lr_now = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
        f"Train: {train_loss:.6f}  Val: {val_loss:.6f}  "
        f"LR: {lr_now:.2e}"
    )

    # ── checkpoint & early stopping ───────────────────────────────────────────
    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss":    best_val,
            "history":     history,
        }, SAVE_PATH)
        print(f"  💾  Checkpoint saved  (val={best_val:.6f})")
    else:
        patience += 1
        if patience >= PATIENCE:
            print(f"\n⏹  Early stopping at epoch {epoch}.")
            break

print(f"\n  Training complete.  Best val loss: {best_val:.6f}")
print(f"    Model saved → '{SAVE_PATH}'")
