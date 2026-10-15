# ══════════════════════════════════════════════════════════════
# 3D U-Net v2 Training — Colab (Python runtime, T4 GPU)
# Same architecture as R code, saves JIT model for R
#
# 1. Runtime → Change runtime type → Python → T4 GPU
# 2. Run all cells
# 3. Download model from Drive when done
# ══════════════════════════════════════════════════════════════

# ── CELL 1: Mount Drive ──────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = '/content/drive/MyDrive/R_project'
TRAINING_DIR = os.path.join(DRIVE_ROOT, 'data', 'training')
MODEL_DIR = os.path.join(DRIVE_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

assert os.path.exists(os.path.join(TRAINING_DIR, 'manifest.csv')), \
    f"manifest.csv not found at {TRAINING_DIR}"
print("Training data found!")
print("Files:", os.listdir(TRAINING_DIR))

# ── CELL 2: Setup ────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import gc, time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ── CELL 3: 3D U-Net (identical to R/03_model_3d_unet.R) ────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)
    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        d = [skip.size(i) - x.size(i) for i in range(2, 5)]
        x = F.pad(x, [d[2]//2, d[2]-d[2]//2,
                       d[1]//2, d[1]-d[1]//2,
                       d[0]//2, d[0]-d[0]//2])
        return self.conv(torch.cat([x, skip], dim=1))

class UNet3D(nn.Module):
    def __init__(self, in_ch=4, n_classes=4, base=32):
        super().__init__()
        self.enc1 = EncoderBlock(in_ch, base)
        self.enc2 = EncoderBlock(base, base*2)
        self.enc3 = EncoderBlock(base*2, base*4)
        self.enc4 = EncoderBlock(base*4, base*8)
        self.bottleneck = ConvBlock3D(base*8, base*16)
        self.dec4 = DecoderBlock(base*16, base*8)
        self.dec3 = DecoderBlock(base*8, base*4)
        self.dec2 = DecoderBlock(base*4, base*2)
        self.dec1 = DecoderBlock(base*2, base)
        self.final = nn.Conv3d(base, n_classes, 1)
    def forward(self, x):
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        f4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        return self.final(
            self.dec1(self.dec2(self.dec3(self.dec4(b, f4), f3), f2), f1))

# ── CELL 4: Dataset + Augmentation ───────────────────────────

def load_r_pt(path):
    """Load .pt file saved by R torch (safetensors format with JSON header)"""
    import json, struct
    with open(path, 'rb') as f:
        raw = f.read()

    header_len = struct.unpack('<Q', raw[:8])[0]
    header_json = raw[8:8+header_len]
    data_bytes = raw[8+header_len:]

    header = json.loads(header_json)
    # Find the first (and usually only) tensor key
    keys = [k for k in header.keys() if k != '__metadata__']
    if not keys:
        keys = list(header.keys())
    meta = header[keys[0]]
    shape = meta['shape']
    dtype_str = meta['dtype']

    dtype_map = {
        'F16': np.float16, 'F32': np.float32, 'F64': np.float64,
        'U8': np.uint8, 'I8': np.int8, 'I16': np.int16,
        'I32': np.int32, 'I64': np.int64, 'BF16': np.float16,
    }
    np_dtype = dtype_map.get(dtype_str, np.float32)

    offsets = meta.get('data_offsets', [0, len(data_bytes)])
    start, end = offsets[0], offsets[1]
    tensor_bytes = data_bytes[start:end]

    data = np.frombuffer(tensor_bytes, dtype=np_dtype)
    return torch.from_numpy(data.reshape(shape).copy())

class TumorDataset(Dataset):
    def __init__(self, df, root, augment=False):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mri = load_r_pt(os.path.join(self.root, row['mri_file'])).float()
        seg = load_r_pt(os.path.join(self.root, row['seg_file'])).long()

        if mri.dim() == 3: mri = mri.unsqueeze(0)
        elif mri.dim() == 4 and mri.shape[-1] <= 4:
            mri = mri.permute(3, 0, 1, 2)

        if self.augment:
            # Random flips
            for ax in [2, 3, 4]:
                if np.random.rand() > 0.5:
                    mri = torch.flip(mri, [ax])
                    seg = torch.flip(seg, [ax - 1])
            # Intensity jitter
            if np.random.rand() > 0.5:
                mri = mri * (0.9 + np.random.rand() * 0.2)
                mri = mri + (np.random.rand() - 0.5) * 0.2
            # Gaussian noise
            if np.random.rand() > 0.7:
                mri = mri + torch.randn_like(mri) * 0.05

        return mri, seg

# ── CELL 5: Loss ─────────────────────────────────────────────

CLASS_WEIGHTS = [0.5, 3.0, 1.5, 2.0]

def weighted_dice_loss(pred, target, smooth=1.0):
    pred_soft = F.softmax(pred, dim=1)
    total = 0.0
    for c in range(pred.shape[1]):
        pc = pred_soft[:, c]
        tc = (target == c).float()
        inter = (pc * tc).sum()
        total += CLASS_WEIGHTS[c] * (1 - (2*inter+smooth) / (pc.sum()+tc.sum()+smooth))
    return total / sum(CLASS_WEIGHTS)

def combined_loss(pred, target):
    w = torch.tensor(CLASS_WEIGHTS, device=pred.device)
    ce = F.cross_entropy(pred, target, weight=w)
    dl = weighted_dice_loss(pred, target)
    return 0.4 * ce + 0.6 * dl

def dice_per_class(pred, target):
    pc = pred.argmax(dim=1)
    d = {}
    for c, name in enumerate(['bg', 'necrosis', 'edema', 'enhancing']):
        p, t = (pc == c), (target == c)
        inter = (p & t).sum().item()
        tot = p.sum().item() + t.sum().item()
        d[name] = 2 * inter / tot if tot > 0 else float('nan')
    return d

# ── CELL 6: Train ────────────────────────────────────────────

manifest = pd.read_csv(f'{TRAINING_DIR}/manifest.csv')
manifest = manifest[manifest['tumor_voxels'] > 0].reset_index(drop=True)
print(f'Samples: {len(manifest)} | Patients: {manifest.patient_id.nunique()}')

np.random.seed(42)
patients = manifest.patient_id.unique()
n_train = int(len(patients) * 0.8)
tp = np.random.choice(patients, n_train, replace=False)
vp = np.setdiff1d(patients, tp)

train_df = manifest[manifest.patient_id.isin(tp)]
val_df = manifest[manifest.patient_id.isin(vp)]
print(f'Train: {len(train_df)} | Val: {len(val_df)}')

BS = 4 if torch.cuda.is_available() else 2
train_dl = DataLoader(TumorDataset(train_df, TRAINING_DIR, True),
                      batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(TumorDataset(val_df, TRAINING_DIR, False),
                    batch_size=BS, num_workers=0, pin_memory=True)

EPOCHS = 150
model = UNet3D(4, 4, 32).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {n_params:,}')

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_dice = 0.0
patience = 20
patience_ctr = 0
history = []

# Resume from checkpoint
ckpt_path = f'{MODEL_DIR}/unet_v2_checkpoint.pt'
hist_path = f'{MODEL_DIR}/unet_v2_history.csv'
if os.path.exists(hist_path):
    history = pd.read_csv(hist_path).to_dict('records')
    best_dice = max(h['val_dice'] for h in history)
    print(f'Resuming from epoch {len(history)+1} | Best dice: {best_dice:.4f}')
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print('Loaded checkpoint')

start_epoch = len(history) + 1
t0 = time.time()

print(f'\n{"="*95}')
print(f'{"Ep":>4} | {"TrLoss":>8} | {"VaLoss":>8} | {"Dice":>7} | {"Necr":>7} | {"Edema":>7} | {"Enh":>7} | {"LR":>10} | {"Time":>6}')
print(f'{"="*95}')

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    losses = []
    for mri, seg in train_dl:
        mri, seg = mri.to(device), seg.to(device)
        optimizer.zero_grad()
        loss = combined_loss(model(mri), seg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()

    model.eval()
    vloss, dices = [], []
    with torch.no_grad():
        for mri, seg in val_dl:
            mri, seg = mri.to(device), seg.to(device)
            pred = model(mri)
            vloss.append(combined_loss(pred, seg).item())
            dices.append(dice_per_class(pred, seg))

    dd = pd.DataFrame(dices)
    dn = dd.necrosis.mean()
    de = dd.edema.mean()
    dh = dd.enhancing.mean()
    dm = np.nanmean([dn, de, dh])
    lr = optimizer.param_groups[0]['lr']
    elapsed = (time.time() - t0) / 60

    mark = ''
    if dm > best_dice:
        best_dice = dm
        torch.save(model.state_dict(), f'{MODEL_DIR}/unet_3d_v2_best.pt')
        patience_ctr = 0
        mark = ' *'
    else:
        patience_ctr += 1

    history.append(dict(epoch=epoch, train_loss=np.mean(losses),
        val_loss=np.mean(vloss), val_dice=dm,
        dice_necrosis=dn, dice_edema=de, dice_enhancing=dh, lr=lr))

    print(f'{epoch:>4} | {np.mean(losses):>8.4f} | {np.mean(vloss):>8.4f} | {dm:>7.4f} | {dn:>7.4f} | {de:>7.4f} | {dh:>7.4f} | {lr:>10.6f} | {elapsed:>5.0f}m{mark}')

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), ckpt_path)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        gc.collect()

    if patience_ctr >= patience:
        print(f'\nEarly stopping at epoch {epoch}')
        break

pd.DataFrame(history).to_csv(hist_path, index=False)

# ── CELL 7: Save JIT model for R ─────────────────────────────

print('\nSaving JIT model for R...')
model.load_state_dict(torch.load(f'{MODEL_DIR}/unet_3d_v2_best.pt',
                                  weights_only=True))
model.eval()
traced = torch.jit.trace(model.cpu(), torch.randn(1, 4, 48, 48, 48))
traced.save(f'{MODEL_DIR}/unet_3d_v2_jit.pt')

results = {
    'Best Val Dice': round(best_dice, 4),
    'Dice - Necrosis': round(float(dd.necrosis.mean()), 4),
    'Dice - Edema': round(float(dd.edema.mean()), 4),
    'Dice - Enhancing': round(float(dd.enhancing.mean()), 4),
    'Epochs Trained': len(history),
    'Parameters': f'{n_params:,}',
}
pd.DataFrame([{'metric': k, 'value': v} for k, v in results.items()])\
    .to_csv(f'{MODEL_DIR}/unet_v2_results.csv', index=False)

print(f'\n{"="*50}')
print(f'  DONE')
print(f'{"="*50}')
for k, v in results.items():
    print(f'  {k}: {v}')
print(f'\nDownload: {MODEL_DIR}/unet_3d_v2_jit.pt')
print(f'  → Replace models/unet_3d_trained_jit.pt')
