"""
Diagnose the displacement (weight) regression head in VesselMLP.

Finding: the shared trunk is trained with
    loss = CrossEntropy(type) + MSELoss(weight_kg) * 1e-8
Weight is on a kg scale (1e3 .. 4e5), so raw MSE is astronomically large and the
1e-8 factor does not normalise it. The regression head therefore receives
meaningless gradient signal and its output is unstable.

This script measures the problem using the SAME scaler/loader path used at
inference time.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlmodel as ml

N = 4096

df = pd.read_csv(ml.COLLECTED_FILE, low_memory=False)
feat = ml.prepare_data(df).tail(N).reset_index(drop=True)

# Use the exact inference path: scaler.transform then tensor
X_scaled = ml._cached_scaler.transform(
    feat[ml.ALL_FEATURES].apply(pd.to_numeric, errors='coerce').values
)
X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_type = ml._cached_le.transform(feat['Actual_Type'].values)
y_t = torch.tensor(feat['Actual_Weight_kg'].values.astype(np.float32))
y_type_t = torch.tensor(y_type)

print('=== Ground-truth displacement scale (kg) ===')
print('min=%9.0f  max=%9.0f  mean=%9.0f  std=%9.0f' % (
    y_t.min(), y_t.max(), y_t.mean(), y_t.std()))

ml._cached_model.eval()
with torch.no_grad():
    logits, w_pred = ml._cached_model(X_t)

raw_mse = nn.MSELoss()(w_pred, y_t).item()
ce = nn.CrossEntropyLoss()(logits, y_type_t).item()

print()
print('=== Loss terms exactly as coded in _retrain() ===')
print('CrossEntropy(type)        = %.6e' % ce)
print('MSELoss(weight_kg)        = %.6e' % raw_mse)
print('weight term used          = %.6e   (x 1e-8)' % (raw_mse * 1e-8))
print('relative weight share     = %.3e' % ((raw_mse * 1e-8) / ce))

print()
print('=== What the weight head actually outputs ===')
p = w_pred.cpu().numpy()
a = y_t.cpu().numpy()
print('pred  min=%.1f  max=%.3e  mean=%.3e' % (p.min(), p.max(), p.mean()))
print('true  min=%.1f  max=%.1f      mean=%.1f' % (a.min(), a.max(), a.mean()))
print('MAE  = %.4g kg' % np.mean(np.abs(p - a)))
safe = np.abs(a) > 1000
print('MAPE (targets>1000kg) = %.4g %%   [n=%d]' % (
    np.mean(np.abs(p[safe] - a[safe]) / np.abs(a[safe])) * 100, safe.sum()))

# ── Gradient contribution to the SHARED trunk ──────────────────────────────
print()
print('=== Gradient reaching the shared trunk ===')
m = ml.VesselMLP(len(ml.ALL_FEATURES), len(ml._cached_le.classes_))
m.load_state_dict(torch.load(ml.MODEL_STATE_FILE, map_location='cpu'))

small = slice(0, 512)
Xs = X_t[small].clone().requires_grad_(False)
logits2, wpred2 = m(Xs)

shared_params = [p for p in m.shared.parameters()]
type_head = [p for p in m.type_head.parameters()]

g_type = torch.autograd.grad(nn.CrossEntropyLoss()(logits2, y_type_t[small]),
                             shared_params + type_head, retain_graph=True,
                             allow_unused=True)
g_wt = torch.autograd.grad(nn.MSELoss()(wpred2, y_t[small]),
                           shared_params + type_head, retain_graph=True,
                           allow_unused=True)

norm = lambda gs: float(sum((g ** 2).sum().item() for g in gs if g is not None) ** 0.5)

nt, nw = norm(g_type), norm(g_wt)
print('|grad| from classification head          = %.6e' % nt)
print('|grad| from weight head (UNSCALED)       = %.6e' % nw)
print('|grad| from weight head (as coded, x1e-8)= %.6e' % (nw * 1e-8))
print()
print('>>> After the 1e-8 factor the weight head supplies')
print('    %.4f%% of the classification head\'s gradient signal.' % (nw * 1e-8 / max(nt, 1e-30) * 100))

# ── What happens with a sane normalised target ─────────────────────────────
print()
print('=== Control: same head, weight target standardised ===')
mu, sd = float(y_t.mean()), float(y_t.std())
y_norm = (y_t - mu) / sd
g_wt_n = torch.autograd.grad(nn.MSELoss()(wpred2, y_norm[small]),
                            shared_params + type_head, retain_graph=False,
                            allow_unused=True)
print('|grad| from weight head (standardised target) = %.6e' % norm(g_wt_n))
print('balance ratio vs classification head          = %.3f' % (norm(g_wt_n) / max(nt, 1e-30)))
print('-> the same 1e-8 factor applied here yields a ratio of %.4f' % (norm(g_wt_n) * 1e-8 / max(nt, 1e-30)))