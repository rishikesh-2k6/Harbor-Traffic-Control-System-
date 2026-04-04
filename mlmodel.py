import pandas as pd
import numpy as np
import os
import time
import tempfile
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
_TMP              = tempfile.gettempdir()

# The ONE growing file that simulator.py writes to
COLLECTED_FILE    = os.path.join(SCRIPT_DIR, "collected_data.csv")
# Snapshot of latest-per-ship predictions (read by traffic_algo.py)
PREDS_FILE        = os.path.join(_TMP, "stage3_live_predictions.csv")

MODEL_STATE_FILE  = os.path.join(SCRIPT_DIR, "model_state.pt")
SCALER_FILE       = os.path.join(SCRIPT_DIR, "scaler.pkl")
LABEL_ENC_FILE    = os.path.join(SCRIPT_DIR, "label_encoder.pkl")

# ── Tuning knobs ──────────────────────────────────────────────────────────────
MIN_ROWS_TO_TRAIN = 100     # Don't attempt training until this many detections exist
RETRAIN_EVERY     = 50      # Retrain whenever this many NEW rows have been added
# ─────────────────────────────────────────────────────────────────────────────

_MAX_RETRIES      = 5
_RETRY_DELAY      = 0.02    # seconds

# Module-level state
_cached_model       = None
_cached_scaler      = None
_cached_le          = None
_last_trained_rows  = 0     # how many rows were in the file when we last trained
_model_is_live      = False # flips True once first training succeeds

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VesselMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.type_head = nn.Linear(32, num_classes)
        self.weight_head = nn.Linear(32, 1)

    def forward(self, x):
        features = self.shared(x)
        type_logits = self.type_head(features)
        weight_pred = self.weight_head(features)
        # Squeeze the weight prediction so it matches target shape [batch_size]
        return type_logits, weight_pred.squeeze(-1)

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions  (must match simulator.py column names exactly)
# ─────────────────────────────────────────────────────────────────────────────
ACOUSTIC_FEATURES = [
    'Received_Level_dB', 'Source_Level_dB', 'Transmission_Loss_dB',
    'Frequency_Hz', 'Doppler_Shift_Hz', 'SNR_dB', 'Ambient_Noise_dB',
]
PROPAGATION_FEATURES = [
    'Sound_Speed_mps', 'Time_of_Arrival_s', 'Bearing_deg',
]
ENVIRONMENTAL_FEATURES = [
    'Water_Temp_C', 'Salinity_ppt', 'Sensor_Depth_m',
    'Sea_State', 'Wind_Speed_knots',
]
ENGINEERED_FEATURES = [
    'Acoustic_Energy', 'SNR_Freq_interaction', 'Received_Level_squared',
    'Doppler_Speed_proxy', 'Transmission_Efficiency', 'Env_Composite',
]
ALL_FEATURES = (ACOUSTIC_FEATURES + PROPAGATION_FEATURES +
                ENVIRONMENTAL_FEATURES + ENGINEERED_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_read_csv(path):
    for _ in range(_MAX_RETRIES):
        try:
            return pd.read_csv(path)
        except (PermissionError, OSError):
            time.sleep(_RETRY_DELAY)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _safe_write_csv(df, path):
    tmp = path + ".tmp"
    for _ in range(_MAX_RETRIES):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
            return
        except (PermissionError, OSError):
            time.sleep(_RETRY_DELAY)


def prepare_data(df):
    """
    Add 6 engineered features to any sensor-feature dataframe.
    Works identically on the collected training data and on live inference slices.
    """
    df = df.copy()
    df['Acoustic_Energy']         = 10 ** (df['Received_Level_dB'] / 10)
    df['SNR_Freq_interaction']    = df['SNR_dB'] * np.log1p(df['Frequency_Hz'])
    df['Received_Level_squared']  = df['Received_Level_dB'] ** 2
    df['Doppler_Speed_proxy']     = np.abs(df['Doppler_Shift_Hz']) / (df['Frequency_Hz'] + 1)
    df['Transmission_Efficiency'] = df['Source_Level_dB'] - df['Transmission_Loss_dB']
    df['Env_Composite']           = df['Water_Temp_C'] * df['Salinity_ppt'] / 100.0
    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Core: dynamic model updater — called EVERY animation frame
# ─────────────────────────────────────────────────────────────────────────────

def update_model_from_collected_data():
    """
    1. Count rows in collected_data.csv.
    2. If we have enough new rows, retrain both models.
    3. Run live inference on the latest per-ship slice.
    4. Write stage3_live_predictions.csv for traffic_algo.py.

    Returns (status_string, is_live_bool).
    """
    global _cached_model, _cached_scaler, _cached_le, _last_trained_rows, _model_is_live, _device

    if not os.path.exists(COLLECTED_FILE):
        return "Waiting for data…", False

    df_all = _safe_read_csv(COLLECTED_FILE)
    if df_all.empty:
        return "Waiting for data…", False

    n_rows = len(df_all)

    # ── Should we (re)train? ─────────────────────────────────────────────────
    if (n_rows >= MIN_ROWS_TO_TRAIN and
            n_rows - _last_trained_rows >= RETRAIN_EVERY):
        _retrain(df_all)

    # ── Live inference ───────────────────────────────────────────────────────
    if _model_is_live and _cached_model is not None and _cached_scaler is not None:
        try:
            # Keep only the latest reading per ship
            latest = df_all.sort_values('Timestamp').groupby('Ship_ID').last().reset_index()
            feat   = prepare_data(latest)
            if not feat.empty and all(c in feat.columns for c in ALL_FEATURES):
                X_np = feat[ALL_FEATURES].values
                X_scaled = _cached_scaler.transform(X_np)
                
                # Move to GPU and predict
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(_device)
                
                _cached_model.eval()
                with torch.no_grad():
                    type_logits, weight_preds = _cached_model(X_tensor)
                    _, type_indices = torch.max(type_logits, 1)
                
                # Move back to CPU
                type_indices = type_indices.cpu().numpy()
                weight_preds = weight_preds.cpu().numpy().astype(int)
                
                pred_types = _cached_le.inverse_transform(type_indices)

                out = pd.DataFrame({
                    'Ship_ID':       feat['Ship_ID'].values,
                    'Pred_Type':     pred_types,
                    'Pred_Weight_kg': weight_preds,
                })
                _safe_write_csv(out, PREDS_FILE)
        except Exception as e:
            print(f"[ML Inference Error] {e}")
            pass
        return f"ML LIVE ({_device.type.upper()}) — {n_rows} detections", True

    return f"Collecting ({n_rows}/{MIN_ROWS_TO_TRAIN})", False


def _retrain(df_all):
    """Train PyTorch MLP (type classification + weight regression) on all collected data using GPU if available."""
    global _cached_model, _cached_scaler, _cached_le, _last_trained_rows, _model_is_live, _device

    try:
        feat = prepare_data(df_all)
        if len(feat) < MIN_ROWS_TO_TRAIN:
            return
        if not all(c in feat.columns for c in ALL_FEATURES):
            return

        X_raw       = feat[ALL_FEATURES].values
        y_type_raw  = feat['Actual_Type'].values
        y_weight_raw= feat['Actual_Weight_kg'].values

        # Scale X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        # Encode Y (categorical)
        le = LabelEncoder()
        y_type_encoded = le.fit_transform(y_type_raw)
        
        # Convert to PyTorch tensors and move to device
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(_device)
        y_type_tensor = torch.tensor(y_type_encoded, dtype=torch.long).to(_device)
        y_weight_tensor = torch.tensor(y_weight_raw, dtype=torch.float32).to(_device)
        
        input_dim = X_tensor.shape[1]
        num_classes = len(le.classes_)
        
        # Initialize or re-initialize model
        model = VesselMLP(input_dim, num_classes).to(_device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        # Type Loss: CrossEntropy, Weight Loss: MSE
        criterion_type = nn.CrossEntropyLoss()
        criterion_weight = nn.MSELoss()
        
        model.train()
        epochs = 100
        for epoch in range(epochs):
            optimizer.zero_grad()
            type_logits, weight_pred = model(X_tensor)
            
            loss_type = criterion_type(type_logits, y_type_tensor)
            loss_weight = criterion_weight(weight_pred, y_weight_tensor)
            
            # Combine losses (scale down MSE loss since weights are large, e.g. 50000 kg)
            # A scale factor of 1e-6 or standardizing weights would be ideal, but scaling loss works too
            loss = loss_type + (loss_weight * 1e-8)
            
            loss.backward()
            optimizer.step()

        # Persist models
        torch.save(model.state_dict(), MODEL_STATE_FILE)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(le, LABEL_ENC_FILE)

        _cached_model      = model
        _cached_scaler     = scaler
        _cached_le         = le
        _last_trained_rows = len(df_all)
        _model_is_live     = True

        print(f"[ML Model Retrained] Device: {_device}, Rows: {_last_trained_rows}, Final Loss: {loss.item():.4f}")

    except Exception as e:
        print(f"[ML Training Error] {e}")
        pass