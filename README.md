# ⚓ Harbor Traffic Control System

A real-time 3D simulation of an **Underwater Wireless Sensor Network (UWSN)** for harbor vessel monitoring. The system tracks ships across three zones (Outer Yard → LOC → Inner Zone), uses acoustic physics to model sensor detections, trains a live ML model to classify vessel types, and provides a browser-based analytics dashboard.

---

## 📁 Project Structure

```
demo-9/
├── main.py             # Main simulation entry point (Tkinter + Matplotlib 3D)
├── simulator.py        # Physics engine, vessel generation, sensor collection
├── mlmodel.py          # PyTorch MLP model — auto-trains from collected data
├── traffic_algo.py     # Traffic management system (TMS) — zone rules & violations
├── dashboard.html      # Live browser dashboard (reads dashboard_data.json)
├── dashboard_data.json # Updated every frame by main.py (auto-generated)
├── collected_data.csv  # Cumulative acoustic sensor readings (auto-generated)
├── model_state.pt      # Saved PyTorch model weights (auto-generated)
├── scaler.pkl          # Sklearn StandardScaler (auto-generated)
└── label_encoder.pkl   # Sklearn LabelEncoder (auto-generated)
```

---

## 🛠️ Prerequisites

### Python Version
- Python **3.9+** recommended

### Required Packages

Install all dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn joblib torch
```

> **GPU Acceleration (optional but recommended):**  
> If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).  
> The simulation automatically uses `cuda` if available, otherwise falls back to `cpu`.

---

## 🚀 How to Run

### 1. Start the 3D Simulation

Open a terminal in the `demo-9/` folder and run:

```bash
python main.py
```

This opens a full-screen **Tkinter window** with a live 3D animated harbor scene.

### 2. Open the Dashboard (optional)

While the simulation is running, open `dashboard.html` in any browser:

```
demo-9/dashboard.html
```

Or use the **Live Server** extension in VS Code for auto-refresh.  
The dashboard reads `dashboard_data.json` which is updated by the simulation every animation frame.

---

## 🖥️ Simulation Features

| Feature | Description |
|---|---|
| **3D Harbor Map** | 10 km × 10 km seabed with Outer Yard, LOC wall, and Inner Zone |
| **35 UWSN Sensors** | Randomly placed at surface, mid-water, and seabed depths |
| **22 Vessels** | 6 types: Cargo Ship, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel |
| **Acoustic Physics** | Mackenzie sound speed, Thorp absorption, Doppler shift, SNR, TOA |
| **Live ML Model** | PyTorch MLP auto-trains after 100 sensor detections; retrains every 50 new rows |
| **Traffic Management** | Zone-based speed limits, halt rules, overspeed violation detection |
| **Dashboard JSON** | Vessel states written to `dashboard_data.json` every frame |

---

## 🗺️ Zone Layout

```
X = 0                X = 700         X = 5000              X = 10000
|----HARBOR----------|----INNER ZONE--|------OUTER YARD------|
(Docked)             (Patrol, 6 kn)  (LOC Wall)  (10 kn max)
```

Ships spawn on the **right side (X > 7500)** and travel **INBOUND** toward the harbor.

---

## 🤖 ML Model Details

- **Architecture:** Multi-layer Perceptron (MLP) with shared backbone
- **Task 1:** Classify vessel type (6 classes)
- **Task 2:** Regress vessel weight (kg)
- **Input features:** 23 acoustic + propagation + environmental features
- **Training:** Starts automatically once `collected_data.csv` has ≥ 100 rows; retrains every 50 new rows
- **Device:** Auto-selects CUDA GPU or CPU

---

## 🎮 Controls

| Control | Action |
|---|---|
| **⟲ Reset View** button | Resets the 3D camera to default angle |
| **Mouse drag** | Rotate the 3D scene |
| **Scroll wheel** | Zoom in/out |
| **Close window** | Stop the simulation |

---

## ⚠️ Notes

- `collected_data.csv` grows continuously while the simulation runs. To start fresh, **delete it** before launching.
- Saved model files (`model_state.pt`, `scaler.pkl`, `label_encoder.pkl`) are overwritten on each retrain — this is expected.
- The dashboard auto-refreshes if opened with a live server; otherwise, manually refresh the browser tab.
