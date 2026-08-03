# ⚓ Harbor Traffic Control System
### Adaptive Multi-Robot Networking in Underwater Wireless Sensor Networks (UWSN)

A collaborative underwater robotic and sensor network simulating harbor monitoring, real-time vessel classification, acoustic signal telemetry, dynamic traffic management, and peer-to-peer acoustic communications across a **10 km × 10 km harbor domain**.

---

## 🌟 Overview & Key Capabilities

The system transforms traditional passive harbor monitoring into an active, collaborative network of **stationary hydrophones** and **mobile underwater robot nodes (AUVs)**. As surface vessels traverse the harbor, hydrophone sensors and patrol AUVs capture acoustic physics telemetry, compute Doppler shifts, feed real-time supervised machine learning models to classify vessels, enforce speed limits across harbor zones, and route alerts to Harbor Control.

---

## 🛠️ Environment Verification & Setup

Before running the simulation, teammates can execute the automated environment verification script to check all Python dependencies, verify modular network models, and test JSON schema compatibility.

### 1. Python Requirements
- **Python 3.9+** recommended.
- Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`, `torch`.

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn joblib torch
```

### 2. Run System & Dependency Check
To verify that all dependencies and core modules are functional, run:

```bash
python verify_environment.py
```

Output highlights:
- `[OK]` Python version & library availability (`numpy`, `pandas`, `matplotlib`, `sklearn`, `joblib`, `torch`)
- `[OK]` Modular network architecture initialization (`network_manager.py`)
- `[OK]` JSON schema serialization & compatibility
- `[OK]` Core simulator & traffic algorithm modules

---

## 📁 Repository Structure

```
Harbor-Traffic-Control-System/
├── main.py                 # Main 3D simulation window (Tkinter + Matplotlib 3D engine)
├── simulator.py            # Underwater acoustic physics engine & sensor telemetry generator
├── network.py              # Depth-Based Routing (DBR) & acoustic multi-hop communication engine
├── network_manager.py      # Swarm topology class definitions (RobotNode, CommLink, NetworkManager)
├── traffic_algo.py         # Harbor Traffic System (TMS) — zone speed limits & docking logic
├── mlmodel.py              # PyTorch Deep Learning model — vessel classification & weight regression
├── dashboard.html          # Real-time harbor vessel and traffic management dashboard
├── networking_dashboard.html # Real-time acoustic network topology & DBR routing dashboard
├── verify_environment.py   # Team dependency & system verification suite
├── dashboard_data.json     # Live telemetry snapshot exported every simulation frame
├── dashboard_data.js       # Automatically compiled JavaScript bindings for live dashboards
├── collected_data.csv      # Accumulated acoustic sensor dataset (auto-generated)
├── model_state.pt          # PyTorch model checkpoint weights (auto-generated)
├── scaler.pkl              # Sklearn feature scaler (auto-generated)
└── label_encoder.pkl       # Sklearn label encoder (auto-generated)
```

---

## 🗺️ Harbor Zone Architecture

```
X = 0 m              X = 700 m        X = 5000 m            X = 10000 m
|---- HARBOR --------|--- INNER ZONE---|------ OUTER YARD -----|
 (Docking Slots)      (Patrol, 6 kn)   (LOC Boundary Wall)   (10 kn max)
```

- **Outer Yard Zone (X: 5000m – 10000m)**: Entry corridor for inbound surface traffic. Max speed limit: **10.0 knots**.
- **Line of Control (LOC) Boundary (X = 5000m)**: Sensor curtain dividing outer and inner zones. Heavy cargo/tanker vessels receive traffic control commands at this boundary.
- **Inner Patrol Zone (X: 700m – 5000m)**: Patrol domain where vessels undergo classification verification and dock slot routing. Max speed limit: **6.0 knots**.
- **Harbor Docking Area (X < 700m)**: Color-coded designated dock zones (Dock A - Cargo, Dock T - Tanker, Dock C - Speedboat, Dock V - Fishing, Dock F - Ferry, Dock B - Cruiser).

---

## 🛰️ Modular Network & Routing Architecture (`network.py` & `network_manager.py`)

The network layer provides an extensible framework for underwater multi-robot cooperation and multi-hop telemetry routing:

### 1. Swarm Entities (`network_manager.py`)
- **`RobotNode`**: Represents an Autonomous Underwater Vehicle (AUV) patrolling assigned harbor zones (Outer Yard, Channel, Inner Harbor) with waypoint navigation and battery management.
- **`CommLink`**: Models physical communication channels (Sensor ↔ Robot, Robot ↔ Robot, Robot ↔ Gateway Buoy) with spherical spreading loss, Thorp absorption, and Signal-to-Noise Ratio (SNR in dB).
- **`NetworkManager`**: Manages swarm deployment, calculates active link statuses, and compiles serialized telemetry payloads.

### 2. Multi-Hop Acoustic Routing & Optimization (`network.py`)
- **Depth-Based Routing (DBR)**: Dynamically routes acoustic telemetry from sensors detecting vessels, through intermediate patrolling AUVs, to three surface gateway buoys (`buoy_alpha`, `buoy_beta`, `buoy_gamma`).
- **Duty Cycling Sleep/Wake States**: Conserves node battery by cycling sensor states (70% active, 30% sleeping) through frame-based schedules.
- **Acoustic Bandwidth Compression**: Features dual-mode telemetry (Default vs Compressed frames) to reduce path latency and optimize energy conservation metrics.
- **Dynamic Topology Graph**: Constructed frame-by-frame via `networkx` to evaluate real-time path connectivity and acoustic link quality.

---

## 🚀 How to Run the System

### 1. Launch 3D Simulation
Execute the main application:
```bash
python main.py
```
This opens an interactive 3D visualization displaying the 10 km × 10 km harbor seabed, 35 UWSN sensor hydrophones, surface vessels, and LOC boundary wall. Patrolling AUVs will display with dynamic range circles and relay links.

### 2. Launch Real-Time Dashboards
The system exports telemetry snapshots every **500 ms** to `dashboard_data.json` and `dashboard_data.js`. Open either of the dashboards in any modern browser:

* **Harbor Traffic Dashboard (`dashboard.html`)**: Focuses on vessel speed violations, classification predictions, and dock slot allocations.
* **Acoustic Network Dashboard (`networking_dashboard.html`)**: Visualizes the active network topology graph, node battery levels, active routing paths, SNR quality, and bandwidth compression efficiency.

---

## 🤖 Machine Learning Model

- **Backbone**: Multi-Task PyTorch Neural Network (MLP).
- **Task 1**: Vessel Class Multi-Class Classification (Cargo Ship, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel).
- **Task 2**: Vessel Weight Regression (kg).
- **Features**: 23 features including Received Level (dB), Source Level (dB), Transmission Loss (dB), Frequency (Hz), Doppler Shift (Hz), SNR (dB), Ambient Noise (dB), Sound Speed (m/s), Time of Arrival (s), Bearing (deg), Water Temp (°C), Salinity (ppt), and Depth (m).
- **Auto-Training**: Automatically initializes training when `collected_data.csv` accumulates ≥ 100 hydrophone detection events and retrains periodically every 50 new samples.

---

## 🎮 User Controls

| Control | Function |
|---|---|
| **⟲ Reset View** | Resets 3D camera elevation and azimuth angles |
| **Mouse Left Drag** | Rotates the 3D scene in 360 degrees |
| **Scroll Wheel** | Zooms in/out of the harbor map |
| **Close Window** | Safely terminates simulation loop |

---

## 📊 Telemetry Data Schema (`dashboard_data.json`)

The simulation outputs a structured JSON payload each frame:
```json
{
  "frame": 120,
  "active_sensors": 16,
  "total_sensors": 35,
  "ml_status": "ML LIVE (CPU) — 1250 detections",
  "violations": [
    { "id": 6, "type": "Cruiser", "zone": "outer_yard", "speed": 14.7, "limit": 10.0 }
  ],
  "vessels": [
    { "id": 1, "type": "Cargo Ship", "pred_type": "Cargo Ship", "pred_weight": 142000, "state": "CROSSING", "cmd": "CONTINUE", "x": 7808.4, "y": 5430.8 }
  ],
  "robots": [
    { "id": 101, "name": "AUV-Alpha", "x": 6500.0, "y": 3000.0, "z": -15.0, "battery": 98.5, "status": "PATROLLING" }
  ],
  "network_links": [
    { "id": "link_R101_S12", "source": "Robot-101", "target": "Sensor-12", "snr_db": 14.8, "status": "ACTIVE", "throughput_kbps": 24.0 }
  ],
  "network_stats": {
    "total_robots": 6,
    "active_links_count": 8,
    "avg_throughput_kbps": 38.5,
    "packet_delivery_rate": 0.95
  }
}
```
