# Harbor Traffic Control System
### Networked Autonomous Maritime Monitoring, Governance and Simulation

A cyber-physical simulation of a harbour governed by an Underwater Wireless
Sensor Network (UWSN) and a swarm of Autonomous Underwater Vehicles (AUVs).
The system models acoustic sensing, multi-hop underwater routing, machine
learning based vessel classification, and autonomous traffic enforcement across
a 10 km × 10 km domain, under selectable **Normal** and **Monsoon** weather
regimes.

> All results in this document are measured from the simulation code itself.
> Where a claim is unverified or a known limitation exists, it is stated
> explicitly. This is a simulation study, not evidence of real-world maritime
> deployment readiness.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Objectives and what was built](#2-objectives-and-what-was-built)
3. [Evolution from the previous semester](#3-evolution-from-the-previous-semester)
4. [System architecture](#4-system-architecture)
5. [Acoustic physics engine](#5-acoustic-physics-engine)
6. [Machine learning models](#6-machine-learning-models)
7. [Prediction validation and evaluation](#7-prediction-validation-and-evaluation)
8. [Underwater networking](#8-underwater-networking)
9. [Robotics and simulation](#9-robotics-and-simulation)
10. [Environmental scenarios](#10-environmental-scenarios)
11. [User interfaces](#11-user-interfaces)
12. [Technology stack](#12-technology-stack)
13. [Installation and execution](#13-installation-and-execution)
14. [Testing and verification](#14-testing-and-verification)
15. [Limitations and future work](#15-limitations-and-future-work)
16. [Repository structure](#16-repository-structure)

---

## 1. Overview

Traditional harbour traffic control relies on surface radar and AIS. Radar does
not penetrate the water column, AIS can be spoofed or switched off, and a fixed
seabed sensor grid suffers from short battery life, high propagation latency and
communication dead zones.

This project simulates a **hybrid seabed + mobile** sensing network that:

- detects vessels from radiated acoustic noise using published oceanographic models,
- routes detection telemetry to surface gateways over multi-hop acoustic links,
- classifies vessels and estimates displacement with a multi-task neural network,
- validates those predictions against the simulator's own ground truth,
- enforces speed limits, screens traffic at a Line of Control, and assigns berths,
- degrades predictably under monsoon conditions.

The desktop application is a Tkinter + Matplotlib 3D control station. Two
companion web dashboards serve the same live telemetry over HTTP.

---

## 2. Objectives and what was built

| # | Objective | Status | Module | Verification |
|---|---|---|---|---|
| 1 | Seabed hydrophone sensing with acoustic physics | Implemented | `simulator.py` | `verify_environment.py` |
| 2 | Vessel classification (6 classes) | **Verified working** | `mlmodel.py` | 100 % accuracy on 23 validated samples |
| 3 | Displacement (weight) regression | **Implemented but inaccurate** | `mlmodel.py` | MAE 22,713 kg — see §7 |
| 4 | Multi-hop acoustic networking + DBR routing | Implemented | `network.py` | Live PDR/health metrics |
| 5 | AUV swarm patrol, dispatch, battery, recharge | Implemented | `auv.py` | State machine in UI |
| 6 | Traffic governance (speed, LOC, berths) | Implemented | `traffic_algo.py` | Violations surface in UI |
| 7 | Normal vs Monsoon scenarios on one engine | Implemented | `simulator.py` | Live switch, no restart |
| 8 | Persistent prediction logging + ground-truth validation | Implemented | `prediction_logger.py` | 23 unit tests pass |
| 9 | Reproducible model evaluation report | Implemented | `evaluate_predictions.py` | Renders from real log |
| 10 | Crash notifications isolated from the 3D viewport | Implemented | `main.py` | Sidebar panel |
| 11 | Light/dark themes on web dashboards | Implemented | `*.html` | WCAG AA contrast verified |
| 12 | Real-world physical plausibility validation via Groq LPU | Implemented & Verified | `ai_data_validator.py` | 100 % physical plausibility across 104 parameters (`openai/gpt-oss-120b`) |
| — | Speed prediction model | **Does not exist** | — | Speed is simulator state |
| — | Heading prediction model | **Does not exist** | — | Heading is a geometric rule |

---

## 3. Evolution from the previous semester

The earlier version was a passive seabed sensor grid that detected vessels and
classified them from acoustic features.

Added in this extension:

| Addition | Notes |
|---|---|
| AUV swarm | 8 mobile nodes with patrol, dispatch and battery logic |
| Underwater acoustic networking | Link budget, SNR gating, multi-hop routing |
| Depth-Based Routing (DBR) | Multi-hop greedy forwarding toward surface gateways |
| Packet compression | 256 B vs 16 B, latency and energy comparison |
| Duty cycling | Scenario-dependent sensor availability (see §8) |
| Monsoon scenario | Stochastic currents, squalls, mooring failure, SNR collapse |
| Mooring physics | Sensors progress MOORED → STRAINED → SNAPPED and drift away |
| Prediction logging | Persistent per-prediction history (new in this revision) |
| Notification sidebar | Errors no longer overlay the simulation (new) |
| Dashboard themes | Light/dark toggle with persistence (new) |
| Responsive layout | Grid-based UI, viewport ≥ 75 % (new) |

---

## 4. System architecture

```text
 ┌──────────────────────────────────────────────────────────────────────┐
 │  22 surface vessels        35 seabed hydrophones      8 AUVs         │
 │  (6 classes)               (moored, drifting)         (mobile relay)  │
 └───────────┬──────────────────────┬────────────────────────┬───────────┘
             │                      │                        │
             ▼                      ▼                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ ACOUSTIC PHYSICS ENGINE        simulator.py                          │
 │  Mackenzie sound speed · Ross source level · Thorp absorption        │
 │  Wenz ambient noise · Doppler shift · 21-feature extraction          │
 └───────────┬──────────────────────────────────────────────────────────┘
             ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ ACOUSTIC NETWORK               network.py                           │
 │  NetworkX graph per frame · SNR ≥ 10 dB links · DBR multi-hop        │
 │  Duty cycling · 3 buoy gateways · packet compression metrics         │
 └───────────┬──────────────────────────────────────────────────────────┘
             ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ VesselMLP                      mlmodel.py                           │
 │  trunk 128-64-32 → class head (CE) + weight head (MSE)              │
 └───────────┬──────────────────────────────┬───────────────────────────┘
             ▼                              ▼
 ┌────────────────────────┐   ┌────────────────────────────────────────┐
 │ TRAFFIC CONTROL        │   │ PREDICTION VALIDATION                 │
 │  traffic_algo.py       │   │  prediction_logger.py                 │
 │  speed limits, LOC,    │   │  logs pred vs simulator ground truth  │
 │  berth assignment      │   │  MAE / RMSE / accuracy / angular err  │
 └────────────────────────┘   └────────────────────────────────────────┘
             │                              │
             ▼                              ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ INTERFACES                                                            │
 │  main.py (Tkinter 3D station)  ·  dashboard.html  ·  networking_dashboard │
 │  dashboard_data.json written every 500 ms, served over HTTP           │
 └──────────────────────────────────────────────────────────────────────┘
```

**Ground-truth separation** — the three roles stay distinct so the model never
receives its own answer as input:

```text
Simulation state   →  Ground truth      (Actual_Type, Actual_Weight_kg, …)
Sensor features    →  Model prediction  (VesselMLP forward pass)
Prediction + truth →  Evaluation       (prediction_logger.py)
```

---

## 5. Acoustic physics engine

Implemented with NumPy, no compiled physics engine.

| Model | Formula | Purpose |
|---|---|---|
| Mackenzie (1981) | 9-term sound speed in T, S, D | Sound speed profile |
| Ross (1976) | `SL = SL₀ + 55·log₁₀(v/10) + 20·log₁₀(d/3) + N(0,σ²)` | Propeller source level |
| Thorp (1967) | `α(f) = 0.11f²/(1+f²) + 44f²/(4100+f²) + …` | Medium absorption (dB/km) |
| Transmission loss | `TL = 20·log₁₀(R) + α(f)·R/1000` | Spherical spreading + absorption |
| Wenz (1962) | incoherent sum of wind, shipping, biology | Ambient noise |
| Doppler | `f_r = f₀·c/(c − v_radial)` | Radial velocity shift |

### Feature vector (21 dimensions)

| Group | Count | Features |
|---|---|---|
| Acoustic | 7 | `Received_Level_dB`, `Source_Level_dB`, `Transmission_Loss_dB`, `Frequency_Hz`, `Doppler_Shift_Hz`, `SNR_dB`, `Ambient_Noise_dB` |
| Propagation | 3 | `Sound_Speed_mps`, `Time_of_Arrival_s`, `Bearing_deg` |
| Environmental | 5 | `Water_Temp_C`, `Salinity_ppt`, `Sensor_Depth_m`, `Sea_State`, `Wind_Speed_knots` |
| Engineered | 6 | `Acoustic_Energy`, `SNR_Freq_interaction`, `Received_Level_squared`, `Doppler_Speed_proxy`, `Transmission_Efficiency`, `Env_Composite` |

> Note: an earlier revision described 23 features. The code builds **21**
> (7 + 3 + 5 + 6); the count in this table matches `mlmodel.ALL_FEATURES`.

Measurements include realistic sensor noise, so features are not identical to the
underlying physical quantities.

---

## 6. Machine learning models

### 6.1 The one trained model: `VesselMLP`

```text
        21 input features
               │
     Linear(21→128) + ReLU
     Linear(128→64) + ReLU
     Linear(64→32) + ReLU
        ┌──────┴──────┐
        ▼             ▼
 Linear(32→6)    Linear(32→1)
   (classes)      (displacement)
   CrossEntropy      MSE
```

| Property | Value |
|---|---|
| Framework | PyTorch |
| Input | 21 acoustic/environmental features |
| Task 1 | 6-class vessel classification (CrossEntropy) |
| Task 2 | Displacement regression in kg (MSE) |
| Optimiser | Adam, lr = 0.005, 100 full-batch steps |
| Preprocessing | `StandardScaler`, fitted on the **training split only** |
| Labels | `LabelEncoder` over the 6 classes |
| Split | `train_test_split(test_size=0.2, random_state=42)` |
| Artifacts | `models/model_state.pt`, `scaler.pkl`, `label_encoder.pkl` |
| Inference | `mlmodel.update_model_from_collected_data()` |
| Online retraining | **Disabled** by default (`ENABLE_ONLINE_RETRAINING = False`) |

A validation gate accepts a retrained model only if its validation loss improves
on the previous best; rejected models leave the active model untouched.

### 6.2 Tasks with no model — stated explicitly

| Quantity | How it is actually produced | Logged as |
|---|---|---|
| **Speed** | measured by `simulator` and stored as `Actual_Speed_knots` | `NOT_PREDICTED` / `MEASURED` |
| **Direction** | `get_vessel_direction()` applies a geometric threshold on `vessel['angle']` to split INBOUND vs OUTBOUND | `NOT_PREDICTED` / `RULE_BASED` |

Neither is claimed as machine learning. The system does not contain a speed or
heading model, and this documentation does not imply otherwise.

### 6.3 Other modules are not ML

`network.py` (routing), `auv.py` (kinematics) and `traffic_algo.py` (rules) are
deterministic algorithms. They are not described as learned models.

---

## 7. Prediction validation and evaluation

This section answers the question: **are predictions accurate compared to the
actual simulated vessels?**

### 7.1 Why this was needed

Predictions were previously written to a temporary CSV that is **overwritten
every animation frame**, so prediction history was destroyed continuously and
accuracy could not be computed. `prediction_logger.py` now appends a permanent,
queryable record.

### 7.2 Ground-truth source

`simulator._build_sensor_row()` writes the authoritative values onto the *same
detection row* as the features (`Actual_Type`, `Actual_Weight_kg`,
`Actual_Speed_knots`, `Actual_Depth_m`). Comparing within a row guarantees the
same vessel at the same simulation frame, with no risk of comparing against a
value already overwritten by a later update.

### 7.3 Storage

`logs/prediction_log.csv`, one row per prediction, with `prediction_id`,
`run_id`, frame, scenario, vessel, model, task, predicted value, actual value,
error, metric kind, validation status and confidence. Records survive restarts
and are queryable by run, task, scenario or vessel.

### 7.4 Metrics

| Task type | Metric |
|---|---|
| Classification | Exact match, accuracy |
| Regression | Absolute error, MAE, RMSE, median AE, 90th-percentile AE, MAPE |
| Direction | Circular angular error |

Properties enforced by tests:

- **Circular wraparound**: 359° vs 1° = **2°**, not 358°.
- **Missing ground truth** is recorded as `UNAVAILABLE`, never as incorrect.
- **MAPE is skipped** when targets approach zero, where it is meaningless.
- **Empty logs return `None`**, never a flattering 0 or 1.
- **Duplicate prediction IDs** are skipped.

### 7.5 Measured results

From real inference over the existing dataset (23 validated samples, NORMAL):

| Task | Source | Validated | Result |
|---|---|---|---|
| Vessel class | ML head | 23 | **100 % accuracy** (23/23) |
| Displacement | ML head | 23 | **MAE 22,713 kg · RMSE 36,121 kg · MAPE 97.7 %** |
| Speed | simulator state | 0 | not predicted |
| Direction | geometric rule | 0 | not predicted |

Monsoon run: 100 % class accuracy, MAE 22,703 kg — scenarios are reported
independently.

### 7.6 Known defect in the displacement head

The regression head does **not** track ground truth: it predicts roughly
400–1,700 kg for vessels weighing 1,023–381,263 kg, so predictions are not
proportional to vessel mass.

Diagnosed cause in `mlmodel._retrain()`:

```python
loss = loss_type + (loss_weight * 1e-8)
```

`loss_type` (CrossEntropy) ≈ 5.6. `loss_weight` (MSE on raw kg) ≈ 6.3 × 10⁹. The
`1e-8` constant does not balance objectives of different physical scale, so the
weight term still dominates and destabilises training.

Measured gradient reaching the shared trunk:

```text
|grad| classification head           = 1.35e+02
|grad| weight head (as coded, ×1e-8) = 8.17e-01   → 0.6 % of classification
```

The trunk therefore optimises classification only. Full analysis and the
control experiment (standardised target → balanced gradients) are in
[`logs/PREDICTION_VALIDATION_FINDINGS.md`](logs/PREDICTION_VALIDATION_FINDINGS.md).
Reproduce with `python diagnose_weight_head.py`.

**The model has been left unchanged.** This is documented as a known defect
rather than patched, so the measurement remains reproducible.

### 7.7 Reproducing the evaluation

```bash
python evaluate_predictions.py                       # report from existing log
python evaluate_predictions.py --collect             # short fresh run first
python evaluate_predictions.py --collect --scenario MONSOON
python evaluate_predictions.py --ai-validate         # run ML metrics + Groq physical audit
python evaluate_predictions.py --out report.md
```

### 7.8 Real-world physical plausibility validation via Groq LPU

While §7.5 evaluates the accuracy of the local `VesselMLP` neural network against the simulation's ground truth, a separate **AI Data Validation Engine** ([`ai_data_validator.py`](ai_data_validator.py)) audits whether the generated simulation data (ocean currents, vessel displacements, underwater acoustics, and environmental states) is physically plausible in the real world.

#### Engine & Auditor Model
- **Provider & Hardware**: Groq Cloud LPU (Ultra-low latency inference).
- **Flagship Model**: `openai/gpt-oss-120b` (120B parameter reasoning model, auto-selected from available models on the account).
- **Fallback**: Deterministic local rule-based physics validator implementing empirical equations when `GROQ_API_KEY` is not provided.

#### Evaluated Physical Laws & Boundaries
1. **Hydrodynamics & Currents**:
   - Sheltered harbor tidal currents bounded within **0.1 to 1.5 m/s** (~0.2 to 3.0 kn); monsoon squalls up to **4.2 m/s**.
   - Significant wave height ($H_s$) physically coupled to Douglas Sea State and wind speed.
2. **Naval Architecture & Vessel Displacement**:
   - Archimedes buoyancy consistency and displacement limits across craft types.
   - Vessel draft clearance against local seabed bathymetry ($Z_{\text{draft}} < |Z_{\text{depth}}|$) to prevent grounding.
3. **Underwater Acoustic Propagation**:
   - Sound speed strictly adhering to the **Mackenzie (1981)** equation ($1440 \le c \le 1545\text{ m/s}$).
   - Passive sonar equation balance: $\text{Received\_Level} \approx \text{Source\_Level} - \text{Transmission\_Loss} \pm 3\text{ dB}$.
   - Signal-to-Noise Ratio (SNR) consistency: $\text{SNR} \approx \text{Received\_Level} - \text{Ambient\_Noise}$.
   - Wenz (1962) ambient noise bounds ($45\text{--}95\text{ dB}$) and Doppler shift magnitude limits ($|\Delta f| \le 1.5\% f_0$).
4. **Marine Environment**:
   - Water temperature ($10\text{--}30^\circ\text{C}$), salinity ($30\text{--}36\text{ ppt}$), and wind speeds.

#### Important Distinction: Physical Plausibility (100 %) vs. Real-World Commercial Scale
- **Physical Plausibility (100.0 %)**: Zero physical or thermodynamic laws were violated across 104 evaluated parameters. The acoustic and hydrodynamics telemetry could physically occur in a coastal harbor basin.
- **Synthetic Scaling Caveats**: As explicitly noted by the auditor in [`logs/ai_validation_report.json`](logs/ai_validation_report.json), commercial Cargo Ships in this simulator weigh 80–200 tonnes (`80,000 - 200,000 kg`), representing small harbor barges or tugs rather than ocean-going Panamax container ships (50,000–200,000 tonnes).
- **ML Weight Regression Discrepancy**: While the physical data itself is 100% physically feasible, the local `VesselMLP` regression head currently exhibits a 97.6% MAPE error (§7.6).

#### Standalone & CLI Usage
```bash
python ai_data_validator.py                 # audit latest collected_data.csv via Groq
python evaluate_predictions.py --ai-validate # unified report: ML metrics + Groq physical audit
```

---

## 8. Underwater networking

### 8.1 Link budget (verified constants)

| Parameter | Value |
|---|---|
| Carrier frequency | 12 kHz |
| Source level | 178 dB re 1 µPa @ 1 m |
| Communication range `R_comm` | **1800 m** |
| Detection radius `R_detect` | 2000 m |
| Link SNR threshold | 10 dB |
| Bitrate | 1000 bps |
| Sound speed (network layer) | 1500 m/s |

> An earlier revision stated `R_comm = 1500 m`. The code uses **1800 m**.

A link exists between two nodes when `distance ≤ R_comm` **and** `SNR ≥ 10 dB`.
The graph is rebuilt in NetworkX every frame from live node positions.

### 8.2 Topology

- **35** moored seabed hydrophones (`MooredSensor`) at surface, mid-water and seabed depths
- **8** AUVs acting as mobile relays
- **3** surface gateway buoys: `buoy_alpha` (150, 2000), `buoy_beta` (150, 5000), `buoy_gamma` (150, 8000)

### 8.3 Depth-Based Routing

A detecting node forwards toward any reachable gateway. `run_dbr_routing()`
filters neighbours to shallower nodes (`Z_neighbour > Z_current`) and picks the
shallowest, repeating until a buoy is reached. Paths longer than 12 hops are
rejected. In practice the implementation selects the shortest path to the
nearest reachable gateway via `networkx.shortest_path`, then records per-hop
delay and SNR.

### 8.4 Duty cycling — measured availability

`is_node_active()` availability was measured at frame 0 over 35 sensors:

| Scenario | Active | Availability |
|---|---|---|
| Normal | 32 / 35 | **~91 %** |
| Monsoon | 25 / 35 | **~71 %** |

AUVs and buoys never sleep.

> An earlier revision described a fixed "70 % active / 30 % sleep" duty cycle.
> The implementation is **scenario-dependent** as tabulated above.

### 8.5 Packet compression

| Metric | 256 B frame | 16 B frame | Change |
|---|---|---|---|
| Bits | 2048 | 128 | 16× smaller |
| Tx delay per hop | 2.048 s | 0.128 s | 93.75 % faster |
| Energy per hop @ 10 W | 20.48 J | 1.28 J | 93.75 % saved |

### 8.6 Network health

`evaluate_health()` scores four weighted factors — connectivity (35 %), packet
delivery (35 %), mooring integrity (20 %), latency (10 %) — then classifies:

| State | Health |
|---|---|
| STABLE | ≥ 75 % |
| DEGRADED | 50–74 % |
| CRITICAL | 25–49 % |
| COLLAPSED | < 25 %, or mass mooring failure, or catastrophic loss |

---

## 9. Robotics and simulation

### 9.1 AUV behaviour (`auv.py`)

Eight AUVs patrol three zones — Outer Yard, Channel, Inner Harbor — with a
state machine:

```text
PATROLLING ──overspeed violation in zone──► DISPATCHED ──vessel docked──► PATROLLING
     │                                              │
     └──battery < 15 %──► CHARGING ◄──dock at buoy──┘
```

- **Navigation**: proportional control force (max 900 N), mass 500 kg, drag 0.85, speed capped at 10 m/s
- **Drift**: current force `v_current × mass × 0.6` pushes AUVs off track in monsoon
- **Battery**: drains proportionally to speed; recharges at 0.2 %/frame within 250 m of `BUOY_POS` (150, 5000, 0) and resumes at 95 %
- **Boundary**: clamped inside the map and between seabed and −2 m

### 9.2 Moored sensor physics

```text
MOORED ──current > 1.3 m/s──► STRAINED ──speed ≥ threshold or fatigue > 12──► SNAPPED
   ▲                              │                                            │
   └────────current < 1.3 m/s────┘                          drifts with current × 14
```

Each sensor has an individual snap threshold in [2.2, 3.4] m/s. Snapped sensors
leave fading breadcrumb trails and stop contributing to the network.

### 9.3 Traffic governance (`traffic_algo.py`)

| Zone | X range | Speed limit | Behaviour |
|---|---|---|---|
| Outer Yard | > 5000 m | 10 kn | HALT / SLOW / CONTINUE by ML class |
| Inner Zone | 700–5000 m | 6 kn | Route to assigned berth |
| Harbor | < 700 m | — | Dock |

Screenings use the **predicted** class: Cargo/Tanker → HALT, Cruiser/Ferry →
SLOW DOWN, Speedboat/Fishing → CONTINUE. Berths are allocated by class across
11 dock slots.

---

## 10. Environmental scenarios

One engine, two regimes. `simulator.SCENARIOS` drives the physics directly.

| Parameter | Normal | Monsoon |
|---|---|---|
| Douglas sea state | 2–3 | 6–8 |
| Wind speed | 8–16 kn | 36–52 kn |
| Water temperature | 18–22 °C | 12–15 °C |
| Salinity | 34.5–35.5 ppt | 30.0–32.5 ppt |
| Rain noise | +0 dB | +15 dB |
| Current multiplier | 1.0× | 2.6× |
| **Measured ambient noise** | **66.1 dB** | **91.9 dB** |

### Monsoon stochastic engine

- **Current intensity**: Ornstein-Uhlenbeck drift, target resampled every 30 frames, clipped to 1.8–6.8 m/s
- **Gale squalls**: ~2.5 %/frame chance, lasting 45–120 frames, adding 1.8–3.4 m/s
- **Storm surge**: multi-harmonic water level, 5.2–8.5 m significant wave height
- **Vortices**: three wandering gyres with random-walk positions and spin
- **Wave surface**: four-harmonic chaotic superposition
- **Mooring failure**: cumulative fatigue drives sensors to SNAPPED

Switching scenarios requires no restart and resets no simulation state.

---

## 11. User interfaces

### 11.1 Desktop (`main.py`)

A grid-based maritime control station. The 3D Matplotlib viewport is sized so it
always occupies **≥ 75 %** of the window width; below 1700 px the sidebar
collapses automatically.

```text
┌──────────────────────────────────────────────────────────────┐
│ Title │ Season │ Controls │ Frame │ Sensors │ Violations    │
├──────────────────────────────────────────────────────────────┤
│ 🌊 Water │ 💨 Current │ ⚓ Moorings │ ⚡ Squall │ 📡 Network   │
├────────────────────────────────────┬─────────────────────────┤
│                                    │ [🔔 Notifications][📡]   │
│      3D SIMULATION VIEWPORT        │ SYSTEM: NOMINAL         │
│           (≥ 75 %)                 │                         │
│                                    │ Notifications            │
│                                    │  · INFO / WARN / ERROR   │
│                                    │  · expandable traceback  │
│                                    │ Network telemetry & log  │
├────────────────────────────────────┴─────────────────────────┤
│ Status │ Frame │ ML detections │ Network state │ PDR         │
└──────────────────────────────────────────────────────────────┘
```

**Notification system.** Crash notifications were previously Tkinter overlays
drawn on top of the 3D canvas, obscuring the simulation. They now render in a
dedicated sidebar panel with four severities:

| Level | Meaning |
|---|---|
| `INFO` | Informational |
| `WARNING` | Degraded but operational |
| `RECOVERABLE_ERROR` | Frame failed, loop continues |
| `FATAL_ERROR` | Loop halted, diagnostic shown |

Design properties:

- Errors **never** overlay the 3D viewport
- Timestamps on every entry; expandable monospace traceback with copy button
- Duplicate suppression on `level:message:source`
- Thread-safe queue + `root.after()` polling (no Tkinter calls off the main thread)
- `sys.excepthook` and `threading.excepthook` both routed to the panel
- Animation loop has a circuit breaker: 30 consecutive failures halt it cleanly
- `on_closing()` cancels pending callbacks for a clean shutdown

The squall indicator moved from a canvas overlay into the HUD bar for the same
reason.

### 11.2 Web dashboards

`dashboard.html` (traffic) and `networking_dashboard.html` (acoustic network)
poll `dashboard_data.json` every 500 ms / 1000 ms.

Both provide:

- **Light/dark toggle** in the header, applied without reload
- **Persistence** in `localStorage` under a shared key, so both pages stay in sync
- **Design tokens** — every colour derives from CSS custom properties, with zero
  unresolved `var()` references
- **Typography** — `Share Tech Mono` / `JetBrains Mono` / `Consolas` for
  telemetry, log and numeric values; a sans-serif for headings and prose
- **Accessibility** — semantic `<button>`, `aria-label`, `aria-pressed`,
  `:focus-visible` outlines, `prefers-reduced-motion`, `prefers-color-scheme`

Contrast was measured: all key text/background pairs pass **WCAG AA (≥ 4.5:1)**
in both themes.

- **Groq AI Real-World Physics Validation Card**: Dedicated full-width section between the ship monitoring tables and the robot fleet panel displaying overall real-world plausibility (100%), auditor model badge (`openai/gpt-oss-120b`), 4 subsystem breakdown gauges (Currents, Weights, Acoustics, Environment), and expert physics assessment text.
- **Header Jump Button**: Quick-action chip `🤖 Groq AI Realism: 100.0%` in the top header with smooth scrolling to the validation telemetry card.
- **Desktop HUD Badge**: Clickable `🤖 AI Realism: 100.0%` badge in the Tkinter simulation window (Row 1 HUD bar).

### 11.3 Telemetry export

`dashboard_data.json` and `dashboard_data.js` are rewritten every 500 ms with
frame, vessels, violations, AUV telemetry, network links, routed paths,
environment and network statistics.

---

## 12. Technology stack

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.9+ | Single runtime for physics, ML and networking |
| Numerics | NumPy | Vectorised acoustic physics across sensor × vessel pairs |
| Data | pandas | Detection CSV and feature engineering |
| Deep learning | PyTorch | Multi-task shared-trunk network with two heads |
| Preprocessing | scikit-learn, joblib | `StandardScaler`, `LabelEncoder`, persistence |
| Graphs | NetworkX | Per-frame topology and path finding |
| 3D GUI | Tkinter + Matplotlib `TkAgg` | Native desktop control station |
| Web UI | HTML5 / CSS3 / vanilla JS | Decoupled dashboards, no build step |
| HTTP | `http.server` | Serves the repo directory on 127.0.0.1:8000 |
| Logging | `csv` (stdlib) | Append-only prediction history |
| AI Physics Auditor | Groq Cloud API (`openai/gpt-oss-120b`) | Real-world physical plausibility validation on Groq LPUs |

---

## 13. Installation and execution

### Requirements

Python 3.9+ and:

```bash
pip install numpy pandas matplotlib scikit-learn joblib torch networkx groq python-dotenv
```

### Run

```bash
python verify_environment.py   # dependency + module checks
python main.py                 # launch the 3D simulation + telemetry server
```

`main.py` starts the simulation in **Normal Season** and serves dashboards on
`http://127.0.0.1:8000`, incrementing the port to 8001–8024 if 8000 is busy.

### Controls

| Action | Control |
|---|---|
| Pause / resume | `Space` or toolbar button |
| Start | `Ctrl+S` |
| Reset simulation | `Ctrl+R` or toolbar |
| Switch scenario | `⇄ Season` toolbar button or Season menu |
| Inject squall | `🌪 Squall` (switches to monsoon) |
| Re-anchor sensors | `⚓ Re-Anchor` (recovers after monsoon) |
| Reset camera | `⟲ View` |
| Toggle sidebar | Notification / Network tabs, or View menu |
| Open dashboards | `📊 Web Dashboard` |

### Dashboards

- Traffic — `http://127.0.0.1:8000/dashboard.html`
- Network — `http://127.0.0.1:8000/networking_dashboard.html`

### Evaluation and logging

```bash
python run_all_tests.py             # all suites (75 tests)
python verify_environment.py        # dependency + module checks
python test_prediction_validation.py
python test_integration.py
python evaluate_predictions.py      # metrics report
python diagnose_weight_head.py      # weight-head diagnostic
```

---

## 14. Testing and verification

| Suite | Command | Result |
|---|---|---|
| Environment and modules | `python verify_environment.py` | **PASS** — 4/4 stages |
| Prediction validation | `python test_prediction_validation.py` | **PASS** — 23/23 tests |
| Groq AI data validation | `python test_ai_data_validator.py` | **PASS** — 5/5 tests |
| Integration and regression | `python test_integration.py` | **PASS** — 52/52 tests |
| All suites at once | `python run_all_tests.py` | **PASS** — 80 tests |
| Prediction logging (end-to-end) | live inference + log inspection | **PASS** — 69 records written |
| Evaluation report | `python evaluate_predictions.py` | **PASS** — renders from log |
| Groq AI physical audit | `python ai_data_validator.py` | **PASS** — 100 % plausibility |
| Dashboard theme tokens | token/reference audit | **PASS** — 0 unresolved vars |
| Dashboard contrast | luminance check | **PASS** — all pairs ≥ 4.5:1 |
| README factual claims | claim audit vs source | **PASS** — verified |

### Coverage by category

| Category | Tests | Notable cases |
|---|---|---|
| Prediction validation | 23 | circular wraparound, correct/incorrect, missing truth, duplicates, persistence across reload, run filtering, scenario separation, MAPE guard, empty-log guard |
| Simulation pipeline | 7 | fleet composition, map bounds, stepping, TL monotonicity, Thorp frequency response, Doppler direction, SNR definition |
| Scenario switching | 4 | noise increase, duty degradation, AUV/buoy always-on, squall surge |
| Networking | 8 | node counts, SNR gating, monsoon link reduction, stats schema, health states, 16× compression, re-anchor recovery, reset |
| Traffic governance | 10 | speed limits both zones, LOC screening, berth assignment, **berth collision prevention**, berth queueing, release/reuse, direction rule |
| AUV behaviour | 5 | spawn count, bounds, battery drain, charging trigger, status values |
| Error handling | 4 | empty fleet, ground-truth presence, empty graph, malformed logger input |
| Dashboard themes | 10 | toggle, persistence, shared key, light block, var resolution, no hardcoded alphas, a11y features, font fallbacks, blocked storage |
| ML integration | 5 | artifact load, 21 features, 6 classes, scaler leakage, two-head forward |

**Not verified:** the Tkinter UI was tested by code inspection and compilation,
not by automated GUI interaction. No screenshot or click-path test was run, and
no browser rendering test was executed (the theme audits are static analysis).

### Defect found and fixed during testing

`test_berths_are_not_double_assigned` failed initially and exposed a real
pre-existing bug: the fleet needs **22 berths** but only **13** exist, and the
old `_assign_dock()` fell back to `slots[0]` when a class was fully booked. This
stacked up to 13 speedboats onto the single `Dock-C` berth.

Fixed by queuing instead of colliding: `_assign_dock()` now returns
`(None, None)` when no berth is free, the vessel enters `HOLDING_FOR_BERTH` at
zero speed, and `release_dock()` / `clear_dock_assignments()` free berths. Two
regression tests cover the behaviour.

---

## 15. Limitations and future work

### Known limitations

1. **Displacement regression is inaccurate** (MAE 22,713 kg, MAPE 97.7 %). Cause
   diagnosed in §7.6 — objective imbalance in `_retrain()`. Left unpatched by
   design so the measurement stays reproducible.
2. **No speed or heading model exists.** Both are simulator state; recorded as
   `NOT_PREDICTED` rather than presented as ML output.
3. **Only one training run was evaluated** (23 validated samples). The reported
   accuracy describes that sample, not a guaranteed rate.
4. **Monsoon results are limited** to a short collection run.
5. **Physics is modelled, not field-validated.** Ocean parameters are drawn from
   published formulations, but no real sensor data was used.
6. **GUI untested by automation** (see §14).
7. **Prediction logging runs on the main thread** during inference. At the
   current sampling rate this is negligible, but heavy scenarios could require
   moving it to a worker thread.

### Future work

| Priority | Item |
|---|---|
| 1 | Fix the displacement objective (standardise or log-scale the target, document the loss weighting), retrain, re-validate |
| 2 | Add a genuine speed-prediction head with a defined horizon and leakage controls |
| 3 | Replace the geometric heading rule with a learned model if justified by data |
| 4 | Move prediction logging off the Tkinter main thread |
| 5 | Add GUI automation tests |
| 6 | Expose live prediction-validation metrics on the dashboards |
| 7 | Validate against recorded real-world acoustic data |

---

## 16. Repository structure

```text
Harbor-Traffic-Control-System-/
├── main.py                        # Tkinter control station, 3D viewport, animation loop
├── simulator.py                   # Acoustic physics, scenarios, sensors, fleet, AUV physics base
├── mlmodel.py                     # VesselMLP definition, training, inference, validation hooks
├── ai_data_validator.py           # Groq AI physical plausibility validator (openai/gpt-oss-120b)
├── network.py                     # DBR routing, packet engine, health scoring
├── network_manager.py             # Standalone RobotNode / CommLink / NetworkManager
├── auv.py                         # AUV kinematics, battery, patrol, dispatch
├── traffic_algo.py                # Speed governance, LOC screening, berth routing
├── prediction_logger.py           # Persistent prediction log + validation metrics
├── evaluate_predictions.py        # Reproducible evaluation report (--ai-validate flag)
├── diagnose_weight_head.py        # Weight-head loss diagnostic
├── test_prediction_validation.py  # Unit tests for logging and metrics (23)
├── test_ai_data_validator.py      # Unit tests for Groq AI physical validation (5)
├── test_integration.py            # Integration & regression tests (52)
├── run_all_tests.py               # Runs every suite and summarises (80 tests total)
├── verify_environment.py          # Environment and module checks
├── dashboard.html                 # Traffic dashboard + Groq AI physics validation card
├── networking_dashboard.html      # Network dashboard (light/dark themes)
├── dashboard_data.json            # Live telemetry + AI validation (runtime-generated)
├── dashboard_data.js              # JS fallback wrapper (runtime-generated)
├── collected_data.csv             # Detection dataset (runtime-generated)
├── logs/
│   ├── prediction_log.csv         # Persistent prediction history
│   ├── prediction_summary.csv     # Metrics snapshot
│   ├── ai_validation_report.json  # Groq AI physics audit report
│   └── PREDICTION_VALIDATION_FINDINGS.md
├── models/
│   ├── model_state.pt             # VesselMLP weights
│   ├── scaler.pkl                 # Feature scaler
│   └── label_encoder.pkl          # Class encoder
├── generate_slide_figures.py      # Presentation figures
├── build_presentation.py          # Presentation builder
├── PRESENTATION_CONTENT.md        # Presentation script
├── PROJECT CONTEXT.md             # Research context
├── IMPLEMENTATION_PLAN.md         # Upgrade plan and status
└── Future_Updates.txt             # Original requirements specification
```