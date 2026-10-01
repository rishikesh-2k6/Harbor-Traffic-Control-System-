# ⚓ Harbor Traffic Control System
### Adaptive Multi-Robot Networking & Acoustic Telemetry in Underwater Wireless Sensor Networks (UWSN)

> **Autonomous Maritime Traffic Management & Multi-Robot Swarm Telemetry**  
> *A cyber-physical simulation combining 3D underwater acoustic physics, selectable environmental scenarios (Normal Season vs. Monsoon Season) on a **single unified simulation engine**, depth-based multi-hop routing (DBR), swarm Autonomous Underwater Vehicle (AUV) kinematics, multi-task PyTorch vessel classification and weight regression, and real-time decoupled web dashboards across a **10 km × 10 km harbor domain**.*

---

## 📑 Table of Contents
1. [Project Objectives](#-1-project-objectives)
2. [Key Demonstrations & Capabilities](#-2-key-demonstrations--capabilities)
3. [The Working Behind the Models](#-3-the-working-behind-the-models)
   - [3.1 Oceanographic Acoustic Physics Engine](#31-oceanographic-acoustic-physics-engine)
   - [3.2 The 23-Feature Acoustic Telemetry Pipeline](#32-the-23-feature-acoustic-telemetry-pipeline)
   - [3.3 Multi-Task Deep Neural Network (`VesselMLP`)](#33-multi-task-deep-neural-network-vesselmlp)
   - [3.4 Pre-Trained Model Inference & Pipeline](#34-pre-trained-model-inference--pipeline)
4. [Environmental Scenarios: Normal vs. Monsoon](#-4-environmental-scenarios-normal-vs-monsoon)
   - [4.1 Unified Engine & Scenario Selection](#41-unified-engine--scenario-selection)
   - [4.2 Comparative Atmospheric & Oceanographic Parameters](#42-comparative-atmospheric--oceanographic-parameters)
   - [4.3 Impact on Acoustic Modem SNR & DBR Routing](#43-impact-on-acoustic-modem-snr--dbr-routing)
   - [4.4 Hydrodynamic Impact on AUV Swarm & Battery](#44-hydrodynamic-impact-on-auv-swarm--battery)
   - [4.5 Monsoon Simulation: Non-Deterministic Physics Engine](#45-monsoon-simulation-non-deterministic-physics-engine)
5. [Network Arrangement: Fixed vs. Dynamic](#-5-network-arrangement-fixed-vs-dynamic)
   - [5.1 Fixed Infrastructure (Seabed Hydrophones & Surface Buoys)](#51-fixed-infrastructure)
   - [5.2 Dynamic Infrastructure (AUV Swarm Kinematics)](#52-dynamic-infrastructure-auv-swarm)
   - [5.3 Harbor Zone Architecture & Spatial Layout](#53-harbor-zone-architecture)
6. [Sensor Networking & Acoustic Communication Protocols](#-6-sensor-networking--acoustic-communication)
   - [6.1 Acoustic Channel Modeling & Link Budget](#61-acoustic-channel-modeling--link-budget)
   - [6.2 Link Hierarchy & Active Connection Types](#62-link-hierarchy--active-connection-types)
   - [6.3 Depth-Based Routing (DBR) Protocol](#63-depth-based-routing-dbr-protocol)
   - [6.4 Duty-Cycling & Swarm Resilience](#64-duty-cycling--swarm-resilience)
   - [6.5 Frame Compression, Latency & Energy Metrics](#65-frame-compression-latency--energy-metrics)
7. [Maritime Traffic Management & Autonomous Control](#-7-maritime-traffic-management--autonomous-control)
8. [System Architecture & End-to-End Data Pipeline](#-8-system-architecture--end-to-end-data-pipeline)
9. [Technology Stack & Architectural Rationale](#-9-technology-stack--selection-rationale)
10. [Project Directory & Module Structure](#-10-project-directory--module-structure)
11. [Quickstart & Execution Guide](#-11-quickstart--execution-guide)
12. [Live Telemetry Data Schema (`dashboard_data.json`)](#-12-live-telemetry-data-schema)
13. [Summary of Innovations](#-13-summary-of-innovations)

---

## 🎯 1. Project Objectives

Traditional maritime traffic control relies predominantly on surface radar (VTS/AIS) and satellite tracking. However, radar cannot penetrate water columns, AIS can be spoofed or deactivated by illicit vessels, and isolated seabed sensor grids suffer from high propagation latency, short battery lifespans, and communication dead zones. 

The primary objective of this project is to create an **Autonomous, Active, Multi-Robot Underwater Wireless Sensor Network (UWSN)** capable of:

1. **Passive-to-Active Transformation**: Bridging stationary seabed hydrophones with a dynamic mobile swarm of Autonomous Underwater Vehicles (AUVs) to monitor a $10\text{ km} \times 10\text{ km}$ coastal harbor domain.
2. **Acoustic Physics Grounding**: Simulating underwater acoustic wave propagation, Doppler shifts, transmission loss, and ambient ocean noise using empirical oceanographic formulations.
3. **Selectable Environmental Scenarios**: Evaluating system resilience under distinct weather regimes (**Normal Season** vs. **Monsoon Season**) using **ONE unified simulation engine** with scientifically grounded parameters.
4. **Multi-Task Vessel Identification**: Passively extracting 23 acoustic signatures from vessel radiated noise to classify ship types (*Cargo Ship, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel*) and predict displacement weight in kilograms using deep learning.
5. **Resilient Underwater Networking**: Dynamically establishing and maintaining multi-hop acoustic communication links between seabed sensors, mobile AUVs, and surface buoys using Depth-Based Routing (DBR) under severe duty-cycling (sleep/wake) conditions.
6. **Automated Traffic Governance**: Monitoring virtual Line of Control (LOC) boundaries, tracking speed violations, dispatching patrolling AUVs to shadow violators, and routing vessels safely to designated harbor dock berths.

---

## 💡 2. Key Demonstrations & Capabilities

* **Unified Single Simulation Engine**: Seamlessly switches between environmental scenarios without duplicating physics engines or creating second Tkinter window lifecycles.
* **Real-time Acoustic Physics Modeling**: Implemented in pure NumPy without heavy external C++ engines, calculating exact Doppler shifts, Thorp attenuation, and Mackenzie sound speed profiles for every detection event.
* **Measurable Monsoon Impact**: Grounded in the Wenz ambient ocean noise model, monsoon gale winds and torrential rain droplet noise surge ambient noise by $+26\text{ dB}$, naturally degrading modem SNR and stressing multi-hop DBR routing.
* **Multi-Task Deep Learning Inference**: Simultaneously classifying vessel categories (6 classes) and regressing displacement weight ($1\,\text{kTon}$ to $400\,\text{kTon}$) via a unified PyTorch neural network sharing latent representations.
* **Swarm AUV Kinematics & Adaptive Relay**: Mobile AUVs patrol designated sectors, steer via 3D waypoint navigation, account for depth-dependent ocean current drift, manage battery drain/dock charging, and act as mobile data mules when static sensors sleep.
* **Dynamic Network Topology & Depth-Based Routing (DBR)**: Frame-by-frame graph construction in NetworkX, resolving multi-hop acoustic routes from seabed depths (up to $-65\text{m}$) to surface gateway buoys ($Z=0\text{m}$).
* **Energy Optimization via Packet Compression**: Comparing default ($256\text{ bytes}$) versus compressed ($16\text{ bytes}$) acoustic packets, demonstrating dramatic reductions in transmission latency and acoustic modem energy consumption.
* **Decoupled Real-Time Web Telemetry**: Serving 3D harbor telemetry snapshots via an asynchronous Python HTTP server to interactive HTML5/Canvas/JS dashboards without blocking the 3D Matplotlib simulation loop.

---

## 🧠 3. The Working Behind the Models

### 3.1 Oceanographic Acoustic Physics Engine

The simulation environment integrates empirical underwater acoustics to generate realistic physical telemetry for each vessel detection:

```text
Vessel Propeller/Hull Noise (Ross 1976)
                  │
                  ▼
   Sound Speed Profile (Mackenzie 1981)
                  │
                  ▼
   Transmission Loss (Spherical + Thorp 1967)
                  │
                  ▼
   Ambient Sea Noise (Wenz 1962 + Rain Impact)
                  │
                  ▼
   Doppler Shift & Time of Arrival (ToA)
```

#### 1. Mackenzie (1981) Nine-Term Sound Speed Equation
Sound speed $c$ ($\text{m/s}$) depends non-linearly on temperature $T$ ($^\circ\text{C}$), salinity $S$ ($\text{ppt}$), and depth $D$ ($\text{m}$):
$$c(T, S, D) = 1448.96 + 4.591T - 5.304 \times 10^{-2}T^2 + 2.374 \times 10^{-4}T^3 + 1.340(S - 35) + 1.630 \times 10^{-2}D + 1.675 \times 10^{-7}D^2 - 1.025 \times 10^{-2}T(S - 35) - 7.139 \times 10^{-13}TD^3$$

#### 2. Ross (1976) Propeller Source Level Equation
Ship radiated acoustic source level $SL$ ($\text{dB re } 1\,\mu\text{Pa @ } 1\text{m}$) is driven by ship speed $v$ ($\text{knots}$), propeller diameter $d_{\text{prop}}$ ($\text{m}$), and base machinery level:
$$SL = SL_{\text{base}} + 55\log_{10}\left(\frac{v}{10}\right) + 20\log_{10}\left(\frac{d_{\text{prop}}}{3.0}\right) + \mathcal{N}(0, \sigma^2)$$

#### 3. Thorp (1967) Medium Absorption Formula
Acoustic absorption coefficient $\alpha(f)$ ($\text{dB/km}$) for frequency $f$ ($\text{kHz}$):
$$\alpha(f) = \frac{0.11 f^2}{1 + f^2} + \frac{44 f^2}{4100 + f^2} + 2.75 \times 10^{-4}f^2 + 0.003$$

#### 4. Transmission Loss ($TL$)
Combining spherical geometric spreading with medium absorption over 3D distance $R$ ($\text{m}$):
$$TL(R, f) = 20\log_{10}(R) + \alpha(f)\cdot \left(\frac{R}{1000}\right)$$

#### 5. Wenz (1962) Ambient Ocean Noise Model with Rain Augmentation
Combining wind-driven surface turbulence, rain droplet impact noise, shipping density factor, and biological background noise:
$$NL_{\text{ambient}} = 10\log_{10}\left(10^{(NL_{\text{wind}} + NL_{\text{rain}})/10} + 10^{NL_{\text{shipping}}/10} + 10^{NL_{\text{bio}}/10}\right)$$
Where $NL_{\text{wind}} = 44 + 20\log_{10}(v_{\text{wind\_knots}})$.

#### 6. Acoustic Doppler Shift
Radial velocity $v_{\text{radial}} = \vec{v}_{\text{ship}} \cdot \hat{u}_{\text{sensor}}$ relative to the sensor shifts observed frequency:
$$f_{\text{received}} = f_{\text{source}} \cdot \left(\frac{c}{c - v_{\text{radial}}}\right), \quad \Delta f = f_{\text{received}} - f_{\text{source}}$$

---

### 3.2 The 23-Feature Acoustic Telemetry Pipeline

When a hydrophone detects a vessel within its sensing range ($R_{\text{detect}} = 2000\text{m}$), it compiles a 23-dimensional feature vector:

| Feature Category | Features Included | Physical Meaning |
|---|---|---|
| **Acoustic Energy (7)** | `Received_Level_dB`, `Source_Level_dB`, `Transmission_Loss_dB`, `Frequency_Hz`, `Doppler_Shift_Hz`, `SNR_dB`, `Ambient_Noise_dB` | Hydrophone decibel levels, signal-to-noise ratio, frequency displacement. |
| **Wave Propagation (3)** | `Sound_Speed_mps`, `Time_of_Arrival_s`, `Bearing_deg` | Hydrodynamic sound velocity, travel latency ($R/c$), azimuth direction. |
| **Environmental Context (5)** | `Water_Temp_C`, `Salinity_ppt`, `Sensor_Depth_m`, `Sea_State`, `Wind_Speed_knots` | Local physical water column characteristics and weather state. |
| **Engineered Features (6)** | `Acoustic_Energy`, `SNR_Freq_interaction`, `Received_Level_squared`, `Doppler_Speed_proxy`, `Transmission_Efficiency`, `Env_Composite` | Non-linear cross-terms: energy in linear domain ($10^{RL/10}$), Doppler-frequency ratio, sound attenuation efficiency. |

---

### 3.3 Multi-Task Deep Neural Network (`VesselMLP`)

Rather than maintaining isolated models for classification and weight estimation, the system implements a unified multi-task PyTorch architecture:

```text
               Input Features (23 Dimensions)
                             │
                             ▼
               Linear(23 → 128) + ReLU
                             │
                             ▼
               Linear(128 → 64) + ReLU
                             │
                             ▼
               Linear(64 → 32) + ReLU
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
   [Classification Head]             [Regression Head]
     Linear(32 → 6)                    Linear(32 → 1)
             │                               │
             ▼                               ▼
  Vessel Class (Softmax)           Displacement Weight (kg)
```

* **Shared Representation**: 3 fully-connected hidden layers ($128 \rightarrow 64 \rightarrow 32$) extract joint latent embeddings representing acoustic vessel signatures.
* **Classification Head**: Predicts probability across 6 maritime vessel classes (*Cargo Ship, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel*) using Cross-Entropy Loss.
* **Regression Head**: Predicts continuous vessel displacement tonnage/weight ($\text{kg}$) using Mean Squared Error.

---

### 3.4 Pre-Trained Model Inference & Pipeline

* **Validated Artifacts**: The simulation automatically loads pre-trained weights (`model_state.pt`), feature scaler (`scaler.pkl`), and label encoder (`label_encoder.pkl`) during startup, ensuring sub-millisecond real-time inference without CPU/GPU blocking.
* **Concurrency Protection**: Implements exponential backoff retry loops (`_safe_read_csv`, `_safe_write_csv`) to prevent Windows disk-lock race conditions.
* **Inference Snapshots**: Writes real-time predictions per active ship to a temporary table (`stage3_live_predictions.csv`), ingested instantly by the traffic control algorithm.
* **Optional Retraining**: The retraining pipeline (`_retrain`) is preserved in `mlmodel.py` and can be toggled via `ENABLE_ONLINE_RETRAINING = True` if online continuous adaptation experiments are desired.

---

## 🌊 4. Environmental Scenarios: Normal vs. Monsoon

The system provides selectable environmental scenarios driven by **ONE underlying simulation engine**. Atmospheric and oceanographic parameters directly tune the acoustic physics, ambient noise, and hydrodynamic current fields.

```text
                 Harbor Traffic Control System
                               │
                      Startup Selection
                       /              \
                      /                \
            ☀️ Normal Season        ⛈️ Monsoon Season
                      \                /
                       \              /
                  SAME SIMULATION ENGINE
                               │
                    Acoustic + Network Model
                               │
                       DBR / AUV / Sensors
                               │
                      Real-Time Dashboards
```

### 4.1 Unified Engine & Scenario Selection
* **Startup Selection**: When `main.py` launches, an overlay card allows the user to select **Normal Season** or **Monsoon Season**.
* **Zero Engine Duplication**: The simulation does not spawn secondary processes or duplicate physics loops. Selecting a scenario updates the centralized `SCENARIOS` state in `simulator.py`.
* **Live Toggle**: Users can also switch scenarios dynamically at runtime using the `⇄ Switch Season` button in the top control bar to observe real-time network degradation and recovery.

### 4.2 Comparative Atmospheric & Oceanographic Parameters

| Environmental Parameter | Normal Season | Monsoon Season | Physical Oceanographic Basis |
|---|---|---|---|
| **Douglas Sea State** | **State 2–3** (Slight / Moderate) | **State 6–8** (Very Rough / High Seas) | Significant wave heights rise from $0.5\text{m}$ to $4.0 - 9.0\text{m}$. |
| **Wind Speed** | $8.0 - 16.0\text{ knots}$ | $36.0 - 52.0\text{ knots}$ | Near-gale to storm-force winds driving intense wave action. |
| **Water Temperature** | $18.0 - 22.0^\circ\text{C}$ | $12.0 - 15.0^\circ\text{C}$ | Coastal cooling due to heavy cloud cover and freshwater runoff. |
| **Water Salinity** | $34.5 - 35.5\text{ ppt}$ | $30.0 - 32.5\text{ ppt}$ | Brackish surface dilution caused by intense precipitation. |
| **Rain Noise Factor** | $0.0\text{ dB}$ | $+15.0\text{ dB}$ | High-frequency acoustic noise from torrential droplet impacts (Wenz / Nystuen model). |
| **Ambient Ocean Noise** | $\mathbf{\approx 65.8\text{ dB}}$ | $\mathbf{\approx 91.9\text{ dB}}$ ($+26\text{ dB}$ surge) | Incoherent sum of wind wave agitation and torrential rain impacts. |
| **Current Multiplier** | $1.0\times$ (peak $\sim 0.35\text{ m/s}$) | $2.6\times$ (peak $\sim 0.9 - 1.2\text{ m/s}$) | Monsoon storm surge combined with tidal oscillations. |

### 4.3 Impact on Acoustic Modem SNR & DBR Routing
Underwater modem communication links operate at $f = 12\text{ kHz}$ with source level $SL = 178\text{ dB re } 1\,\mu\text{Pa @ } 1\text{m}$ and threshold $\text{SNR} \ge 10.0\text{ dB}$.

* **Normal Season**:
  * Baseline ambient noise $\approx 65.8\text{ dB}$.
  * At $1000\text{m}$, modem link SNR is $\approx 50.5\text{ dB} \gg 10\text{ dB}$.
  * Modems communicate reliably across full range ($1500\text{m}$), producing direct, high-capacity links and high Packet Delivery Ratios ($>95\%$).
* **Monsoon Season**:
  * Ambient noise surges to $\approx 91.9\text{ dB}$ ($+26.1\text{ dB}$ noise floor increase).
  * At $1000\text{m}$, modem link SNR collapses to $\approx 24.5\text{ dB}$.
  * Links at the periphery ($>1200\text{m}$) drop below threshold and disconnect.
  * Direct transmission paths are severed; DBR routing is forced to discover deeper, multi-hop relay routes through mobile AUV nodes.

### 4.4 Hydrodynamic Impact on AUV Swarm & Battery
* **Hydrodynamic Drift**: Water current vectors ($v_x, v_y, v_z$) in `simulator.py` scale by $2.6\times$ under Monsoon conditions, applying significant force to AUV hulls ($F_{\text{current}} = \vec{v}_{\text{current}} \cdot M \cdot 0.6$).
* **Drift & Waypoint Keeping**: AUVs experience lateral drift during patrol legs, requiring active propulsion compensation to reach target coordinates.
* **Accelerated Battery Discharge**: Higher motor output against storm currents causes increased battery drain, triggering earlier return-to-buoy recharge maneuvers.

### 4.5 Monsoon Simulation: Non-Deterministic Physics Engine

The monsoon is not a static parameter change — it drives a live stochastic physics engine that evolves every simulation frame, producing genuinely unpredictable conditions.

#### Non-Deterministic Ocean Current Model (`OCEAN_STATE`)

A global `OCEAN_STATE` dictionary is stepped every frame by `step_ocean_environment()`, evolving via three coupled stochastic processes:

| Component | Mechanism | Intensity Range |
|---|---|---|
| **Base current intensity** | Ornstein-Uhlenbeck drift-diffusion: target resampled every 30 frames, Gaussian turbulence $\mathcal{N}(0, 0.16)$ per frame | $1.8 - 6.8\text{ m/s}$ |
| **Sudden gale squall bursts** | Poisson-rate random trigger (~2.5%/frame); duration 45–120 frames; surge layered on top of base | $+1.8 - 3.4\text{ m/s}$ extra |
| **Storm surge water level** | Multi-harmonic oscillation ($3.2 + 1.8\sin(t) + 0.9\cos(2t)$) with squall bonus | $+0.5 - 5.5\text{ m}$ |
| **Wandering vortex gyres** | 3 turbulent eddies with random-walk positions and spin-rate perturbation each frame | Radius $\approx 800\text{ m}$ |

The **wave surface field** is a 4-harmonic chaotic superposition evaluated at every grid point:
$$Z_{\text{surf}}(x,y,t) = \text{surge} + 2.4\sin\!\left(\tfrac{x+y}{620} - 3.5t\right) + 1.9\cos\!\left(\tfrac{x-y}{440} + 4.2t\right) + 1.2\sin\!\left(\tfrac{x}{280}+5.1t\right)\cos\!\left(\tfrac{y}{310}-4.6t\right) + 0.7\sin\!\left(\tfrac{x-y}{160}+6.8t\right)$$

The **current vector field** combines a directional storm-surge flow, multi-scale spatial turbulence, and the contribution of all 3 wandering vortex gyres scaled by the instantaneous current intensity $I_{\text{current}}$.

#### Moored Sensor Hydrodynamic Model (`MooredSensor`)

Each of the 35 hydrophones is modelled as a `MooredSensor` object with a physical anchor on the seabed, an elastic tether, and a fatigue accumulator:

```text
  MOORED ──── current > 1.3 m/s ────► STRAINED ──── speed ≥ break_threshold ────► SNAPPED
    ▲                                     │            OR strain_accum > 12.0         │
    └───────── current < 0.8 m/s ─────────┘                                          │
                                                                         Freely drifts with
                                                                         current field × 14×
                                                                         visual multiplier
```

| State | Visual | Tether | Effect |
|---|---|---|---|
| **MOORED** | Blue (idle) | Cyan dotted | Gentle sway around anchor ($\le 220\text{ m}$) |
| **STRAINED** | Orange | Orange solid, thick | Fatigue accumulating; tether at risk |
| **SNAPPED / DRIFTING** | Red blinking | Broken stub shown | Sensor driven away by currents; network link severed |

* **Break threshold**: Each sensor has a unique random snap threshold $\in [2.2, 3.4]\text{ m/s}$, producing staggered failure during escalating storm surges.
* **Drift trails**: Snapped sensors leave ghost breadcrumb trails (11 segments, fading orange → red) tracing their journey across the harbor.
* **Re-anchor**: The **⚓ Re-Anchor Sensors** button in the top bar restores all 35 sensors to their original moorings.

#### Monsoon Visual Effects (3D Viewport)

| Effect | Implementation | Trigger |
|---|---|---|
| **Dynamic wave surface** | 18×18 mesh redrawn every 2 frames with 4-harmonic elevation | Always in Monsoon |
| **Water current quivers** | 9-point arrow grid; red during squalls, orange during normal Monsoon | Always in Monsoon |
| **Turbulent vortex eyes** | 3 `✕` markers + pulsing dashed rings (orange/purple by spin) | Always in Monsoon |
| **Rain particles** | 180 `|` scatter points falling and drifting with storm current | Always in Monsoon |
| **⚡ Lightning flash** | Tkinter overlay label flickers on/off every 4 frames; figure background darkens | During squall bursts |
| **🚨 Network collapse banner** | Centred Tkinter overlay pulsates when ≥ 50% of sensors are snapped | When network broken |
| **Drift trails** | 11 fading line segments per drifting sensor | When sensor SNAPPED |

---

## 🌐 5. Network Arrangement: Fixed vs. Dynamic

```text
Z = 0 m   (Surface)     [Buoy Alpha]      [Buoy Beta]      [Buoy Gamma]
                             ▲                 ▲                ▲
                             │                 │                │
Z = -10 m (Mid-Water)    [AUV-Echo] ──────► [AUV-Alpha] ──────► [AUV-Bravo]  (Mobile Swarm)
                             ▲                 ▲                ▲
                             │                 │                │
Z = -50 m (Seabed)       [Sensor-1]       [Sensor-14]      [Sensor-28]       (Fixed Grid)
```

### 5.1 Fixed Infrastructure

* **35 Stationary Seabed & Mid-Water Hydrophones (`MooredSensor`)**:
  * Stratified into surface ($Z = -2\text{m}$), mid-water ($Z = \text{seabed}/2$), and seabed ($Z = \text{seabed} + 2\text{m}$, down to $-65\text{m}$).
  * Omnidirectional acoustic modems ($R_{\text{comm}} = 1500\text{m}$, sensing radius $R_{\text{detect}} = 2000\text{m}$).
  * Each sensor has physical anchor, elastic tether, and fatigue accumulator — progressing through **MOORED → STRAINED → SNAPPED** states under monsoon currents (see §4.5).
* **3 Surface Gateway Buoys**:
  * Positioned near the harbor entrance along $X = 150\text{m}$: `buoy_alpha` $(150, 2000, 0)$, `buoy_beta` $(150, 5000, 0)$, `buoy_gamma` $(150, 8000, 0)$.
  * Act as high-speed RF/Satellite gateway sinks connected to harbor traffic control and provide inductive recharge stations for AUVs.

### 5.2 Dynamic Infrastructure (AUV Swarm)

A swarm of 8 Autonomous Underwater Vehicles (`RobotNode`) in [auv.py](file:///p:/Projects/Harbor-Traffic-Control-System-/auv.py) provides adaptive mobile sensing and communication relays across three operational zones:

* **Outer Yard ($X > 5000\text{m}$)**: Screen inbound traffic and provide deep-water acoustic relay.
* **Channel ($1200\text{m} < X < 5000\text{m}$)**: Bridge communications between seabed sensors and surface gateways.
* **Inner Harbor ($200\text{m} < X < 1200\text{m}$)**: Coordinate approach traffic and harbor basin coverage.

**AUV Swarm Behaviors**:
* **Waypoint Patrol**: Autonomous 3D waypoint navigation within assigned sectors.
* **Violation Intercept**: Automatically switches from `PATROLLING` to `DISPATCHED` when a vessel in its zone commits an overspeed violation, shadowing the vessel for verification.
* **Battery State Machine**: Discharges battery during movement; when battery drops below $15\%$, autonomously navigates to the nearest buoy for recharge ($0.2\%/\text{frame}$).

### 5.3 Harbor Zone Architecture

```text
X = 0 m              X = 700 m        X = 5000 m                       X = 10000 m
|---- HARBOR --------|--- INNER ZONE---|---------------- OUTER YARD -----------------|
 (Docking Berths)     (Patrol, 6 kn)   (LOC Tripwire Wall)            (10 kn Limit)
 (Surface Buoys)      (Channel AUVs)   (Tripwire Barrier)             (Outer AUVs)
```

1. **Outer Yard Zone ($X: 5000\text{m} – 10000\text{m}$)**: Inbound fairway. Speed limit: **10.0 knots**.
2. **Line of Control (LOC) Boundary ($X = 5000\text{m}$)**: Virtual acoustic tripwire triggering identity verification and classification screening.
3. **Inner Patrol Zone ($X: 700\text{m} – 5000\text{m}$)**: Controlled navigation channel. Speed limit: **6.0 knots**.
4. **Harbor Docking Area ($X < 700\text{m}$)**: Dedicated docking slips categorized by vessel classification.

---

## 📡 6. Sensor Networking & Acoustic Communication

### 6.1 Acoustic Channel Modeling & Link Budget
* **Carrier Frequency ($f_{\text{comm}}$)**: $12.0\text{ kHz}$
* **Modem Transmit Source Level ($SL_{\text{comm}}$)**: $178.0\text{ dB re } 1\,\mu\text{Pa @ } 1\text{m}$
* **Communication Range ($R_{\text{comm}}$)**: $1500\text{ meters}$
* **Link SNR Threshold**: $10.0\text{ dB}$ (minimum for packet decoding)
* **Acoustic Bitrate**: $1000\text{ bps}$ ($1\text{ kbps}$)

```text
             Tx Node (Source Level = 178 dB)
                          │
                          ▼
              Transmission Loss: TL(d) = 20*log10(d) + alpha*d
                          │
                          ▼
              Rx Signal Level = 178 - TL(d)
                          │
                          ▼
              Ambient Noise Level (Scenario-Adaptive Wenz)
                          │
                          ▼
              Link SNR = Rx Signal Level - Noise Level
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
   SNR >= 10 dB                        SNR < 10 dB
  [Link Active (Graph Edge)]         [Link Severed (Out of Range / High Noise)]
```

### 6.2 Link Hierarchy & Active Connection Types

Every simulation frame, [network.py](file:///p:/Projects/Harbor-Traffic-Control-System-/network.py) constructs a dynamic spatial graph containing 3 active acoustic link types:
1. **Seabed Sensor $\leftrightarrow$ AUV Link**: Stationary hydrophones uplink detection packets to patrolling AUVs within range.
2. **AUV $\leftrightarrow$ AUV Link (Inter-Robot Mesh)**: Patrolling AUVs form an ad-hoc peer-to-peer relay backbone.
3. **AUV $\leftrightarrow$ Surface Buoy Gateway Link**: Surfacing AUVs offload aggregated telemetry packets directly to surface buoys.

*Note: Vessels radiate passive sound detected by sensors and AUVs; they do not carry interactive acoustic network transponders.*

### 6.3 Depth-Based Routing (DBR) Protocol

To route packets from seabed depths to surface gateways without GPS, the system implements **Depth-Based Routing (DBR)**:

1. **Packet Origin**: A detecting node (sensor or AUV) generates a telemetry packet.
2. **Gateway Check**: If any surface buoy is directly reachable, the packet hops directly to the nearest buoy.
3. **Depth Gradient Selection**: If no buoy is in range, the node queries active neighbors and filters for **shallower nodes** ($Z_{\text{neighbor}} > Z_{\text{current}}$, i.e., closer to depth $0\text{m}$).
4. **Greedy Forwarding**: The packet is forwarded to the neighbor with the highest depth coordinate.
5. **Loop Avoidance**: Visited nodes are recorded; paths exceeding 12 hops or encountering depth local minima are safely terminated.

### 6.4 Duty-Cycling & Swarm Resilience

To conserve battery, stationary seabed sensors operate on a **70% active / 30% sleep duty cycle** (active for 35 frames, sleeping for 15 frames).
* When a seabed sensor sleeps, nearby mobile AUVs detect the vessel directly and act as dynamic relay bridges, maintaining network connectivity across the harbor domain.

### 6.5 Frame Compression, Latency & Energy Metrics

The networking layer compares standard uncompressed telemetry payloads against optimized compressed binary telemetry:

| Metric | Default Uncompressed Frame | Compressed Telemetry Frame | Performance Improvement |
|---|---|---|---|
| **Frame Size** | $256\text{ bytes}$ ($2048\text{ bits}$) | $16\text{ bytes}$ ($128\text{ bits}$) | **$16\times$ Payload Reduction** |
| **Transmission Delay per Hop** | $\frac{2048\text{ b}}{1000\text{ bps}} = 2.048\text{ s}$ | $\frac{128\text{ b}}{1000\text{ bps}} = 0.128\text{ s}$ | **$93.75\%$ Faster Tx Delay** |
| **Energy Consumption per Hop** | $10\text{W} \times 2.048\text{s} = 20.48\text{ Joules}$ | $10\text{W} \times 0.128\text{s} = 1.28\text{ Joules}$ | **$93.75\%$ Energy Saved per Hop** |
| **End-to-End Latency (3 Hops)** | $\approx 6.5\text{ seconds}$ | $\approx 0.7\text{ seconds}$ | **Near Real-Time Telemetry** |

---

## 🚦 7. Maritime Traffic Management & Autonomous Control

The traffic management algorithm ([traffic_algo.py](file:///p:/Projects/Harbor-Traffic-Control-System-/traffic_algo.py)) governs harbor movements based on acoustic detection events and ML classifications:

### 1. Speed Governance & Violation Detection
* **Outer Yard ($X > 5000\text{m}$)**: Speed limit $= 10.0\text{ knots}$.
* **Inner Patrol Zone ($X: 700\text{m} - 5000\text{m}$)**: Speed limit $= 6.0\text{ knots}$.
* **Overspeed Alert**: If measured speed exceeds the limit, the vessel status flips to `OVERSPEED`, triggering dashboard alerts and dispatching the nearest AUV to shadow the vessel.

### 2. Line of Control (LOC) Screening
* Inbound vessels crossing the LOC ($X = 5000\text{m}$) are evaluated against predicted ship types from `VesselMLP`:
  * *Cargo Ship / Tanker*: Issued `HALT` (heavy vessels wait at LOC fairway).
  * *Cruiser / Ferry*: Issued `SLOW DOWN` (reduce speed to 50%).
  * *Speedboat / Fishing Vessel*: Issued `CONTINUE` (proceed through designated lanes).

### 3. Automated Dock Slot Allocation
Upon entering the inner approach ($X \le 700\text{m}$), vessels are routed to dedicated dock berths:
* `Dock-A`: Cargo Ships ($Y \approx 2000\text{m}$)
* `Dock-T1..T2`: Tankers ($Y \approx 2700 - 3000\text{m}$)
* `Dock-C`: Speedboats ($Y \approx 4125\text{m}$)
* `Dock-V1..V4`: Fishing Vessels ($Y \approx 5200 - 6100\text{m}$)
* `Dock-F`: Ferries ($Y \approx 6825\text{m}$)
* `Dock-B1..B4`: Cruisers ($Y \approx 7500 - 8450\text{m}$)

---

## 🏗️ 8. System Architecture & Data Pipeline

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        10 km × 10 km 3D HARBOR SIMULATION                              │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │ 22 Surface Ships      │  │ 35 Seabed Hydrophones  │  │ 8 Mobile Patrolling       │  │
│  │ (Cargo, Tanker, etc.) │  │ (3 Depth Layers, DBR)  │  │ AUVs (Swarm Kinematics)   │  │
│  └───────────┬───────────┘  └───────────┬────────────┘  └─────────────┬─────────────┘  │
└──────────────┼──────────────────────────┼─────────────────────────────┼────────────────┘
               │                          │                             │
               ▼                          ▼                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ACOUSTIC PHYSICS & TELEMETRY ENGINE                             │
│  • Selectable Scenarios: Normal Season (65 dB noise) vs. Monsoon Season (92 dB noise)   │
│  • Mackenzie Sound Speed (T, S, D)      • Ross Propeller Source Level (SL)             │
│  • Thorp Absorption & Transmission Loss • Doppler Radial Velocity Shift                │
│  • Wenz Ocean Ambient Noise + Rain      • 23-Feature Telemetry Extraction              │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       DEPTH-BASED ACOUSTIC ROUTING (DBR)                               │
│  • NetworkX Dynamic Adjacency Graph     • Duty Cycling Simulation (70/30)              │
│  • Multi-Hop Greedy Depth Forwarding    • Frame Compression (256B → 16B)               │
│  • Surface Buoy Gateway Sinks           • PDR, Energy & Latency Tracking               │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   MULTI-TASK DEEP NEURAL NETWORK (VesselMLP)                           │
│  • Shared Hidden Layers (128 → 64 → 32) • Task 1: 6-Class Ship Classifier (Cross-Entropy)│
│  • Pre-Trained Fast Offline Inference   • Task 2: Weight Regression in kg (MSE)        │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│     MARITIME TRAFFIC CONTROL SYSTEM       │ │       REAL-TIME MONITORING SUITE          │
│ • Speed Limit Enforcement (Outer/Inner)   │ │ • Native 3D Matplotlib GUI Loop           │
│ • LOC Virtual Curtain Screening           │ │ • Background Python HTTP Server (Port 8000)│
│ • Classification-Based Dock Allocation    │ │ • Harbor Web Dashboard (dashboard.html)   │
│ • AUV Shadowing of Speed Violators        │ │ • Network Topology (networking_dashboard) │
└───────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## 💻 9. Technology Stack & Selection Rationale

| Component | Technology | Rationale & Architectural Trade-offs |
|---|---|---|
| **Simulation Core** | **Python 3.9+** | Unifies deep learning (`torch`), graph theory (`networkx`), and scientific computing (`numpy`, `pandas`) into a single runtime without inter-language bridge overhead. |
| **Deep Learning** | **PyTorch (`VesselMLP`)** | Multi-task architecture sharing 3 hidden layers across classification and regression for high-speed offline inference. |
| **Acoustic Physics** | **Vectorized NumPy** | Pure Python/NumPy implementation calculating exact oceanographic formulas cross-platform without platform-specific compiled C++ physics engines. |
| **Network Topology** | **NetworkX** | Constructs spatial graphs frame-by-frame, evaluating acoustic SNR thresholds and executing Depth-Based Routing (DBR) across dynamic node coordinates. |
| **Interactive 3D GUI** | **Tkinter + Matplotlib 3D (`TkAgg`)** | Native 3D visualization with interactive 360° orbital camera rotation, zoom controls, and a startup environment selection overlay. |
| **Decoupled Telemetry UI** | **HTML5 / CSS3 / Vanilla JS + `http.server`** | Decouples browser visualization from the 3D animation loop. Simulation writes JSON snapshots every $500\text{ms}$, served asynchronously to avoid UI rendering lag. |

---

## 📁 10. Project Directory & Module Structure

```text
Harbor-Traffic-Control-System/
├── README.md                      # Comprehensive System Documentation
├── PROJECT CONTEXT.md             # Research context & multi-robot extension goals
├── Future_Updates.txt             # Planned feature roadmap
├── main.py                        # Central orchestrator, startup scenario selection & 3D Tkinter GUI
├── simulator.py                   # Acoustic physics engine, environmental scenarios & 10km harbor
├── mlmodel.py                     # Multi-task PyTorch neural network (VesselMLP) & inference pipeline
├── network.py                     # Depth-Based Routing (DBR), modem link budgets & packet compression
├── network_manager.py             # RobotNode, CommLink & NetworkManager classes (standalone)
├── auv.py                         # Standalone NumPy AUV kinematics, battery state machine & dispatch
├── traffic_algo.py                # Speed governance, LOC tripwire screening & dock berth routing
├── verify_environment.py          # Automated system diagnostics & test runner
├── dashboard.html                 # Real-time harbor traffic & vessel telemetry web dashboard
├── networking_dashboard.html      # Acoustic network topology & DBR routing web dashboard
├── dashboard_data.json            # Live exported JSON telemetry stream (runtime-generated)
├── dashboard_data.js              # JavaScript telemetry wrapper (runtime-generated)
├── collected_data.csv             # Continuous dataset of physical sensor detections (runtime-generated)
└── models/                        # Pre-trained ML model artifacts
    ├── model_state.pt             # PyTorch VesselMLP weights (multi-task: type + weight)
    ├── scaler.pkl                 # Scikit-learn StandardScaler for 23 acoustic features
    ├── label_encoder.pkl          # Scikit-learn LabelEncoder for 6 ship classes
    ├── model_type.pkl             # Legacy classifier artifact
    └── model_weight.pkl           # Legacy regressor artifact
```

---

## 🚀 11. Quickstart & Execution Guide

### 1. Installation & Environment Setup
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/rishikesh-2k6/Harbor-Traffic-Control-System-.git
cd Harbor-Traffic-Control-System-
pip install numpy pandas matplotlib scikit-learn joblib torch networkx
```

### 2. Run Automated Verification Suite
Verify dependency integrity, network classes, and telemetry serialization:

```bash
python verify_environment.py
```

### 3. Launch 3D Simulation & Telemetry Server
Execute the main application:

```bash
python main.py
```

1. **Select Environmental Scenario**:
   * A startup overlay appears presenting two choices:
     * `[ ▶ Select Normal Season ]` (Calm seas, low noise, standard ~1.5 km acoustic links)
     * `[ ▶ Select Monsoon Season ]` (Rough seas, torrential rain noise, storm currents, SNR degradation)
2. **Interactive 3D Simulation**:
   * Click and drag to orbit 360°, scroll wheel to zoom, click **⟲ Reset View** to restore default perspective.
   * Click **⇄ Switch Season** in the top bar to toggle between Normal and Monsoon on the fly.
   * Click **🌪️ Gale Squall Surge** to inject a violent non-deterministic current burst immediately (auto-switches to Monsoon).
   * Click **⚓ Re-Anchor Sensors** to restore all 35 sensors to their seabed moorings after a monsoon event.
3. **HUD Telemetry Bar** (below the top bar) shows live:
   * 🌊 Water level / peak wave height | 💨 Current speed (m/s) | ⚓ Sensor mooring status | 📡 Network PDR
3. **Auto-Started Telemetry Server**:
   * Serves live telemetry at `http://127.0.0.1:8000`.

### 4. Access Live Web Dashboards
Open either dashboard in your browser:
* **Harbor Traffic Dashboard**: `http://127.0.0.1:8000/dashboard.html`  
  *(Live ship tracking, speed violations, ML predictions, dock slot occupancy, active weather badge)*
* **Acoustic Network Dashboard**: `http://127.0.0.1:8000/networking_dashboard.html`  
  *(Dynamic topology graph, active acoustic links, DBR hop counts, AUV battery levels, PDR)*

---

## 📊 12. Live Telemetry Data Schema

The simulation continuously streams JSON telemetry snapshots to [dashboard_data.json](file:///p:/Projects/Harbor-Traffic-Control-System-/dashboard_data.json):

```json
{
  "frame": 135,
  "active_sensors": 26,
  "total_sensors": 35,
  "ml_status": "ML LIVE (CPU) — 131562 detections",
  "violations": [],
  "vessels": [
    {
      "id": 1,
      "type": "Cargo Ship",
      "pred_type": "Cargo Ship",
      "pred_weight": 142000,
      "state": "ROUTING",
      "cmd": "ROUTE→Dock-A",
      "zone": "inner_zone",
      "direction": "INBOUND",
      "overspeed": false,
      "halt_time": 0,
      "dock_slot": "Dock-A",
      "speed_knots": 9.8,
      "x": 4107.0,
      "y": 4560.4,
      "sensor_detected": true,
      "sensor_count": 5
    }
  ],
  "robots": [
    {
      "id": 1,
      "x": 7544.8,
      "y": 2923.2,
      "z": -10.8,
      "vx": -1.4,
      "vy": 2.05,
      "vz": 0.07,
      "battery_pct": 85.0,
      "status": "PATROLLING",
      "waypoint": [7141.7, 8492.4, -14.9],
      "current_zone": "Outer Yard"
    }
  ],
  "network_links": [
    {
      "source_id": "sensor_0",
      "target_id": "auv_4",
      "snr": 25.7,
      "delay": 0.5925,
      "payload_type": "BEACON"
    }
  ],
  "routed_detections": {
    "delivered": [14, 16],
    "paths": {
      "14": [["sensor_9", "auv_3", "buoy_beta"], ["sensor_19", "auv_6", "buoy_beta"]]
    }
  },
  "environment": {
    "scenario": "Monsoon Season",
    "scenario_key": "MONSOON",
    "badge": "⛈️ Monsoon Season",
    "description": "Severe sea state, gale-force winds, torrential rain noise, turbulent currents, acoustic SNR degradation.",
    "sea_state": "Douglas 6–8",
    "wind_speed_knots": "36.0–52.0 kn",
    "ambient_noise_db": 91.9,
    "current_mult": 2.6
  },
  "network_stats": {
    "scenario": "Monsoon Season",
    "scenario_key": "MONSOON",
    "ambient_noise_db": 91.9,
    "pdr": 92.4,
    "avg_hop_count": 1.8,
    "active_alerts": 0,
    "total_energy_saved_j": 172.8,
    "avg_latency_default_s": 4.919,
    "avg_latency_compressed_s": 1.463,
    "node_states": {
      "sensor_0": "ACTIVE",
      "sensor_4": "SLEEPING",
      "auv_1": "ACTIVE",
      "buoy_alpha": "ACTIVE"
    }
  }
}
```

---

## 🏆 13. Summary of Innovations

1. **Unified Environmental Scenarios on a Single Engine**: Compares Normal vs. Monsoon conditions on the exact same simulation core without duplicating code or triggering Tkinter window crashes.
2. **Non-Deterministic Monsoon Physics Engine**: Ornstein-Uhlenbeck stochastic current intensity, random gale squall burst injection, 4-harmonic chaotic wave surface, and 3 wandering turbulent vortex gyres — producing genuinely unpredictable storm conditions every run.
3. **Moored Sensor Hydrodynamic Failure Model**: Each hydrophone progresses through MOORED → STRAINED → SNAPPED states under current load, with per-sensor random break thresholds and cumulative fatigue tracking. Snapped sensors drift freely and leave fading ghost trails.
4. **Rich Monsoon Visualization Suite**: Rain particles, ⚡ lightning flash on squall events, vortex eye markers, sensor drift breadcrumb trails, and a 🚨 network-collapse banner — all updating live in the 3D Tkinter viewport.
5. **Acoustic Physics Accuracy**: Full implementation of Mackenzie, Thorp, Ross, Wenz, and Doppler equations without external heavy physics engines.
6. **Multi-Task Deep Learning**: Simultaneous ship classification and weight estimation via shared PyTorch latent representations with sub-millisecond inference.
7. **Adaptive Swarm Relay**: Autonomous AUVs dynamically bridging communication partitions caused by sensor sleep duty cycles and monsoon link severances.
8. **Depth-Based Acoustic Routing (DBR)**: Multi-hop greedy depth routing achieving high Packet Delivery Ratios under realistic oceanographic constraints.
9. **Decoupled Real-Time Visualization**: High-performance dual dashboard architecture serving live telemetry without degrading 3D simulation frame rates.
10. **Organised Project Structure**: ML model artifacts (`model_state.pt`, `scaler.pkl`, `label_encoder.pkl`) isolated in `models/` — separated from runtime-generated data and source code.
