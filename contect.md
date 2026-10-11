# ⚓ Harbor Traffic Control System: Comprehensive Project Context & Reference Guide

> **Note**: This file (`contect.md`) is provided as an alias to [`context.md`](file:///p:/Projects/Harbor-Traffic-Control-System-/context.md). Both files contain the full master documentation for project reports and presentations.

Please refer to the full master documentation at [context.md](file:///p:/Projects/Harbor-Traffic-Control-System-/context.md) or read the complete copy below:

---

# ⚓ Harbor Traffic Control System: Comprehensive Project Context & Reference Guide

> **Master Reference Document for Technical Reports, Academic Papers, and Presentations**  
> *Autonomous Maritime Traffic Governance, Multi-Robot Swarm Telemetry, and Acoustic Physics in Underwater Wireless Sensor Networks (UWSN)*

---

## 📑 Table of Contents

1. [Executive Summary & Elevator Pitch](#-1-executive-summary--elevator-pitch)
2. [Problem Statement & Motivation](#-2-problem-statement--motivation)
3. [Research Objectives & Novel Contributions](#-3-research-objectives--novel-contributions)
4. [Harbor Domain & Physical Environment](#-4-harbor-domain--physical-environment)
5. [Oceanographic Acoustic Physics Engine](#-5-oceanographic-acoustic-physics-engine)
6. [23-Feature Acoustic Telemetry Pipeline](#-6-23-feature-acoustic-telemetry-pipeline)
7. [Multi-Task Deep Learning Framework (`VesselMLP`)](#-7-multi-task-deep-learning-framework-vesselmlp)
8. [Underwater Wireless Sensor Networking & Routing Protocols](#-8-underwater-wireless-sensor-networking--routing-protocols)
9. [Autonomous Mobile AUV Swarm Kinematics & Behaviors](#-9-autonomous-mobile-auv-swarm-kinematics--behaviors)
10. [Autonomous Maritime Traffic Governance & Enforcement](#-10-autonomous-maritime-traffic-governance--enforcement)
11. [Selectable Environmental Regimes: Normal vs. Monsoon Season](#-11-selectable-environmental-regimes-normal-vs-monsoon-season)
12. [System Architecture & Software Engineering Stack](#-12-system-architecture--software-engineering-stack)
13. [Codebase Directory & Module Directory](#-13-codebase-directory--module-directory)
14. [Step-by-Step Execution & Live Demo Guide](#-14-step-by-step-execution--live-demo-guide)
15. [Slide-by-Slide Presentation Blueprint](#-15-slide-by-slide-presentation-blueprint)
16. [Formal Technical Report & Paper Structure](#-16-formal-technical-report--paper-structure)
17. [Quantitative Results & Evaluator Q&A Defense Cheat Sheet](#-17-quantitative-results--evaluator-qa-defense-cheat-sheet)

---

## 🧭 1. Executive Summary & Elevator Pitch

### What is this Project?
The **Harbor Traffic Control System** is an advanced, cyber-physical simulation and autonomous maritime management platform. It combines **underwater acoustic physics**, a **hybrid network of fixed seabed hydrophones and autonomous underwater vehicles (AUVs)**, a **multi-task PyTorch deep neural network**, and **depth-based multi-hop routing (DBR)** to detect, classify, track, and regulate vessel traffic across a massive **$10\text{ km} \times 10\text{ km}$ ($100\text{ km}^2$) coastal harbor basin**.

### The 30-Second Elevator Pitch (For Presentations)
> *"Traditional harbor traffic systems rely on surface radar and AIS, which are vulnerable to spoofing, visual blind spots, and zero subsurface awareness. Our project introduces a smart, autonomous cyber-physical system powered by underwater acoustics. Using 35 moored seabed sensors and a cooperating swarm of 8 mobile AUVs, we passively capture 23 acoustic signatures from vessel radiated sound. A multi-task PyTorch neural network classifies ship types into 6 categories and regresses their displacement weight with sub-millisecond speed. Multi-hop Depth-Based Routing (DBR) guarantees reliable packet delivery even under severe duty-cycling and extreme monsoon storm conditions, while automated traffic algorithms govern speed, enforce a virtual Line of Control (LOC), dispatch AUVs to shadow violators, and assign dock slips autonomously."*

---

## 🚨 2. Problem Statement & Motivation

### Real-World Deficiencies in Maritime Traffic Control
1. **Radar / AIS Limitations**: Surface Vessel Traffic Services (VTS) rely heavily on Automatic Identification Systems (AIS) and marine radar. Illicit vessels (smugglers, illegal fishing, hostile craft) routinely switch off or spoof transponders. Marine radar cannot penetrate beneath the water column and is susceptible to surface clutter during severe storms.
2. **Subsurface Blind Spots**: Harbors face growing underwater threats, submerged hazards, silent hull intrusion, and navigation risks that surface radar cannot detect.
3. **Severe Constraints of Underwater Acoustics**:
   - **Sound Speed**: Sound propagates at $\approx 1500\text{ m/s}$ in seawater—five orders of magnitude slower than electromagnetic waves ($3 \times 10^8\text{ m/s}$), leading to high propagation latency.
   - **Severe Attenuation**: Radio frequency (RF) and GPS signals cannot penetrate seawater beyond a few meters. Acoustic waves must be used, which suffer severe frequency-dependent absorption.
   - **Low Bandwidth**: Acoustic modems typically operate at mere $1\text{ kbps}$ transmission bitrates.
4. **Failure Modes of Static UWSNs**:
   - Traditional Underwater Wireless Sensor Networks (UWSNs) use static seabed sensors that quickly deplete their batteries, necessitating aggressive sleep/wake duty cycles.
   - A single anchor failure or high ambient noise surge (e.g., during monsoon squalls) creates permanent communication voids and severed data routes.

### The Solution: Passive-to-Active Collaborative Swarm Network
This project transforms a passive, vulnerable seabed sensor grid into an **active, adaptive multi-robot network**:
* **35 Moored Seabed Hydrophones** capture acoustic pressure waves passively.
* **8 Mobile Autonomous Underwater Vehicles (AUVs)** navigate dynamically across depth layers, acting as mobile data mules, acoustic relays, and patrol interceptors.
* **3 Surface Buoy Gateways** aggregate acoustic data and bridge to shore-based harbor traffic control via high-bandwidth RF/satellite links.

---

## 🎯 3. Research Objectives & Novel Contributions

| Objective | Innovation / Technical Implementation |
|---|---|
| **1. Grounded Acoustic Physics** | Pure NumPy implementation of Mackenzie 9-term sound speed, Ross propeller cavitation source level, Thorp acoustic absorption, Wenz ambient noise, and Doppler frequency shifts. |
| **2. Multi-Task Deep Learning** | Simultaneous vessel classification (6 classes) and continuous weight regression ($1\text{ kTon} - 400\text{ kTon}$) in a single PyTorch network sharing 3 latent layers. |
| **3. Mobile Swarm Adaptation** | AUV swarm kinematics navigating 3D waypoints, resisting depth-dependent hydrodynamic currents, managing battery drain/recharging, and shadowing violators. |
| **4. Resilient UWSN Routing** | Depth-Based Routing (DBR) forwarding packets up depth gradients to surface buoys despite a 70/30 sensor sleep duty cycle. |
| **5. Packet Compression Optimization** | Compression from 256 bytes to 16 bytes, cutting transmission delay and modem battery consumption by $93.75\%$. |
| **6. Unified Dual-Season Engine** | Single simulation core dynamically comparing Normal Season ($65.8\text{ dB}$ noise) against extreme Monsoon Season ($91.9\text{ dB}$ noise, stochastic squalls, vortex gyres, and sensor tether snap failures). |
| **7. Autonomous Traffic Governance** | Automated enforcement of zone speed limits, virtual Line of Control (LOC) ship screening, and automated dock slip allocation. |

---

## 🗺️ 4. Harbor Domain & Physical Environment

```text
X = 0 m              X = 700 m        X = 5000 m                       X = 10000 m
│◄──── HARBOR BASIN ───►│◄── INNER ZONE ──►│◄───────────── OUTER YARD ──────────────►│
│  • Dock Berths (A..B4) │  • 6.0 kn Limit  │  • 10.0 kn Limit                         │
│  • 3 Surface Buoys     │  • Channel AUVs  │  • Outer Fairway Screening              │
│    (Z = 0 m)           │  • Mid Hydrophones│  • Deep Seabed Hydrophones (Z = -65 m)   │
│                        │  • LOC Barrier   │  • Outer Patrol AUVs                    │
```

### 1. Spatial Dimensions & Coordinate System
* **Harbor Domain**: $10\,000\text{ m} \times 10\,000\text{ m}$ ($10\text{ km} \times 10\text{ km}$ area).
* **Depth Profile**: Seabed slopes smoothly from shallow harbor docks ($-20\text{ m}$ at $X = 0$) to deep oceanic waters ($-65\text{ m}$ at $X = 10\,000\text{ m}$).
* **Coordinate Conventions**:
  - $X \in [0, 10000]$: Longitudinal axis from docks ($X=0$) out to open sea ($X=10000$).
  - $Y \in [0, 10000]$: Latitudinal fairway spread.
  - $Z \in [-65, 0]$: Vertical depth axis ($Z=0$ is sea surface, negative values indicate depth).

### 2. Operational Zones & Traffic Regulations
1. **Harbor Docking Basin ($X < 700\text{ m}$)**:
   - Dedicated docking slips arranged by vessel type:
     - `Dock-A` ($Y \approx 2000\text{ m}$): Cargo Ships
     - `Dock-T1, T2` ($Y \approx 2700 - 3000\text{ m}$): Tankers
     - `Dock-C` ($Y \approx 4125\text{ m}$): Speedboats
     - `Dock-V1..V4` ($Y \approx 5200 - 6100\text{ m}$): Fishing Vessels
     - `Dock-F` ($Y \approx 6825\text{ m}$): Ferries
     - `Dock-B1..B4` ($Y \approx 7500 - 8450\text{ m}$): Cruisers
2. **Inner Patrol Zone ($700\text{ m} \le X < 5000\text{ m}$)**:
   - Speed limit: **$6.0\text{ knots}$**. High-density navigation channel.
3. **Line of Control (LOC) Acoustic Tripwire ($X = 5000\text{ m}$)**:
   - Virtual screening curtain. Inbound vessels undergo ML type verification:
     - Heavy vessels (*Cargo / Tanker*): Issued **HALT** command (must hold outside fairway).
     - Medium vessels (*Cruiser / Ferry*): Issued **SLOW DOWN** command (reduce speed by 50%).
     - Light vessels (*Speedboat / Fishing*): Issued **CONTINUE** command.
4. **Outer Yard Fairway ($5000\text{ m} \le X \le 10000\text{ m}$)**:
   - Speed limit: **$10.0\text{ knots}$**. Inbound approach sector.

### 3. Network Infrastructure Inventory
* **35 Moored Seabed Hydrophones**: Placed in seabed, mid-water, and sub-surface depths ($R_{\text{comm}} = 1800\text{ m}, R_{\text{detect}} = 2000\text{ m}$).
* **8 Autonomous Underwater Vehicles (AUVs)**: Mobile nodes dynamically patrolling Outer Yard, Channel, and Inner Harbor.
* **3 Surface Gateway Buoys**: `buoy_alpha` $(150, 2000, 0)$, `buoy_beta` $(150, 5000, 0)$, `buoy_gamma` $(150, 8000, 0)$ equipped with acoustic modems and satellite/RF surface uplinks.
* **22 Surface Vessels**: Inbound and outbound cargo ships, tankers, passenger cruisers, ferries, speedboats, and fishing vessels.

---

## 🌊 5. Oceanographic Acoustic Physics Engine

The simulation incorporates authentic physical oceanography equations implemented in vectorized NumPy:

### 1. Mackenzie (1981) Nine-Term Sound Speed Equation
The velocity of sound in seawater $c$ ($\text{m/s}$) is calculated dynamically as a function of temperature $T$ ($^\circ\text{C}$), salinity $S$ ($\text{ppt}$), and depth $D$ ($\text{m}$):
$$c(T, S, D) = 1448.96 + 4.591T - 5.304 \times 10^{-2}T^2 + 2.374 \times 10^{-4}T^3 + 1.340(S - 35) + 1.630 \times 10^{-2}D + 1.675 \times 10^{-7}D^2 - 1.025 \times 10^{-2}T(S - 35) - 7.139 \times 10^{-13}TD^3$$

### 2. Ross (1976) Propeller Source Level Equation
Ship radiated noise is primarily generated by propeller cavitation and machinery vibrations:
$$SL = SL_{\text{base}} + 55\log_{10}\left(\frac{v}{10}\right) + 20\log_{10}\left(\frac{d_{\text{prop}}}{3.0}\right) + \mathcal{N}(0, \sigma^2)$$
* $v$: Vessel speed in knots.
* $d_{\text{prop}}$: Propeller diameter in meters (e.g., $1.2\text{ m}$ for speedboats, $6.5\text{ m}$ for tankers).
* $SL_{\text{base}}$: Base acoustic radiated level ($135 - 170\text{ dB re } 1\,\mu\text{Pa @ } 1\text{m}$).

### 3. Thorp (1967) Acoustic Medium Absorption
Acoustic absorption coefficient $\alpha(f)$ ($\text{dB/km}$) for carrier/acoustic frequency $f$ ($\text{kHz}$):
$$\alpha(f) = \frac{0.11 f^2}{1 + f^2} + \frac{44 f^2}{4100 + f^2} + 2.75 \times 10^{-4}f^2 + 0.003$$

### 4. Acoustic Transmission Loss ($TL$)
Combines spherical geometric wavefront spreading with medium absorption over 3D slant range $R$ ($\text{m}$):
$$TL(R, f) = 20\log_{10}(R) + \alpha(f) \cdot \left(\frac{R}{1000}\right)$$

### 5. Wenz (1962) Ambient Ocean Noise with Torrential Rain Augmentation
Total ambient noise floor $NL_{\text{ambient}}$ ($\text{dB re } 1\,\mu\text{Pa}$) combines wind wave agitation, rain impact, shipping density, and thermal/biological noise:
$$NL_{\text{ambient}} = 10\log_{10}\left(10^{(NL_{\text{wind}} + NL_{\text{rain}})/10} + 10^{NL_{\text{shipping}}/10} + 10^{NL_{\text{bio}}/10}\right)$$
Where wind-induced noise is:
$$NL_{\text{wind}} = 44 + 20\log_{10}(v_{\text{wind\_knots}})$$

### 6. Acoustic Doppler Shift & Time of Arrival (ToA)
As a ship moves with velocity vector $\vec{v}_{\text{ship}}$, its radial velocity relative to the sensor is:
$$v_{\text{radial}} = \vec{v}_{\text{ship}} \cdot \frac{\vec{p}_{\text{sensor}} - \vec{p}_{\text{ship}}}{\|\vec{p}_{\text{sensor}} - \vec{p}_{\text{ship}}\|}$$
The received frequency is shifted according to:
$$f_{\text{received}} = f_{\text{source}} \cdot \left(\frac{c}{c - v_{\text{radial}}}\right), \quad \Delta f = f_{\text{received}} - f_{\text{source}}$$
Travel latency (Time of Arrival) over distance $R$ is:
$$\text{ToA} = \frac{R}{c(T, S, D)}$$

---

## 📊 6. 23-Feature Acoustic Telemetry Pipeline

When a hydrophone or AUV detects a vessel within its detection horizon ($R_{\text{detect}} = 2000\text{ m}$), it constructs a comprehensive 23-dimensional feature vector:

```text
                         [Vessel Radiated Sound]
                                    │
       ┌────────────────────┬───────┴────────────┬────────────────────┐
       ▼                    ▼                    ▼                    ▼
[Acoustic Energy]   [Wave Propagation]   [Environmental]     [Engineered Cross]
(7 Decibel Metrics) (ToA, Speed, Azimuth)(T, S, Depth, Wind)  (Nonlinear Ratios)
       │                    │                    │                    │
       └────────────────────┴───────┬────────────┴────────────────────┘
                                    ▼
                         23-Dimensional Vector
                                    │
                                    ▼
                       StandardScaler Normalization
                                    │
                                    ▼
                      Multi-Task PyTorch VesselMLP
```

| Category | Count | Feature Names | Physical & Operational Significance |
|---|:---:|---|---|
| **Acoustic Energy** | 7 | `Received_Level_dB`, `Source_Level_dB`, `Transmission_Loss_dB`, `Frequency_Hz`, `Doppler_Shift_Hz`, `SNR_dB`, `Ambient_Noise_dB` | Captures signal decibel strength, signal-to-noise margin, and cavitation pitch. |
| **Wave Propagation** | 3 | `Sound_Speed_mps`, `Time_of_Arrival_s`, `Bearing_deg` | Estimates sound travel time, physical water velocity, and vessel azimuth heading. |
| **Environmental Context**| 5 | `Water_Temp_C`, `Salinity_ppt`, `Sensor_Depth_m`, `Sea_State`, `Wind_Speed_knots` | Calibrates ML inference against local water column characteristics and sea states. |
| **Engineered Features** | 6 | `Acoustic_Energy`, `SNR_Freq_interaction`, `Received_Level_squared`, `Doppler_Speed_proxy`, `Transmission_Efficiency`, `Env_Composite` | Linear-domain power ($10^{RL/10}$), Doppler/frequency ratios, and geometric loss efficiency. |

---

## 🧠 7. Multi-Task Deep Learning Framework (`VesselMLP`)

### Architecture
Rather than deploying two separate neural networks (which would double inference latency and memory footprint), the system employs a unified multi-task deep neural network implemented in **PyTorch**:

```text
                      Input Features: 23 Dimensions
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
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
    [Classification Head]                       [Regression Head]
      Linear(32 → 6)                              Linear(32 → 1)
                │                                       │
                ▼                                       ▼
       Softmax Probabilities                   Displacement Weight (kg)
  (Cargo, Tanker, Cruiser, etc.)               (Range: 1kTon – 400kTon)
```

### Key Network Characteristics
* **Shared Representation**: 3 fully-connected hidden layers ($128 \rightarrow 64 \rightarrow 32$) extract generalizable acoustic latent embeddings representing hull and engine dynamics.
* **Task 1 — Classification Head**: Predicts probability distribution across 6 classes:
  1. *Cargo Ship* (Heavy displacement, slow, low frequency)
  2. *Tanker* (Massive displacement, steady propulsion)
  3. *Cruiser* (Moderate-to-high displacement, multi-generator sound)
  4. *Ferry* (Medium displacement, cyclical schedules)
  5. *Speedboat* (Low displacement, ultra-high speed, intense high-pitch Doppler shift)
  6. *Fishing Vessel* (Low displacement, irregular low-speed trolling)
  - Loss Function: **Cross-Entropy Loss**.
* **Task 2 — Weight Regression Head**: Predicts vessel displacement mass in kilograms ($\text{kg}$).
  - Loss Function: **Mean Squared Error (MSE) Loss**.
* **Real-Time Inference Pipeline**:
  - Automatically loads pre-trained artifacts from `models/` (`model_state.pt`, `scaler.pkl`, `label_encoder.pkl`).
  - Pre-trained weights allow **sub-millisecond (< 1 ms)** inference per detection without blocking the 3D rendering loop.
  - Safe CSV handling using retry loops with exponential backoff prevents Windows disk-locking race conditions.

---

## 📡 8. Underwater Wireless Sensor Networking & Routing Protocols

### 1. Acoustic Modem Channel Parameters
* **Carrier Frequency ($f_{\text{comm}}$)**: $12.0\text{ kHz}$
* **Source Transmit Level ($SL_{\text{comm}}$)**: $178.0\text{ dB re } 1\,\mu\text{Pa @ } 1\text{m}$
* **Nominal Acoustic Range ($R_{\text{comm}}$)**: $1800\text{ meters}$ ($1.8\text{ km}$)
* **Modem Decoding SNR Threshold**: **$\ge 10.0\text{ dB}$** (links dropping below $10\text{ dB}$ disconnect)
* **Nominal Bitrate**: $1000\text{ bps}$ ($1\text{ kbps}$)
* **Transmit Power**: $10.0\text{ W}$

### 2. 3-Tier Active Link Hierarchy
Every frame, `network.py` dynamically evaluates distances and ambient noise to generate graph edges across 3 link types:
1. **Seabed Sensor $\leftrightarrow$ AUV Link**: Stationary hydrophones uplink detection packets to mobile AUVs patrolling above.
2. **AUV $\leftrightarrow$ AUV Link (Inter-Robot Swarm Mesh)**: Mobile AUVs form an ad-hoc multi-robot mesh relay backbone.
3. **AUV $\leftrightarrow$ Surface Buoy Link**: AUVs surfacing near $X = 150\text{ m}$ offload aggregated data to buoys.

```text
Z = 0 m   (Surface)         [Buoy Alpha]      [Buoy Beta]      [Buoy Gamma]
                                 ▲                 ▲                ▲
                                 │                 │                │
Z = -10 m (Mid-Water)       [AUV-Echo] ──────► [AUV-Alpha] ─────► [AUV-Bravo] (Mobile Swarm)
                                 ▲                 ▲                ▲
                                 │                 │                │
Z = -50 m (Seabed)          [Sensor-03]       [Sensor-14]      [Sensor-28]    (Fixed Grid)
```

### 3. Depth-Based Routing (DBR) Protocol
Under water, GPS signals cannot penetrate, so nodes cannot rely on geographic coordinates. DBR routes data using only depth (pressure sensor) measurements:
1. **Packet Generation**: When a node detects a vessel, it originates a telemetry packet.
2. **Direct Buoy Check**: If a surface buoy ($Z = 0\text{ m}$) is in direct acoustic range ($\text{SNR} \ge 10\text{ dB}$), the packet hops immediately to the buoy.
3. **Greedy Depth Forwarding**: If no buoy is directly reachable, the node queries neighboring nodes with active links and filters for **shallower nodes** ($Z_{\text{neighbor}} > Z_{\text{current}}$).
4. **Best Hop Selection**: The packet is forwarded to the neighbor closest to the surface.
5. **Loop Prevention & Termination**: Packet headers maintain a visited-node bloom set. If a hop count exceeds 12 or a local depth minimum is reached, fallback routing is engaged.

### 4. Energy Conservation via Frame Compression
Because acoustic transmission consumes high electrical power ($10\text{ W}$) at slow bitrates ($1\text{ kbps}$), reducing payload size yields dramatic improvements:

| Metric | Uncompressed Telemetry Frame | Compressed Binary Frame | Gain / Improvement |
|---|:---:|:---:|:---:|
| **Frame Size** | $256\text{ bytes}$ ($2048\text{ bits}$) | $16\text{ bytes}$ ($128\text{ bits}$) | **$16\times$ Data Reduction** |
| **Transmission Delay / Hop** | $\frac{2048\text{ b}}{1000\text{ bps}} = \mathbf{2.048\text{ s}}$ | $\frac{128\text{ b}}{1000\text{ bps}} = \mathbf{0.128\text{ s}}$ | **$93.75\%$ Faster Transmission** |
| **Modem Energy / Hop** | $10\text{W} \times 2.048\text{s} = \mathbf{20.48\text{ Joules}}$ | $10\text{W} \times 0.128\text{s} = \mathbf{1.28\text{ Joules}}$ | **$93.75\%$ Energy Saved** |
| **3-Hop End-to-End Latency** | $\approx 6.5\text{ seconds}$ | $\approx 0.7\text{ seconds}$ | **Sub-Second Delivery** |

### 5. Duty-Cycling & Swarm Resilience
Stationary seabed sensors adhere to a **70% active / 30% sleep duty cycle** (35 frames active, 15 frames sleeping) to prolong deployment lifespan.
* When a seabed sensor sleeps, nearby mobile AUVs automatically bridge the gap, detecting vessels directly and forwarding packets as dynamic data mules.

---

## 🤖 9. Autonomous Mobile AUV Swarm Kinematics & Behaviors

The system deploys 8 Autonomous Underwater Vehicles (`RobotNode` in `auv.py`):

```text
               ┌─────────────────────────────────────────┐
               │         PATROLLING (Normal State)       │
               │  • 3D Waypoint Navigation               │
               │  • Sector Bounds Keeping                │
               │  • Drift Compensation                   │
               └───────────┬─────────────────┬───────────┘
                           │                 │
             Vessel Overspeed                 Battery < 15%
             Violation Trigger                Depletion
                           │                 │
                           ▼                 ▼
         ┌───────────────────────┐     ┌───────────────────────┐
         │      DISPATCHED       │     │       CHARGING        │
         │  • Intercept & Shadow │     │  • Return to Buoy     │
         │  • Video/Acoustic Log │     │  • Inductive Recharge │
         │  • Resume Patrol      │     │  • Resume at > 95%    │
         └───────────────────────┘     └───────────────────────┘
```

### 1. Kinematic Motion Model
AUVs are modeled with mass $M = 500\text{ kg}$, maximum propulsion thrust $F_{\text{max}} = 900\text{ N}$, drag coefficient $C_d = 0.85$, and top speed $v_{\text{max}} = 10\text{ m/s}$. Motion integrates depth-dependent current drift:
$$\vec{F}_{\text{total}} = \vec{F}_{\text{propulsion}} - C_d \|\vec{v}\|\vec{v} + \vec{F}_{\text{current}}$$
$$\vec{a} = \frac{\vec{F}_{\text{total}}}{M}, \quad \vec{v}(t + \Delta t) = \vec{v}(t) + \vec{a}\Delta t, \quad \vec{p}(t + \Delta t) = \vec{p}(t) + \vec{v}\Delta t$$

### 2. Autonomous Behavioral State Machine
1. **`PATROLLING`**: The AUV navigates autonomously between randomized 3D waypoints inside its designated sector (*Outer Yard*, *Channel*, or *Inner Harbor*).
2. **`DISPATCHED`**: When `traffic_algo.py` detects an overspeed violation in an AUV's sector, it assigns the violator as a target. The AUV accelerates to intercept, shadows the vessel for 60 frames, and verifies tracking.
3. **`CHARGING`**: Battery drains continuously during operation ($0.015\%/\text{frame}$). When battery drops below $15\%$, the AUV halts patrol, surfaces, and docks at the nearest surface buoy for inductive charging ($0.2\%/\text{frame}$), returning to service at $>95\%$.

---

## 🚦 10. Autonomous Maritime Traffic Governance & Enforcement

The traffic controller (`traffic_algo.py`) executes rules based on telemetry and ML predictions:

### 1. Speed Enforcement
* **Outer Yard ($X > 5000\text{ m}$)**: Limit $= 10.0\text{ knots}$.
* **Inner Zone ($700\text{ m} \le X \le 5000\text{ m}$)**: Limit $= 6.0\text{ knots}$.
* **Violation Action**: If measured speed exceeds the threshold, the vessel is flagged as `OVERSPEED`. An alert is pushed to the HUD, and the nearest idle AUV is dispatched to shadow the ship.

### 2. Line of Control (LOC) Screening at $X = 5000\text{ m}$
```text
Inbound Vessel Crosses LOC (X = 5000 m)
                 │
                 ▼
       Query VesselMLP Model
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
[Cargo/Tanker] [Cruiser/Ferry] [Speedboat/Fishing]
    │            │            │
    ▼            ▼            ▼
  HALT       SLOW DOWN    CONTINUE
 (Anchor at   (50% Speed   (Proceed to
   Fairway)    Reduction)    Inner Zone)
```

### 3. Automated Classification-Based Dock Berth Assignment
Upon entering $X \le 700\text{ m}$, vessels are assigned to specific dock coordinates:
* `Dock-A`: Cargo Ships ($Y = 2000\text{ m}$)
* `Dock-T1 / T2`: Tankers ($Y = 2700\text{ m}, 3000\text{ m}$)
* `Dock-C`: Speedboats ($Y = 4125\text{ m}$)
* `Dock-V1..V4`: Fishing Vessels ($Y = 5200\text{ m} - 6100\text{ m}$)
* `Dock-F`: Ferries ($Y = 6825\text{ m}$)
* `Dock-B1..B4`: Cruisers ($Y = 7500\text{ m} - 8450\text{ m}$)

---

## ⛈️ 11. Selectable Environmental Regimes: Normal vs. Monsoon Season

The platform implements **ONE unified simulation engine** with dynamically selectable scenarios:

```text
                     Harbor Traffic Control Core Engine
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
           ☀️ NORMAL SEASON                      ⛈️ MONSOON SEASON
        • Ambient Noise: ~65.8 dB              • Ambient Noise: ~91.9 dB (+26 dB)
        • Calm Seas (State 2–3)                • Storm Seas (State 6–8)
        • PDR > 95%                            • Modem SNR Drops ~26 dB
        • Direct Single/Dual Hops              • Forced Multi-Hop DBR
        • Intact Moored Sensors                • Tether Strain & Sensor Snaps
```

### Comparative Parameters Table

| Parameter | Normal Season | Monsoon Season | Physical / Operational Impact |
|---|:---:|:---:|---|
| **Douglas Sea State** | **State 2–3** | **State 6–8** | Significant wave height rises from $0.5\text{ m}$ to $4.0 - 9.0\text{ m}$. |
| **Wind Speed** | $8.0 - 16.0\text{ kn}$ | $36.0 - 52.0\text{ kn}$ | Gale-force winds drive surface turbulence and wave spray. |
| **Water Temperature** | $18.0 - 22.0^\circ\text{C}$ | $12.0 - 15.0^\circ\text{C}$ | Coastal cooling due to heavy cloud cover and runoff. |
| **Water Salinity** | $34.5 - 35.5\text{ ppt}$ | $30.0 - 32.5\text{ ppt}$ | Surface dilution from torrential downpours. |
| **Rain Noise Factor** | $0.0\text{ dB}$ | $\mathbf{+15.0\text{ dB}}$ | Continuous high-frequency droplet impact noise. |
| **Ambient Ocean Noise**| $\mathbf{\approx 65.8\text{ dB}}$ | $\mathbf{\approx 91.9\text{ dB}}$ | **$+26.1\text{ dB}$ noise surge** masks weak acoustic transmissions. |
| **Acoustic Modem SNR** | $\approx 50.5\text{ dB}$ (@ 1km) | $\approx 24.5\text{ dB}$ (@ 1km) | Long-range peripheral links collapse below $10\text{ dB}$ threshold. |
| **Hydrodynamic Current**| $1.0\times$ ($\le 0.35\text{ m/s}$) | $2.6\times$ ($1.8 - 6.8\text{ m/s}$) | Severe hull drag on AUVs; rapid battery drain. |
| **Sensor Mooring State**| **MOORED** (stable) | **STRAINED $\rightarrow$ SNAPPED** | Tethers snap under excessive drag; sensors drift freely. |

### Stochastic Monsoon Phenomena
1. **Ornstein-Uhlenbeck Current Turbulence**: Continuous Gaussian drift-diffusion process updating current intensity frame-by-frame.
2. **Poisson Gale Squall Bursts**: Spontaneous squall surges ($+1.8 - 3.4\text{ m/s}$) triggering visual ⚡ lightning flashes.
3. **Wandering Vortex Gyres**: 3 chaotic turbulent eddies with $800\text{ m}$ radii that drift through the harbor.
4. **Moored Sensor Hydrodynamic Failure**: Hydrophones accumulate strain; when current exceeds their randomized break threshold ($2.2 - 3.4\text{ m/s}$), tethers snap. The hydrophone drifts, severed from the network, leaving fading breadcrumb trails.

---

## 🏗️ 12. System Architecture & Software Engineering Stack

### End-to-End Cyber-Physical Architecture Diagram

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        10 km × 10 km 3D HARBOR SIMULATION                              │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │ 22 Surface Ships      │  │ 35 Seabed Hydrophones  │  │ 8 Mobile Patrolling       │  │
│  │ (Cargo, Tanker, etc.) │  │ (MooredSensor State)   │  │ AUVs (Swarm Kinematics)   │  │
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

### Technology Selection Rationale

| Layer | Selected Tech | Rationale & Architectural Advantage |
|---|---|---|
| **Language Runtime** | **Python 3.9+** | Unifies scientific modeling (`numpy`), deep learning (`torch`), graph theory (`networkx`), and desktop GUI (`tkinter`) in a single runtime without cross-language binding overhead. |
| **Physics Computation** | **NumPy (Vectorized)** | Eliminates external compiled C++ dependencies (e.g. PyBullet), providing high-speed cross-platform performance on Windows, Linux, and macOS. |
| **Deep Learning** | **PyTorch (`torch.nn`)** | Flexible multi-task network with shared latent representations, GPU/CPU auto-fallback, and sub-millisecond inference. |
| **Network Analysis** | **NetworkX** | Constructs spatial 3D graphs frame-by-frame, calculates Euclidean distances, checks SNR criteria, and executes Depth-Based Routing (DBR). |
| **Desktop Simulation GUI** | **Tkinter + Matplotlib 3D (`TkAgg`)** | Native interactive 360° orbital viewing, zoom controls, scenario toggle menus, and real-time HUD metrics without requiring heavy game engines. |
| **Live Telemetry Web UI** | **HTML5 Canvas / CSS3 / Vanilla JS** | Fully decoupled architecture: simulation writes `dashboard_data.json` every $500\text{ ms}$; browser polls via an embedded HTTP server (Port 8000) with zero rendering drag on the 3D physics loop. |

---

## 📁 13. Codebase Directory & Module Directory

```text
Harbor-Traffic-Control-System/
├── context.md                     # Master Technical Reference & Presentation Guide
├── contect.md                     # Alias of context.md
├── PROJECT CONTEXT.md             # High-level research context & multi-robot extension goals
├── README.md                      # Detailed system overview & markdown documentation
├── Future_Updates.txt             # Development roadmap & phase breakdown
├── main.py                        # Central orchestrator, scenario menu, Tkinter 3D GUI loop & server
├── simulator.py                   # Acoustic physics engine, Wenz noise, Mackenzie speed & 10km harbor
├── mlmodel.py                     # Multi-task PyTorch VesselMLP, training routine & live inference
├── network.py                     # Depth-Based Routing (DBR), modem SNR link budget & packet metrics
├── network_manager.py             # RobotNode, CommLink & NetworkManager modular classes
├── auv.py                         # Standalone NumPy AUV kinematics, battery state machine & dispatch
├── traffic_algo.py                # Speed governance, LOC tripwire rules & dock slip allocation
├── verify_environment.py          # Automated pre-flight system diagnostics & test runner
├── dashboard.html                 # Real-time harbor traffic & vessel telemetry web dashboard
├── networking_dashboard.html      # Acoustic network topology & DBR routing web dashboard
├── dashboard_data.json            # Runtime-generated JSON telemetry stream
├── dashboard_data.js              # Runtime-generated JS wrapper for file-protocol viewing
├── collected_data.csv             # Runtime-logged sensor acoustic telemetry dataset
└── models/                        # Pre-trained ML model artifacts
    ├── model_state.pt             # PyTorch VesselMLP neural network weights
    ├── scaler.pkl                 # Scikit-learn StandardScaler for 23 acoustic features
    └── label_encoder.pkl          # Scikit-learn LabelEncoder for 6 vessel classes
```

---

## 🚀 14. Step-by-Step Execution & Live Demo Guide

### Step 1: Pre-Flight Environment Verification
Before demonstrating or recording, execute the automated verification suite:
```bash
python verify_environment.py
```
*Expected Output*:
* `[OK]` Python 3.9+ and dependencies (`numpy`, `pandas`, `matplotlib`, `sklearn`, `joblib`, `torch`).
* `[OK]` `RobotNode`, `CommLink`, `NetworkManager` class instantiations.
* `[OK]` JSON telemetry serialization verified.

### Step 2: Launch the 3D Cyber-Physical Simulation
Launch the central simulation orchestrator:
```bash
python main.py
```
* The 3D viewport opens immediately with **Normal Season** active by default.
* The embedded HTTP server automatically binds to `http://127.0.0.1:8000`.

### Step 3: Open the Dual Real-Time Web Dashboards
Open the following links in any modern web browser:
1. **Harbor Traffic & Fleet Dashboard**:
   ```text
   http://127.0.0.1:8000/dashboard.html
   ```
   *Displays live vessel coordinates, predicted ship types, displacement weights, overspeed alerts, and assigned dock berths.*
2. **Acoustic Network & Swarm Telemetry Dashboard**:
   ```text
   http://127.0.0.1:8000/networking_dashboard.html
   ```
   *Displays active acoustic communication links, dynamic topology graph, DBR hop counts, AUV battery levels, and energy savings.*

### Step 4: Recommended Presentation Demo Sequence
1. **Normal Operations**:
   - Point out the 35 blue hydrophones, 8 patrolling AUVs (marked in yellow/green), and surface vessels navigating the fairway.
   - Show the HUD bar: Water level normal, current speed low ($\sim 0.3\text{ m/s}$), ambient noise low ($65.8\text{ dB}$), PDR $> 95\%$.
2. **Vessel Detection & ML Inference**:
   - Highlight an incoming vessel crossing into the detection perimeter ($2\text{ km}$).
   - Show how the 23 acoustic features are extracted and fed into `VesselMLP`, instantly displaying the predicted ship class and displacement tonnage.
3. **Traffic Enforcement**:
   - Observe an overspeeding ship (highlighted in red).
   - Point out the nearest AUV switching to `DISPATCHED` mode and maneuvering to shadow the violator.
   - Show the LOC line at $X = 5000\text{ m}$ issuing `HALT` to cargo ships and tankers.
4. **Switch to Monsoon Season**:
   - Press **`Ctrl+S`** or select **Season $\rightarrow$ Monsoon Season** from the menu.
   - Highlight the dramatic environmental transformation:
     - 4-harmonic turbulent wave surface appears.
     - Ambient noise jumps by $+26.1\text{ dB}$ (to $91.9\text{ dB}$).
     - Modem links degrade; peripheral links drop; DBR engages multi-hop relay routes through mobile AUVs.
5. **Trigger Gale Squall Surge**:
   - Click the **🌪️ Gale Squall Surge** button in the top bar.
   - Observe the ⚡ lightning flash effect, extreme current surge ($> 4.5\text{ m/s}$), and strained hydrophone tethers turning orange.
   - Watch hydrophone tethers snap under current fatigue, transitioning to drifting red markers with ghost trails.
6. **Recovery**:
   - Click **⚓ Re-Anchor Sensors** to restore all sensors to seabed moorings.

---

## 📽️ 15. Slide-by-Slide Presentation Blueprint

Use this ready-to-present outline for a 10–12 slide presentation deck:

### Slide 1: Title & Team
* **Title**: Harbor Traffic Control System: Adaptive Multi-Robot Networking & Acoustic Telemetry in UWSN
* **Subtitle**: Autonomous Cyber-Physical Maritime Governance with Deep Learning & Swarm Robotics
* **Presenter Names / Roles**: [List team members and contributions]
* **Visual**: High-resolution screenshot of the 3D harbor simulation with AUVs and vessel tracks.

### Slide 2: The Maritime Challenge (Problem Statement)
* **Key Points**:
  - Radar and AIS cannot see underwater; AIS is prone to spoofing and tampering.
  - Radio signals do not penetrate seawater; acoustic signals are slow ($1500\text{ m/s}$) and suffer heavy attenuation.
  - Fixed seabed sensors suffer from battery depletion, sleep cycles, and single-point-of-failure dead zones.
* **Talking Point**: *"We need an intelligent, active subsurface network that combines fixed sensors with autonomous mobile robots to monitor harbors reliably."*

### Slide 3: System Overview & Architecture
* **Key Points**:
  - Scale: $10\text{ km} \times 10\text{ km}$ coastal harbor domain.
  - Three-tier infrastructure: 35 Moored Hydrophones, 8 Mobile AUVs, 3 Surface Buoys.
  - Four operational zones: Outer Yard (10 kn), LOC Tripwire (Screening), Inner Zone (6 kn), Harbor Docks.
* **Visual**: Diagram showing the 3-tier depth architecture (Surface Buoys $\rightarrow$ AUV Swarm $\rightarrow$ Seabed Grid).

### Slide 4: Oceanographic Acoustic Physics
* **Key Points**:
  - Real-world oceanographic equations modeled in vectorized NumPy.
  - Mackenzie 9-term equation for sound speed profiles ($c(T, S, D)$).
  - Ross propeller cavitation formula for vessel source level ($SL$).
  - Thorp absorption & transmission loss; Wenz ambient noise model.
  - Doppler frequency shifts and Time-of-Arrival (ToA) tracking.
* **Visual**: Graph or equations of sound speed vs. depth and acoustic transmission loss.

### Slide 5: The 23-Feature Acoustic Telemetry Pipeline
* **Key Points**:
  - Purely passive acoustic listening—vessels do not need transponders.
  - 23 extracted features: 7 Acoustic Energy, 3 Wave Propagation, 5 Environmental Context, 6 Engineered Non-linear Cross-Terms.
  - Standardized and scaled for real-time deep learning inference.
* **Visual**: Table or feature pipeline graphic showing the 4 feature categories.

### Slide 6: Multi-Task Deep Neural Network (`VesselMLP`)
* **Key Points**:
  - Single PyTorch network with 3 shared hidden layers ($128 \rightarrow 64 \rightarrow 32$).
  - Dual heads: Task 1 classifies 6 vessel types; Task 2 predicts continuous displacement weight in kg.
  - Offline pre-trained weights allow sub-millisecond (< 1 ms) real-time inference.
* **Visual**: Neural network architecture diagram showing shared layers and dual output heads.

### Slide 7: Underwater Sensor Networking & Depth-Based Routing (DBR)
* **Key Points**:
  - Dynamic 3D network topology constructed frame-by-frame with NetworkX.
  - Depth-Based Routing (DBR) greedily forwards packets upward to surface gateway buoys without GPS.
  - Duty-cycling: 70% active / 30% sleep cycle. Patrolling AUVs bridge communication voids as dynamic data mules.
* **Visual**: Multi-hop routing diagram showing packet hops from seabed hydrophones through AUVs to surface buoys.

### Slide 8: Packet Compression & Energy Optimization
* **Key Points**:
  - Default uncompressed telemetry frame: $256\text{ bytes}$ ($2048\text{ bits}$).
  - Compressed binary telemetry frame: $16\text{ bytes}$ ($128\text{ bits}$).
  - Results: **$16\times$ data reduction**, transmission latency drops from $2.05\text{ s}$ to $0.13\text{ s}$ ($93.75\%$ reduction), and modem energy consumption cut by **$93.75\%$**.
* **Visual**: Bar chart comparing $256\text{ B}$ vs. $16\text{ B}$ latency and energy consumption.

### Slide 9: Autonomous AUV Swarm Kinematics & Dispatch
* **Key Points**:
  - 8 AUVs navigating 3D waypoints with full hydrodynamic drag and current drift modeling.
  - Autonomous battery management: returns to buoys at $<15\%$ battery for inductive recharge ($0.2\%/\text{frame}$).
  - Active intercept: automatically dispatched to shadow overspeed violators.
* **Visual**: AUV state machine diagram (Patrolling $\rightarrow$ Dispatched $\rightarrow$ Charging).

### Slide 10: Unified Environmental Scenarios (Normal vs. Monsoon)
* **Key Points**:
  - Single simulation engine supports on-the-fly toggling between Normal and Monsoon seasons.
  - Monsoon impact: Ambient noise surges $+26.1\text{ dB}$, dropping modem SNR and stressing DBR multi-hop routing.
  - Stochastic physics: Ornstein-Uhlenbeck current turbulence, wandering vortex gyres, Poisson squall bursts, and tether fatigue failures.
* **Visual**: Side-by-side comparison table of Normal vs. Monsoon parameters.

### Slide 11: Real-Time Dual Dashboards & Demonstration
* **Key Points**:
  - Decoupled asynchronous architecture: simulation writes JSON snapshots without frame rate lag.
  - Harbor Traffic Dashboard (`dashboard.html`): live vessel positions, predicted classes, speed alerts, dock slips.
  - Acoustic Network Dashboard (`networking_dashboard.html`): active links, hop counts, AUV battery levels, PDR.
* **Visual**: Screenshots of both web dashboards side by side.

### Slide 12: Conclusion & Future Roadmap
* **Key Points**:
  - Successfully bridged passive sensing and active swarm robotics in a $100\text{ km}^2$ coastal domain.
  - Highly resilient multi-hop acoustic networking under extreme oceanic conditions.
  - Future Work: Hardware-in-the-loop acoustic modem testing, multi-agent reinforcement learning (MARL) for AUV patrolling, and multi-vessel collision avoidance (COLREGs).
* **Closing Statement**: Thank you! Open for questions.

---

## 📝 16. Formal Technical Report & Paper Structure

Use this outline to structure a comprehensive academic project report or paper:

```text
1. INTRODUCTION
   1.1 Background: Maritime Traffic Control & VTS
   1.2 Limitations of Terrestrial Radar and AIS
   1.3 Underwater Wireless Sensor Networks (UWSN) Challenges
   1.4 Proposed Cyber-Physical Hybrid Swarm Solution

2. HARBOR DOMAIN & HYDROGRAPHIC MODELING
   2.1 10 km × 10 km Coastal Domain & Depth Bathymetry
   2.2 Operational Zonal Architecture (Outer Yard, LOC, Inner Zone, Docks)
   2.3 Fixed Infrastructure (Moored Sensors & Surface Buoys)
   2.4 Mobile Infrastructure (AUV Swarm Allocation)

3. OCEANOGRAPHIC ACOUSTIC PHYSICS ENGINE
   3.1 Empirical Sound Speed Profiling (Mackenzie 1981)
   3.2 Ship Acoustic Radiated Source Level (Ross 1976)
   3.3 Acoustic Absorption & Transmission Loss (Thorp 1967)
   3.4 Ambient Ocean Noise & Rain Impingement (Wenz 1962)
   3.5 Acoustic Doppler Shift & Latency Formulations
   3.6 Hydrodynamic Mooring Fatigue & Stochastic Current Modeling

4. ACOUSTIC FEATURE EXTRACTION & DEEP LEARNING INFERENCE
   4.1 The 23-Feature Acoustic Signature Vector
   4.2 Feature Normalization & Concurrency-Safe Pipelines
   4.3 Multi-Task Deep Neural Network Architecture (VesselMLP)
   4.4 Multi-Class Vessel Classification & Loss Formulations
   4.5 Continuous Displacement Weight Regression
   4.6 Real-Time Inference Performance Benchmarks

5. UNDERWATER WIRELESS SENSOR NETWORKING & PROTOCOLS
   5.1 Acoustic Modem Link Budget & SNR Decoding Thresholds
   5.2 Dynamic 3D Spatial Adjacency Graph Construction
   5.3 Depth-Based Routing (DBR) Protocol Implementation
   5.4 Sensor Sleep Duty Cycling & AUV Swarm Relay Mules
   5.5 Frame Compression Analysis: Latency & Energy Conservation

6. AUTONOMOUS AUV SWARM & TRAFFIC GOVERNANCE
   6.1 AUV 3D Kinematics & Hydrodynamic Current Drift
   6.2 Autonomous Battery State Machine & Dock Recharging
   6.3 Speed Governance & Violation Shadowing
   6.4 Line of Control (LOC) Screening Policies
   6.5 Autonomous Dock Slip Allocation Algorithms

7. EXPERIMENTAL EVALUATION: NORMAL VS. MONSOON REGIMES
   7.1 Experimental Setup & Unified Engine Architecture
   7.2 Baseline Performance under Normal Season Conditions
   7.3 Network Stress Analysis under Monsoon Storm Surges
   7.4 Packet Delivery Ratio (PDR) & Hop-Count Comparisons
   7.5 Moored Sensor Tether Failure & Network Degradation Analysis

8. SOFTWARE ARCHITECTURE & IMPLEMENTATION
   8.1 Modular Object-Oriented Codebase Architecture
   8.2 Decoupled Web Telemetry Architecture (Port 8000 Server)
   8.3 Verification & Quality Assurance Suite

9. CONCLUSION & FUTURE WORK
   9.1 Summary of Contributions
   9.2 Identified Limitations
   9.3 Future Directions (MARL, COLREGs, Real Sea Trials)

REFERENCES (Citing Mackenzie, Ross, Thorp, Wenz, PyTorch, NetworkX)
```

---

## 💡 17. Quantitative Results & Evaluator Q&A Defense Cheat Sheet

### Key Numbers & Metrics (Keep these memorized!)
* **Domain Size**: $10\text{ km} \times 10\text{ km}$ ($100\text{ km}^2$), depth from $-20\text{ m}$ to $-65\text{ m}$.
* **Fixed Sensors**: 35 Moored Hydrophones across 3 depth strata.
* **Mobile Robots**: 8 AUVs ($M = 500\text{ kg}$, $F_{\text{max}} = 900\text{ N}$, $v_{\text{max}} = 10\text{ m/s}$).
* **Surface Gateways**: 3 Buoys (`buoy_alpha`, `buoy_beta`, `buoy_gamma` at $X = 150\text{ m}$).
* **Vessel Fleet**: 22 Surface Vessels across 6 classes (*Cargo, Tanker, Cruiser, Ferry, Speedboat, Fishing*).
* **Acoustic Modem**: $f = 12\text{ kHz}$, $SL = 178\text{ dB}$, $R_{\text{comm}} = 1800\text{ m}$, Bitrate $= 1000\text{ bps}$, Power $= 10\text{ W}$, SNR Threshold $= 10.0\text{ dB}$.
* **Telemetry Features**: 23 extracted physical acoustic parameters.
* **ML Latency**: Sub-millisecond ($< 1\text{ ms}$) offline inference per detection event.
* **Packet Compression**: $256\text{ bytes} \rightarrow 16\text{ bytes}$ ($16\times$ reduction, $93.75\%$ latency and energy savings).
* **Normal vs. Monsoon Noise**: $65.8\text{ dB}$ (Normal) vs. $91.9\text{ dB}$ (Monsoon) — a **$+26.1\text{ dB}$ noise surge**.
* **Speed Limits**: Outer Yard $= 10.0\text{ knots}$, Inner Zone $= 6.0\text{ knots}$, LOC Tripwire $= 5000\text{ m}$.

---

### Top 10 Anticipated Questions & Bullet-Proof Answers

#### Q1: Why not just use GPS and surface radar instead of complex underwater acoustics?
> **Answer**: Radio waves and GPS signals attenuate within millimeters in seawater. Surface radar and AIS only track cooperative surface ships. If a ship turns off or spoofs its AIS transponder, or in conditions of dense fog/sea clutter, surface systems fail. Our underwater acoustic network passively detects vessels by their acoustic propeller and engine signatures regardless of transponder status, providing an un-spoofable subsurface surveillance layer.

#### Q2: Why did you implement a multi-task neural network instead of separate classification and regression models?
> **Answer**: Maintaining separate networks doubles memory usage, requires two forward passes, and doubles inference latency. A multi-task network (`VesselMLP`) shares 3 hidden layers ($128 \rightarrow 64 \rightarrow 32$), allowing both heads to share rich latent acoustic representations of hull and engine dynamics. It performs classification and continuous weight regression in a single forward pass in under $1\text{ ms}$.

#### Q3: Why is Depth-Based Routing (DBR) used instead of standard shortest-path algorithms (like Dijkstra or A*)?
> **Answer**: Dijkstra and A* require global topological knowledge and precise 3D node coordinates. In an underwater environment, GPS is unavailable, and node positions shift dynamically due to water currents. DBR requires only local pressure sensor readings (depth $Z$). A node simply forwards packets to neighbors that are shallower (closer to the surface buoys), making it robust, lightweight, and decentralized.

#### Q4: How does the network survive when seabed sensors sleep during duty cycling?
> **Answer**: Seabed sensors operate on a 70% active / 30% sleep cycle to conserve battery. When a stationary sensor sleeps, patrolling AUVs act as dynamic data mules. Because AUVs navigate across all three harbor sectors and communicate peer-to-peer, they detect vessels directly and bridge communication voids, maintaining overall network packet delivery.

#### Q5: What is the benefit of packet compression in an acoustic network?
> **Answer**: In underwater acoustics, bandwidth is severely limited ($1\text{ kbps}$) and modems consume significant power ($10\text{ W}$). Transmitting a standard $256\text{ byte}$ frame takes $2.048\text{ seconds}$ and burns $20.48\text{ Joules}$ per hop. By compressing the payload into a $16\text{ byte}$ compact binary format, transmission time drops to $0.128\text{ seconds}$ and energy drops to $1.28\text{ Joules}$—a massive $93.75\%$ reduction in latency and energy per hop.

#### Q6: How do you justify the $+26\text{ dB}$ noise surge in Monsoon Season?
> **Answer**: Our ambient noise engine implements the empirical Wenz (1962) ocean noise model and Nystuen rain models. In Monsoon season, winds increase from $12\text{ knots}$ to $45\text{ knots}$ (Douglas Sea State 7), driving violent wave turbulence. Combined with a $+15\text{ dB}$ high-frequency droplet noise component from torrential rain, the incoherent decibel sum surges the ambient noise floor from $\approx 65.8\text{ dB}$ to $\approx 91.9\text{ dB}$.

#### Q7: How does the simulation handle heavy computation without dropping frame rates?
> **Answer**: We achieved this through three architectural decisions: (1) All acoustic formulas are vectorized in pure NumPy; (2) The PyTorch model operates in pre-trained offline inference mode ($<1\text{ ms}$); (3) The web dashboards are completely decoupled from the 3D GUI—the simulation writes lightweight JSON snapshots to disk asynchronously, which the browser polls independently via a lightweight background HTTP server on port 8000.

#### Q8: What happens when an AUV runs low on battery?
> **Answer**: AUVs execute an autonomous battery state machine. If an AUV's battery drops below $15\%$, it aborts its patrol or dispatch mission, surfaces, and navigates to the nearest of 3 surface buoys. At the buoy, it engages an inductive recharge cycle at $0.2\%/\text{frame}$. Once replenished above $95\%$, it automatically returns to its designated patrol sector.

#### Q9: What happens when a hydrophone tether snaps in Monsoon Season?
> **Answer**: In extreme current conditions (e.g., during a gale squall surge), the drag force on the sensor exceeds its anchor threshold ($2.2 - 3.4\text{ m/s}$). The sensor transitions from `MOORED` to `STRAINED` and finally `SNAPPED`. It then drifts freely with the current, severing its acoustic link and leaving a fading breadcrumb trail on the 3D display. The operator can restore the network using the `⚓ Re-Anchor Sensors` button.

#### Q10: How does the system handle moving vessels that cross the Line of Control (LOC)?
> **Answer**: When an inbound vessel reaches $X = 5000\text{ m}$, the traffic controller evaluates the predicted ship type from `VesselMLP`. Heavy cargo vessels and oil tankers receive a `HALT` order to prevent harbor basin congestion. Medium passenger ferries and cruisers receive a `SLOW DOWN` order, while light craft are allowed to `CONTINUE`. If any ship exceeds zone speed limits, the nearest AUV is automatically dispatched to intercept and shadow it.
