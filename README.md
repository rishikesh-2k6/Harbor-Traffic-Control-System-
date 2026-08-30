# ⚓ Harbor Traffic Control System
### Adaptive Multi-Robot Networking & Acoustic Telemetry in Underwater Wireless Sensor Networks (UWSN)

> **Autonomous Maritime Traffic Management & Multi-Robot Swarm Telemetry**  
> *A cyber-physical simulation combining 3D underwater acoustic physics, depth-based multi-hop routing (DBR), swarm-based Autonomous Underwater Vehicle (AUV) kinematics, multi-task PyTorch vessel classification and weight regression, and real-time decoupled web dashboards across a **10 km × 10 km harbor domain**.*

---

## 📑 Table of Contents
1. [Executive Summary & Core Objectives](#-1-project-objectives)
2. [What This Project Demonstrates (Key Claims)](#-2-project-claims--demonstrations)
3. [The Working Behind the Models](#-3-the-working-behind-the-models)
   - [3.1 Oceanographic Acoustic Physics Engine](#31-oceanographic-acoustic-physics-engine)
   - [3.2 The 23-Feature Acoustic Telemetry Pipeline](#32-the-23-feature-acoustic-telemetry-pipeline)
   - [3.3 Multi-Task Deep Neural Network (`VesselMLP`)](#33-multi-task-deep-neural-network-vesselmlp)
   - [3.4 Non-Blocking Online Retraining Pipeline](#34-non-blocking-online-retraining-pipeline)
4. [Network Arrangement: Fixed vs. Dynamic Topology](#-4-network-arrangement-fixed-vs-dynamic)
   - [4.1 Fixed Infrastructure (Seabed Hydrophones & Surface Buoys)](#41-fixed-infrastructure)
   - [4.2 Dynamic Infrastructure (AUV Swarm Kinematics)](#42-dynamic-infrastructure-auv-swarm)
   - [4.3 Harbor Zone Architecture & Spatial Layout](#43-harbor-zone-architecture)
5. [Sensor Networking & Acoustic Communication Protocols](#-5-sensor-networking--acoustic-communication)
   - [5.1 Acoustic Channel Modeling & Link Budget](#51-acoustic-channel-modeling--link-budget)
   - [5.2 Link Hierarchy & Inter-Node Connections](#52-link-hierarchy--inter-node-connections)
   - [5.3 Depth-Based Routing (DBR) Protocol](#53-depth-based-routing-dbr-protocol)
   - [5.4 Duty-Cycling & Swarm Resilience](#54-duty-cycling--swarm-resilience)
   - [5.5 Frame Compression, Latency & Energy Metrics](#55-frame-compression-latency--energy-metrics)
6. [Maritime Traffic Management & Autonomous Control](#-6-maritime-traffic-management--control)
7. [System Architecture & End-to-End Data Pipeline](#-7-system-architecture--data-pipeline)
8. [Technology Stack & Architectural Rationale](#-8-technology-stack--selection-rationale)
9. [Project Directory & Module Structure](#-9-project-directory--module-structure)
10. [Quickstart, Verification & Deployment](#-10-quickstart--execution-guide)
11. [Telemetry Data Schema (`dashboard_data.json`)](#-11-telemetry-data-schema)

---

## 🎯 1. Project Objectives

Traditional maritime traffic control relies predominantly on surface radar (VTS/AIS) and satellite tracking. However, radar cannot penetrate deep water channels, AIS can be spoofed or deactivated by illicit vessels, and isolated seabed sensor grids suffer from high latency, short battery lifespans, and communication dead zones. 

The primary objective of this project is to create an **Autonomous, Active, Multi-Robot Underwater Wireless Sensor Network (UWSN)** capable of:

1. **Passive-to-Active Transformation**: Bridging static seabed hydrophones with a dynamic mobile swarm of Autonomous Underwater Vehicles (AUVs) to monitor a $10\text{ km} \times 10\text{ km}$ coastal harbor domain without blind spots.
2. **Acoustic Physics Grounding**: Accurately simulating underwater acoustic wave propagation, Doppler shifts, transmission loss, and ambient ocean noise using empirical oceanographic formulations.
3. **Multi-Task Vessel Identification**: Passively extracting 23 acoustic signatures from vessel radiated noise to classify ship types (*Cargo Ship, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel*) and predict displacement weight in kilograms using deep learning.
4. **Resilient Underwater Networking**: Dynamically establishing and maintaining multi-hop acoustic communication links between seabed sensors, mobile AUVs, and surface buoys using Depth-Based Routing (DBR) under severe duty-cycling (sleep/wake) conditions.
5. **Automated Traffic Governance**: Monitoring virtual Line of Control (LOC) boundaries, tracking speed violations, dispatching patrolling AUVs for acoustic verification, and routing vessels safely to designated harbor docks.

---

## 💡 2. Project Claims & Demonstrations

This project demonstrates a complete, closed-loop cyber-physical system. Specifically, it proves:

* **Real-time Acoustic Physics Modeling**: Realized in pure NumPy without heavy external C++ engines, calculating exact Doppler shifts, Thorp attenuation, and Mackenzie sound speed profiles for every detection event.
* **Multi-Task Deep Learning from Noisy Radiated Sound**: Simultaneously classifying vessel categories (6 classes) and regressing displacement weight ($1\,\text{kTon}$ to $200\,\text{kTon}$) via a unified PyTorch neural network sharing latent representations.
* **Swarm AUV Kinematics & Adaptive Relay**: Mobile AUVs dynamically patrol designated sectors, steer via 3D waypoint navigation, account for depth-dependent ocean current drift, manage battery drain/dock charging, and act as mobile data mules when static sensors sleep.
* **Dynamic Network Topology & Depth-Based Routing (DBR)**: Frame-by-frame graph construction in NetworkX, resolving multi-hop acoustic routes from seabed depths (up to $-65\text{m}$) to surface gateway buoys ($Z=0\text{m}$), achieving $>95\%$ Packet Delivery Ratio (PDR).
* **Energy Optimization via Packet Compression**: Evaluating default (256 bytes) versus compressed (16 bytes) acoustic packets, demonstrating dramatic reductions in transmission latency and acoustic modem energy consumption (Joules).
* **Decoupled Real-Time Web Telemetry**: Serving 3D harbor telemetry snapshots via an asynchronous Python HTTP server to interactive HTML5/Canvas/JS dashboards without blocking the 30 FPS Matplotlib simulation loop.

---

## 🧠 3. The Working Behind the Models

### 3.1 Oceanographic Acoustic Physics Engine

The simulation environment integrates empirical underwater acoustics to generate realistic physical telemetry for each vessel detection:

```
Vessel Propeller/Hull Noise (Ross 1976)
                  │
                  ▼
   Sound Speed Profile (Mackenzie 1981)
                  │
                  ▼
   Transmission Loss (Spherical + Thorp 1967)
                  │
                  ▼
   Ambient Sea Noise (Wenz 1962)
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

#### 5. Wenz (1962) Ambient Ocean Noise Model
Combining wind-driven surface turbulence, shipping density factor, and biological background noise:
$$NL_{\text{ambient}} = 10\log_{10}\left(10^{NL_{\text{wind}}/10} + 10^{NL_{\text{shipping}}/10} + 10^{NL_{\text{bio}}/10}\right)$$

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

```
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
* **Classification Task**: Predicts probability across 6 maritime vessel classes (*Cargo Ship, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel*) using Cross-Entropy Loss ($\mathcal{L}_{\text{class}}$).
* **Regression Task**: Predicts continuous vessel displacement tonnage/weight ($\text{kg}$) using Mean Squared Error ($\mathcal{L}_{\text{weight}}$).
* **Total Multi-Task Loss**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{class}} + \lambda \cdot \mathcal{L}_{\text{weight}}$$

---

### 3.4 Non-Blocking Online Retraining Pipeline

The simulator writes real sensor detections directly to `collected_data.csv`. The ML subsystem operates continuously in the background:

1. **Safe Concurrent File Access**: Implements exponential backoff retry loops (`_safe_read_csv`, `_safe_write_csv`) to prevent Windows disk-lock race conditions between simulation writes and training reads.
2. **Dynamic Training Triggers**: Automatically trains once $100$ initial detection records accumulate, and retrains dynamically every $50$ newly collected samples.
3. **Inference Snapshots**: Writes real-time predictions per active ship to a memory-mapped temporary table (`stage3_live_predictions.csv`), ingested instantly by the traffic control algorithm.

---

## 🌐 4. Network Arrangement: Fixed vs. Dynamic

The harbor network combines fixed underwater sensor infrastructure with a dynamic mobile AUV swarm:

```
Z = 0 m   (Surface)     [Buoy Alpha]      [Buoy Beta]      [Buoy Gamma]
                             ▲                 ▲                ▲
                             │                 │                │
Z = -10 m (Mid-Water)    [AUV-Echo] ──────► [AUV-Alpha] ──────► [AUV-Bravo]  (Mobile Swarm)
                             ▲                 ▲                ▲
                             │                 │                │
Z = -50 m (Seabed)       [Sensor-1]       [Sensor-14]      [Sensor-28]       (Fixed Grid)
```

### 4.1 Fixed Infrastructure

* **35 Stationary Seabed & Mid-Water Hydrophones**:
  * Distributed across the $10\text{ km} \times 10\text{ km}$ harbor domain.
  * Stratified into 3 depth levels:
    * Surface sensors ($Z = -2\text{m}$)
    * Mid-water sensors ($Z = \text{seabed} / 2$)
    * Seabed sensors ($Z = \text{seabed} + 2\text{m}$, up to $-65\text{m}$)
  * Equipped with omnidirectional acoustic modems ($R_{\text{comm}} = 1500\text{m}$, sensing radius $R_{\text{detect}} = 2000\text{m}$).
* **3 Fixed Surface Gateway Buoys**:
  * Positioned near the harbor entrance along $X = 150\text{m}$:
    * `buoy_alpha`: $(150\text{m}, 2000\text{m}, 0\text{m})$
    * `buoy_beta`: $(150\text{m}, 5000\text{m}, 0\text{m})$
    * `buoy_gamma`: $(150\text{m}, 8000\text{m}, 0\text{m})$
  * Serve as high-speed RF/Satellite gateway sinks connected to the harbor master command station and provide inductive recharge docks for AUVs.

---

### 4.2 Dynamic Infrastructure (AUV Swarm)

A swarm of 6–8 Autonomous Underwater Vehicles (AUVs / `RobotNode`) provides adaptive mobile sensing and communication relays:

| AUV Name | Assigned Zone | Patrol Coordinates / Waypoints | Primary Role |
|---|---|---|---|
| **AUV-Alpha (101)** | Outer Yard ($X > 5000\text{m}$) | $(6500, 2000) \rightarrow (8500, 3500) \rightarrow (7500, 5000)$ | Inbound vessel screening & telemetry relay |
| **AUV-Bravo (102)** | Outer Yard ($X > 5000\text{m}$) | $(7500, 6000) \rightarrow (9000, 7500) \rightarrow (8000, 9000)$ | Deep-water monitoring & long-range relay |
| **AUV-Charlie (103)** | LOC Boundary ($X \approx 5000\text{m}$) | $(5000, 1000) \leftrightarrow (5000, 4500)$ | Virtual tripwire acoustic barrier patrol |
| **AUV-Delta (104)** | LOC Boundary ($X \approx 5000\text{m}$) | $(5000, 5500) \leftrightarrow (5000, 9000)$ | Speed violation intercept & tracking |
| **AUV-Echo (105)** | Inner Zone ($700\text{m} < X < 5000\text{m}$) | $(2000, 2500) \rightarrow (4000, 4500) \rightarrow (2500, 5500)$ | Mid-channel guidance & bridge to buoys |
| **AUV-Foxtrot (106)** | Inner Zone ($700\text{m} < X < 5000\text{m}$) | $(1500, 6500) \rightarrow (3500, 8500) \rightarrow (2000, 9000)$ | Harbor approach traffic coordination |

#### AUV Kinematics & Swarm Physics:
* **Waypoint Steering**: Directs AUV thrust vectors towards current 3D waypoints.
* **Environmental Drift**: Evaluates depth-decaying 3D water current vectors ($v_x, v_y, v_z$) based on harbor tidal oscillations.
* **Drag Damping**: Applies hydrodynamic drag damping ($\text{drag\_coeff} = 0.85$, mass $= 500\text{ kg}$).
* **Battery State Machine**: Discharges battery during cruising ($0.015\%/\text{frame}$); if battery drops below $15\%$, enters `CHARGING` status and autonomously navigates to the nearest surface buoy for rapid recharge ($0.2\%/\text{frame}$).

---

### 4.3 Harbor Zone Architecture

```
X = 0 m              X = 700 m        X = 5000 m                       X = 10000 m
|---- HARBOR --------|--- INNER ZONE---|---------------- OUTER YARD -----------------|
 (Docking Slots)      (Patrol, 6 kn)   (LOC Tripwire Wall)            (10 kn Limit)
 (Surface Buoys)      (AUVs 105, 106)  (AUVs 103, 104)                (AUVs 101, 102)
```

1. **Outer Yard Zone ($X: 5000\text{m} – 10000\text{m}$)**: Entry fairway for all inbound shipping traffic. Speed limit: **10.0 knots**.
2. **Line of Control (LOC) Boundary ($X = 5000\text{m}$)**: Virtual acoustic tripwire. Inbound ships crossing the LOC trigger mandatory identity verification and speed checks.
3. **Inner Patrol Zone ($X: 700\text{m} – 5000\text{m}$)**: Active interception zone where AUVs shadow vessels exhibiting classification discrepancies or overspeed flags. Speed limit: **6.0 knots**.
4. **Harbor Docking Area ($X < 700\text{m}$)**: Dedicated docking slips categorized by vessel classification.

---

## 📡 5. Sensor Networking & Acoustic Communication

### 5.1 Acoustic Channel Modeling & Link Budget

Underwater wireless acoustic communication is fundamentally constrained by slow sound propagation ($c \approx 1500\text{ m/s}$) and narrow bandwidths:

* **Carrier Frequency ($f_{\text{comm}}$)**: $12.0\text{ kHz}$
* **Modem Transmit Source Level ($SL_{\text{comm}}$)**: $178.0\text{ dB re } 1\,\mu\text{Pa @ } 1\text{m}$
* **Communication Range ($R_{\text{comm}}$)**: $1500\text{ meters}$
* **Link SNR Threshold**: $10.0\text{ dB}$ (minimum for packet decoding)
* **Acoustic Bitrate**: $1000\text{ bps}$ ($1\text{ kbps}$)

```
            Tx Node (Source Level = 178 dB)
                         │
                         ▼
             Transmission Loss: TL(d) = 20*log10(d) + alpha*d
                         │
                         ▼
             Rx Signal Level = 178 - TL(d)
                         │
                         ▼
             Ambient Noise Level (Wenz Sea State 3)
                         │
                         ▼
             Link SNR = Rx Signal Level - Noise Level
                         │
       ┌─────────────────┴─────────────────┐
       ▼                                   ▼
  SNR >= 10 dB                        SNR < 10 dB
 [Link Active (Graph Edge)]         [Link Dropped (Out of Range)]
```

---

### 5.2 Link Hierarchy & Inter-Node Connections

Every simulation frame, `NetworkManager` and `network.py` construct a dynamic spatial graph containing 4 distinct link types:

1. **Seabed Sensor $\leftrightarrow$ AUV Link**: When a stationary seabed sensor detects a ship, it establishes an uplink to any patrolling AUV within $1500\text{m}$.
2. **AUV $\leftrightarrow$ AUV Link (Inter-Robot Mesh)**: Mobile AUVs within mutual communication range form an ad-hoc peer-to-peer relay backbone.
3. **AUV $\leftrightarrow$ Surface Buoy Gateway Link**: Shallow/surfacing AUVs offload aggregated telemetry packets directly to surface buoys (`buoy_alpha`, `buoy_beta`, `buoy_gamma`).
4. **AUV $\leftrightarrow$ Vessel Interrogation Link**: Direct acoustic pings between patrolling AUVs and vessel transponders for on-site speed enforcement.

Link status is categorized dynamically:
* `ACTIVE` ($\text{SNR} > 12\text{ dB}$): High-quality link, up to $64\text{ kbps}$ nominal capacity.
* `DEGRADED` ($3\text{ dB} < \text{SNR} \le 12\text{ dB}$): Weak link with reduced throughput and packet retries.
* `DISCONNECTED` ($\text{SNR} \le 3\text{ dB}$): Link severed due to excessive transmission loss or ambient noise.

---

### 5.3 Depth-Based Routing (DBR) Protocol

Because acoustic transmission loss increases with distance and GPS is unavailable underwater, the system implements **Depth-Based Routing (DBR)**:

```
[Seabed Sensor (Z = -50m)] ──(Deeper)──┐
                                       ▼
                       [Intermediate AUV (Z = -25m)] ──(Shallower)──┐
                                                                    ▼
                                                    [Surfaced AUV (Z = -5m)]
                                                                    │
                                                                    ▼
                                                     [Surface Buoy (Z = 0m)]
```

#### DBR Routing Algorithm:
1. **Packet Origin**: A detecting node (sensor or AUV) initiates a telemetry packet.
2. **Gateway Check**: If any surface buoy is directly reachable, the packet hops directly to the nearest buoy.
3. **Depth Gradient Selection**: If no buoy is in range, the node queries its active 1-hop neighbors and filters for **shallower nodes** ($Z_{\text{neighbor}} > Z_{\text{current}}$, i.e., closer to depth $0\text{m}$).
4. **Greedy Forwarding**: The packet is forwarded to the neighbor with the highest depth coordinate (closest to the surface).
5. **Loop Avoidance**: Visited nodes are recorded; paths exceeding 12 hops or encountering depth local minima are safely terminated.

---

### 5.4 Duty-Cycling & Swarm Resilience

To conserve battery life, stationary seabed sensors operate on a **$70\%$ active / $30\%$ sleep duty cycle** (active for 35 frames, sleeping for 15 frames). 

* **The Problem**: When seabed sensors sleep, static network topologies suffer severe network partitions, dropping vessel detection alerts.
* **The Swarm Solution**: Mobile AUVs stay continuously awake, patrolling across sensor clusters. When a static sensor enters sleep mode, an adjacent AUV detects the vessel directly and acts as a dynamic mobile relay bridge, maintaining $>95\%$ Packet Delivery Ratio (PDR) across the entire harbor domain.

---

### 5.5 Frame Compression, Latency & Energy Metrics

The networking layer compares standard uncompressed telemetry payloads against optimized compressed binary telemetry:

| Metric | Default Uncompressed Frame | Compressed Telemetry Frame | Performance Improvement |
|---|---|---|---|
| **Frame Size** | $256\text{ bytes}$ ($2048\text{ bits}$) | $16\text{ bytes}$ ($128\text{ bits}$) | **$16\times$ Payload Reduction** |
| **Transmission Delay per Hop** | $\frac{2048\text{ b}}{1000\text{ bps}} = 2.048\text{ s}$ | $\frac{128\text{ b}}{1000\text{ bps}} = 0.128\text{ s}$ | **$93.75\%$ Faster Tx Delay** |
| **Energy Consumption per Hop** | $10\text{W} \times 2.048\text{s} = 20.48\text{ Joules}$ | $10\text{W} \times 0.128\text{s} = 1.28\text{ Joules}$ | **$93.75\%$ Energy Saved per Hop** |
| **End-to-End Latency (3 Hops)** | $\approx 6.5\text{ seconds}$ | $\approx 0.7\text{ seconds}$ | **Near Real-Time Telemetry** |

---

## 🚦 6. Maritime Traffic Management & Control

The traffic management algorithm (`traffic_algo.py`) executes automated harbor governance based on live telemetry:

### 1. Speed Limit Enforcement
* **Outer Yard ($X > 5000\text{m}$)**: Limit $= 10.0\text{ knots}$.
* **Inner Patrol Zone ($X: 700\text{m} - 5000\text{m}$)**: Limit $= 6.0\text{ knots}$.
* **Violation Alert**: If $v_{\text{knots}} > v_{\text{limit}}$, vessel state flips to `OVERSPEED`, triggering real-time web alerts and dispatching the nearest AUV to shadow the vessel.

### 2. Line of Control (LOC) Screening
* As inbound ships cross $X = 5000\text{m}$, the system cross-references true acoustic signatures against the PyTorch `VesselMLP` model to verify ship identity and cargo weight.

### 3. Automated Dock Slot Allocation
Upon approaching $X \le 700\text{m}$, vessels are assigned to dedicated color-coded docking slots:

```
Y = 10000 m ┌──────────────────────────────────────────────┐
            │ [Dock-B1..B4]  Cruisers         (Y: 7500-8450) │
            │ [Dock-F]       Ferries          (Y: 6825)      │
            │ [Dock-V1..V4]  Fishing Vessels  (Y: 5200-6100) │
            │ [Dock-C]       Speedboats       (Y: 4125)      │
            │ [Dock-T1..T2]  Tankers          (Y: 2700-3000) │
            │ [Dock-A]       Cargo Ships      (Y: 2000)      │
Y = 0 m     └──────────────────────────────────────────────┘
            X = 300 m (Dock Berths)
```

---

## 🏗️ 7. System Architecture & Data Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        10 km × 10 km 3D HARBOR SIMULATION                              │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │ 18 Surface Ships      │  │ 35 Seabed Hydrophones  │  │ 6-8 Mobile Patrolling     │  │
│  │ (Cargo, Tanker, etc.) │  │ (3 Depth Layers, DBR)  │  │ AUVs (Swarm Kinematics)   │  │
│  └───────────┬───────────┘  └───────────┬────────────┘  └─────────────┬─────────────┘  │
└──────────────┼──────────────────────────┼─────────────────────────────┼────────────────┘
               │                          │                             │
               ▼                          ▼                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ACOUSTIC PHYSICS & TELEMETRY ENGINE                             │
│  • Mackenzie Sound Speed (T, S, D)      • Ross Propeller Source Level (SL)             │
│  • Thorp Absorption & Transmission Loss • Doppler Radial Velocity Shift                │
│  • Wenz Ocean Ambient Noise Model       • 23-Feature Telemetry Extraction              │
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
│  • Dynamic Online Retraining            • Task 2: Weight Regression in kg (MSE)        │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│     MARITIME TRAFFIC CONTROL SYSTEM       │ │       REAL-TIME MONITORING SUITE          │
│ • Speed Limit Enforcement (Outer/Inner)   │ │ • Native 3D Matplotlib GUI Loop           │
│ • LOC Virtual Curtain Crossings           │ │ • Background Python HTTP Server (Port 8000)│
│ • Collision Avoidance Safe Distances      │ │ • Harbor Web Dashboard (dashboard.html)   │
│ • Classification-Based Dock Allocation    │ │ • Network Topology (networking_dashboard) │
└───────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## 💻 8. Technology Stack & Selection Rationale

| Component | Technology | Rationale & Architectural Trade-offs |
|---|---|---|
| **Simulation Core** | **Python 3.9+** | Unifies deep learning (`torch`), graph theory (`networkx`), and scientific computing (`numpy`, `pandas`) into a single runtime without inter-language bridge overhead. |
| **Deep Learning** | **PyTorch (`VesselMLP`)** | Multi-task architecture shares 3 hidden layers across classification and regression. Enables dynamic online weight updates during continuous simulation runs. |
| **Acoustic Physics** | **Vectorized NumPy** | Replaces heavy C++ game physics (PyBullet/Unity). Custom NumPy implementation executes cross-platform without platform-specific compiler dependencies. |
| **Network Topology** | **NetworkX** | Constructs spatial graphs frame-by-frame, evaluating acoustic SNR thresholds and executing Depth-Based Routing (DBR) across dynamic node coordinates. |
| **Interactive 3D GUI** | **Tkinter + Matplotlib 3D (`TkAgg`)** | Provides immediate, native 3D visualization with interactive 360° orbital camera rotation, zoom controls, and projection resets. |
| **Decoupled Telemetry UI** | **HTML5 / CSS3 / Vanilla JS + `http.server`** | Decouples browser visualization from the 3D animation loop. Simulation writes JSON snapshots every $500\text{ms}$, served asynchronously to avoid UI rendering lag. |

---

## 📁 9. Project Directory & Module Structure

```
Harbor-Traffic-Control-System/
├── README.md                      # Comprehensive A-Z System Documentation
├── PROJECT CONTEXT.md             # Research context & multi-robot extension goals
├── main.py                        # Central orchestrator & interactive 3D Tkinter GUI
├── simulator.py                   # Acoustic physics engine & 10km harbor environment
├── mlmodel.py                     # Multi-task PyTorch neural network (VesselMLP)
├── network.py                     # Depth-Based Routing (DBR) & acoustic packet logic
├── network_manager.py             # AUV swarm management & inter-node CommLinks
├── auv.py                         # Standalone NumPy AUV kinematics & battery state machine
├── traffic_algo.py                # Speed governance, LOC tripwire & dock slot assignment
├── verify_environment.py          # Automated system diagnostics & test runner
├── dashboard.html                 # Real-time harbor traffic & vessel telemetry web dashboard
├── networking_dashboard.html      # Acoustic network topology & DBR routing web dashboard
├── dashboard_data.json            # Live exported JSON telemetry stream
├── dashboard_data.js              # JavaScript telemetry wrapper
├── model_state.pt                 # Pre-trained PyTorch multi-task model weights
├── scaler.pkl                     # Scikit-learn StandardScaler for 23 features
├── label_encoder.pkl              # Scikit-learn LabelEncoder for 6 ship classes
└── collected_data.csv             # Continuous growing dataset of physical sensor detections
```

---

## 🚀 10. Quickstart & Execution Guide

### 1. Installation & Environment Setup
Clone the repository and install the required Python dependencies:

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

* **Interactive 3D Controls**: Left-click & drag to rotate 360°, scroll wheel to zoom, click **⟲ Reset View** to restore default perspective.
* **Auto-Started Telemetry Server**: Automatically spawns an HTTP server on `http://127.0.0.1:8000`.

### 4. Access Live Web Dashboards
Open either dashboard in your browser:
* **Harbor Traffic Dashboard**: `http://127.0.0.1:8000/dashboard.html`  
  *(Live ship tracking, speed violations, ML predictions, dock slot occupancy)*
* **Acoustic Network Dashboard**: `http://127.0.0.1:8000/networking_dashboard.html`  
  *(Dynamic topology graph, active acoustic links, DBR hop counts, AUV battery levels)*

---

## 📊 11. Telemetry Data Schema

The simulation continuously streams JSON telemetry snapshots to `dashboard_data.json`:

```json
{
  "frame": 140,
  "active_sensors": 18,
  "total_sensors": 35,
  "ml_status": "ML LIVE (CPU) — 1450 detections",
  "violations": [
    {
      "id": 6,
      "type": "Cruiser",
      "zone": "outer_yard",
      "speed": 14.7,
      "limit": 10.0
    }
  ],
  "vessels": [
    {
      "id": 1,
      "type": "Cargo Ship",
      "pred_type": "Cargo Ship",
      "pred_weight": 142000,
      "state": "INBOUND",
      "cmd": "CONTINUE",
      "x": 7808.4,
      "y": 5430.8,
      "dock_slot": "Dock-A"
    }
  ],
  "robots": [
    {
      "id": 101,
      "name": "AUV-Alpha",
      "x": 6500.0,
      "y": 3000.0,
      "z": -15.0,
      "zone": "outer_yard",
      "battery": 97.8,
      "status": "PATROLLING",
      "comm_range": 1200.0,
      "active_links": 3
    }
  ],
  "network_links": [
    {
      "id": "link_R101_S12",
      "source": "Robot-101",
      "target": "Sensor-12",
      "distance_m": 842.1,
      "snr_db": 16.4,
      "status": "ACTIVE",
      "throughput_kbps": 35.0
    }
  ],
  "network_stats": {
    "total_robots": 6,
    "active_links_count": 14,
    "degraded_links_count": 2,
    "total_links": 16,
    "avg_throughput_kbps": 36.8,
    "packet_delivery_rate": 0.96
  }
}
```

---

## 🏆 Summary of Engineering Innovations

1. **Acoustic Physics Accuracy**: Full implementation of Mackenzie, Thorp, Ross, Wenz, and Doppler equations without external heavy physics engines.
2. **Unified Multi-Task Deep Learning**: Simultaneous ship classification and weight estimation with dynamic online model updates.
3. **Adaptive Swarm Relay**: Autonomous AUVs dynamically bridging communication partitions caused by sensor sleep duty cycles.
4. **Depth-Based Acoustic Routing (DBR)**: Multi-hop greedy depth routing achieving $>95\%$ Packet Delivery Ratio under realistic oceanographic constraints.
5. **Decoupled Real-Time Visualization**: High-performance dual dashboard architecture serving live telemetry without degrading 3D simulation frame rates.
