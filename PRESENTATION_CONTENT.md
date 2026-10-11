# ⚓ Harbor Traffic Control System: Presentation Content & Speaker Guide
### Course: INTRODUCTION TO AI ROBOTICS (22AIE214)
**Amrita Vishwa Vidyapeetham, Amaravati Campus**  
**Under the Guidance of:** Dr. Ravishankar Prakash Desai  
**Project Team:** J Kiran Sri Datta Sai (AV.SC.U4AIE24121), Mogalapalli H N V Satya Sai Ganesh (AV.SC.U4AIE24139), Penigonda Rahul (AV.SC.U4AIE24151)

---

## 📂 Generated Presentation File
- **PowerPoint File:** [`Harbor_Traffic_Control_Presentation.pptx`](file:///p:/Projects/Harbor-Traffic-Control-System-/Harbor_Traffic_Control_Presentation.pptx)
- **High-Resolution Slide Figures:** [`presentation_assets/`](file:///p:/Projects/Harbor-Traffic-Control-System-/presentation_assets)
  - [`flowchart.png`](file:///p:/Projects/Harbor-Traffic-Control-System-/presentation_assets/flowchart.png)
  - [`ml_results.png`](file:///p:/Projects/Harbor-Traffic-Control-System-/presentation_assets/ml_results.png)
  - [`networking_results.png`](file:///p:/Projects/Harbor-Traffic-Control-System-/presentation_assets/networking_results.png)
  - [`harbor_architecture.png`](file:///p:/Projects/Harbor-Traffic-Control-System-/presentation_assets/harbor_architecture.png)

---

## 📽️ Slide-by-Slide Detailed Content & Speaker Notes

### Slide 1: Title Slide
- **Header:** Amrita Vishwa Vidyapeetham, Amaravati Campus
- **Category:** Lab-Based Project Title
- **Title:** Adaptive Multi-Robot Networking & Acoustic Telemetry in UWSN for Harbor Monitoring
- **Team Table:**
  | Name of the Student | Roll No. |
  | :--- | :--- |
  | J Kiran Sri Datta Sai | AV.SC.U4AIE24121 |
  | Mogalapalli H N V Satya Sai Ganesh | AV.SC.U4AIE24139 |
  | Penigonda Rahul | AV.SC.U4AIE24151 |
- **Guidance:** Under the Guidance of **Dr. Ravishankar Prakash Desai**
- **Footer:** 04-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 1

> **🎤 Speaker Notes (Slide 1):**  
> *"Good morning/afternoon respected guide Dr. Ravishankar Prakash Desai and panel members. Today, our team—Kiran, Ganesh, and Rahul—presents our AI Robotics project titled 'Adaptive Multi-Robot Networking & Acoustic Telemetry in Underwater Wireless Sensor Networks for Harbor Monitoring'. In this work, we transition conventional passive underwater monitoring into an active, collaborative robotic swarm of Autonomous Underwater Vehicles (AUVs) that autonomously monitor maritime traffic, classify vessels via acoustic signatures, and preserve communication links even under severe oceanic storms."*

---

### Slide 2: Introduction
- **Slide Title:** Introduction
- **Bullet Points:**
  - Maritime commerce accounts for over 80% of global freight transport; harbor traffic density poses extreme navigational safety and coastal defense challenges.
  - Conventional monitoring depends heavily on surface radar (VTS) and transponders (AIS), which cannot penetrate water columns and can be deactivated by rogue vessels.
  - Underwater Wireless Sensor Networks (UWSNs) provide submerged monitoring, but static seabed sensors suffer from high acoustic attenuation, multi-path fading, and battery depletion.
  - Harsh ocean dynamics, thermocline gradients, and seasonal storms (monsoons) severely degrade acoustic modem Signal-to-Noise Ratio (SNR), severing static links.
  - Integrating Autonomous Underwater Vehicles (AUVs) as dynamic swarm relays bridges communication dead zones and enables proactive autonomous maritime traffic governance.
- **Embedded Graphic:** Multi-tier harbor architecture showing Surface Buoys (Z=0m), mobile AUV relays (Z=-10m to -25m), and seabed hydrophones (Z=-50m).

> **🎤 Speaker Notes (Slide 2):**  
> *"Ports and harbor channels are the economic lifelines of global commerce. Currently, harbor authorities rely on radar and AIS transponders. However, AIS can be switched off by illicit vessels, and radar cannot penetrate below the water surface. While underwater sensors can listen to acoustic radiated noise from ship propellers and machinery, underwater acoustics is notoriously harsh—sound travels slowly (~1500 m/s), experiences frequency-dependent absorption, and suffers from severe multipath fading. Our innovation is introducing mobile Autonomous Underwater Vehicles (AUVs) that actively patrol the harbor, dynamically bridge communication gaps, and provide robotic inspection."*

---

### Slide 3: Problem Statement, Objectives and Programming Language
- **Slide Title:** Problem Statement, Objectives and Programming Language
- **Problem Statement:**  
  *Traditional coastal monitoring cannot detect non-cooperative underwater and surface vessels in all-weather conditions. Static underwater sensor networks suffer from limited communication range, high ambient noise, and topology fragmentation during storm surges.*
- **Objectives:**
  - Simulate a 10 km × 10 km harbor domain with 35 moored hydrophones, 8 AUVs, and 3 surface gateway buoys.
  - Model empirical underwater acoustics: Mackenzie sound speed, Ross propeller noise, Thorp attenuation, and Wenz ambient noise.
  - Extract 23 acoustic features and train a multi-task deep neural network (`VesselMLP`) to classify 6 ship types and predict displacement weight.
  - Implement Depth-Based Routing (DBR) with binary packet compression to achieve low latency and conserve battery energy.
  - Automate harbor traffic governance: speed violation detection, Line of Control (LOC) fairway screening, and dock berth allocation.
- **Tools & Environment Card:**
  - **Language:** Python 3.9+
  - **AI / ML:** PyTorch (`VesselMLP`), Scikit-Learn
  - **Networking:** NetworkX (Dynamic DBR Graph Construction)
  - **Physics & Math:** NumPy, SciPy (Acoustic Equations & Kinematics)
  - **GUI & Dashboards:** Matplotlib 3D, Tkinter, HTML5/Canvas/JS
  - **Dataset:** 132,000+ Physical Acoustic Telemetry Records
  - **Deployment:** Decoupled Async Telemetry Server (Port 8000)

> **🎤 Speaker Notes (Slide 3):**  
> *"Our problem statement tackles two fundamental challenges: first, passive acoustic detection in severe oceanic noise; second, routing telemetry packets from seabed depths to surface gateways when static nodes enter sleep modes or get disconnected by storm currents. Our objectives cover the end-to-end cyber-physical pipeline: accurate ocean acoustic physics modeling, deep neural multi-task inference, swarm networking via Depth-Based Routing, and automated traffic rules. We implemented the entire framework in Python using PyTorch for the deep learning model, NetworkX for dynamic topology graphs, and decoupled HTML5 dashboards for real-time visualization."*

---

### Slide 4: Methodology
- **Slide Title:** Methodology
- **Structured Sections:**
  - **Robotic System Architecture:** Multi-tier cyber-physical network across a 10 km × 10 km harbor domain consisting of 35 moored seabed hydrophones (-50m to -65m), 8 autonomous mobile AUVs (-10m to -25m), and 3 surface gateway buoys (Z = 0m).
  - **Acoustic Physics & Sensing Pipeline:** Ships radiate noise modeled by Ross (1976). Sensors compute Doppler shifts and Thorp transmission loss using Mackenzie 9-term sound speed, compiling a 23-feature physical telemetry vector.
  - **Multi-Task Deep Learning Inference (`VesselMLP`):** A unified PyTorch network with 3 shared latent layers (128 → 64 → 32) feeding two heads simultaneously: 6-class classification (Cargo, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel) and continuous weight regression (1 kTon to 400 kTon).
  - **Adaptive Depth-Based Routing (DBR) & Swarm Relaying:** Telemetry packets forward greedily along negative depth gradients toward surface buoys. Mobile AUVs act as dynamic data mules during sensor sleep duty cycles (70/30) and storm severances.
  - **Environmental Simulation & Protocol Optimization:** Unified engine comparing Normal Season (65.8 dB noise) against Monsoon Season (91.9 dB noise, 2.6× currents). 16-byte packet compression saves 93.75% transmission delay and modem battery energy.

> **🎤 Speaker Notes (Slide 4):**  
> *"Our methodology is structured around four interlocking pillars. First, physics-based acoustic telemetry: we calculate Mackenzie sound speed profiles and Thorp attenuation for every detection event. Second, our multi-task neural network: instead of running separate heavy models, our VesselMLP uses shared latent representations to classify ship types and predict displacement weight in sub-millisecond execution time. Third, our networking protocol: we use Depth-Based Routing, where nodes forward packets upward to shallower neighbors without requiring full-network routing tables or GPS, which is unavailable underwater. Finally, AUVs patrol sectors, shadow speeding vessels, and recharge autonomously at surface buoys when battery drops below 15%."*

---

### Slide 5: Flow-chart
- **Slide Title:** Flow-chart
- **Flow Description:**
  1. `[START]`
  2. 10 km × 10 km Ocean Simulation & Vessel Dynamics (22 Surface Ships, Normal vs. Monsoon Season)
  3. Acoustic Physics & Telemetry Engine (Mackenzie Sound Speed, Ross Source Level, Thorp TL, Doppler)
  4. 23-Dimensional Acoustic Feature Vector Compilation (7 Energy, 3 Propagation, 5 Environmental, 6 Engineered)
  5. Multi-Task Deep Neural Network (`VesselMLP`): Shared Layers (128→64→32) → 6-Class Head + Weight Head
  6. Depth-Based Routing (DBR) & AUV Swarm Relay (Greedy Depth Gradient Forwarding, 70/30 Duty Cycle, Compression)
  7. Autonomous Harbor Traffic Governance (Overspeed Tracking, LOC Fairway Screening, Dock Berth Routing)
  8. Real-Time Telemetry Broadcast & Decoupled Web Dashboards (3D Matplotlib GUI + HTML5/Canvas Live Stream)
  9. `[END]`

> **🎤 Speaker Notes (Slide 5):**  
> *"Slide 5 illustrates the complete procedural pipeline of the Harbor Traffic Control System. It begins with the continuous physical simulation of ships and hydrodynamics. When a vessel enters the 2 km sensing envelope of a hydrophone, acoustic equations compute transmission loss and Doppler shift. The resulting 23-feature vector is passed to VesselMLP. The classification and weight are packed into a 16-byte compressed frame, routed via DBR through seabed nodes and mobile AUVs up to surface buoys, and fed directly into the traffic controller to enforce speed limits and allocate dock berths."*

---

### Slide 6: Simulation Result (Machine Learning & Classification)
- **Slide Title:** Simulation Result: Machine Learning & Classification
- **Visuals:** 4-panel evaluation dashboard (`presentation_assets/ml_results.png`):
  1. Multi-Class ROC Curves (Macro AUC = 0.989)
  2. Normalized Confusion Matrix (Test Accuracy = 95.80%)
  3. Joint Training/Validation Loss Convergence (1.020 → 0.048)
  4. Accuracy Progression Curve (Reaching 95.8%)
- **Performance Highlights:**
  - • **Overall Test Accuracy:** **95.80%** across 6 vessel categories.
  - • **Macro Average AUC:** **0.989**, proving robust discrimination despite acoustic noise.
  - • **Class Sensitivity:** Speedboat (98.0%), Tanker (98.0%), Fishing Vessel (97.0%), Cargo Ship (96.0%).
  - • **Loss Convergence:** Combined Cross-Entropy + MSE loss dropped smoothly from 1.020 to 0.048.
  - • **Continuous Weight Estimation:** Highly accurate displacement tonnage regression from 1 kTon to 400 kTon.
  - • **Inference Latency:** Sub-millisecond inference (<0.8 ms per ship) ensures real-time operational responsiveness.

> **🎤 Speaker Notes (Slide 6):**  
> *"Turning to our machine learning results, our model was evaluated on over 132,000 real-time acoustic samples. As seen in the confusion matrix and ROC curves, our model achieves an overall classification accuracy of 95.80% with an impressive Macro AUC of 0.989. Speedboats and Tankers are classified with 98% sensitivity due to their distinct acoustic energy and Doppler profiles. The joint loss curve demonstrates smooth convergence without overfitting, thanks to our shared representation architecture, and sub-millisecond inference ensures no CPU lag in the 3D loop."*

---

### Slide 7: Simulation Result (Multi-Robot Networking & Scenarios)
- **Slide Title:** Simulation Result: Multi-Robot Networking & Scenarios
- **Visuals:** Comparative bar charts (`presentation_assets/networking_results.png`):
  1. Environmental Resilience (Normal vs. Monsoon Season)
  2. Telemetry Frame Compression Gains (256 Bytes vs. 16 Bytes)
- **Performance Highlights:**
  - • **Monsoon Acoustic Degradation:** Torrential rain and storm wind noise surge ambient noise by **+26.1 dB** (65.8 dB → 91.9 dB), reducing 1 km acoustic modem SNR from 50.5 dB to 24.5 dB.
  - • **Swarm Relay Resilience:** Despite peripheral link drops and drifting hydrophones, mobile AUV relays maintain a **92.4% Packet Delivery Ratio** (compared to 96.8% in calm seas).
  - • **Adaptive Multi-Hop Routing:** DBR automatically scales average forwarding hops from **1.4 → 2.3 hops** to navigate around acoustic dead zones.
  - • **Frame Compression Impact:** Compressing telemetry from 256 bytes to 16 bytes reduces transmission delay from **2.048 s to 0.128 s per hop** (**93.75% reduction**).
  - • **Energy Conservation:** Modem energy consumption drops from **20.48 J → 1.28 J per hop**, dramatically extending underwater battery lifetime.

> **🎤 Speaker Notes (Slide 7):**  
> *"Slide 7 showcases our networking and swarm performance under realistic ocean weather conditions. Under our Monsoon scenario, ambient noise surges by 26 dB due to gale winds and rain impact, collapsing modem SNR. In a static sensor network, this would cause catastrophic communication blackouts. However, our AUV swarm acts as dynamic relays, maintaining a 92.4% packet delivery ratio. Furthermore, our 16-byte binary telemetry compression delivers a 93.75% savings in transmission latency and battery energy, proving essential for practical underwater acoustic deployments."*

---

### Slide 8: Conclusion & Future Scope
- **Slide Title:** Conclusion
- **Conclusion:**
  - • Successfully engineered an autonomous multi-robot cyber-physical harbor traffic management system in a 10 km × 10 km domain.
  - • Multi-task PyTorch `VesselMLP` reliably classifies 6 ship classes (95.8% accuracy) and estimates displacement weight from 23 passive acoustic telemetry features.
  - • Autonomous AUV swarm dynamically preserves network connectivity, achieving >92% PDR even under severe monsoon noise (+26 dB) and 70/30 sensor sleep duty cycles.
  - • Binary telemetry frame compression achieves a 93.75% reduction in acoustic transmission delay and modem battery consumption.
  - • Autonomous traffic governance enforces harbor speed limits, screens vessels at the Line of Control, and automatically allocates designated dock berths.
- **Future Scope:**
  - • **Physical Hardware Validation:** Deploy on marine testbeds using physical acoustic modems (EvoLogics / Subnero).
  - • **Multi-Agent Reinforcement Learning (MARL):** Implement cooperative swarm pathfinding, collision avoidance, and energy-optimal shadowing.
  - • **Heterogeneous Cross-Domain Swarms:** Integrate Autonomous Surface Vessels (ASVs) and aerial patrol drones (UAVs) for hybrid air-sea tracking.
  - • **Adaptive Acoustic Modulation:** Dynamic physical-layer OFDM bitrate switching based on instantaneous SNR and Doppler feedback.
  - • **Edge Deployment:** Port lightweight quantized models onto embedded AUV companion computers (NVIDIA Jetson Orin Nano).

> **🎤 Speaker Notes (Slide 8):**  
> *"In conclusion, this project demonstrates how integrating robotics with underwater acoustic physics and deep learning overcomes the critical bottlenecks of maritime traffic control. Our multi-robot swarm guarantees communication resilience, enables passive vessel classification, and automates harbor safety governance. For future work, we plan to validate this on physical subsea modems, train multi-agent reinforcement learning policies for cooperative shadowing, and integrate aerial UAVs for cross-domain surveillance."*

---

### Slide 9: References
- **Slide Title:** References
- **Citations:**
  - [1] Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans. *The Journal of the Acoustical Society of America*, 70(3), 807–812.
  - [2] Ross, D. (1976). *Mechanics of Underwater Noise*. Pergamon Press, New York.
  - [3] Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra and sources. *The Journal of the Acoustical Society of America*, 34(12), 1936–1956.
  - [4] Yan, H., Shi, Z. J., & Cui, J. H. (2008). DBR: Depth-based routing for underwater sensor networks. *IFIP International Conference on Research in Networking*, 283–294. Springer.
  - [5] Akyildiz, I. F., Pompili, D., & Melodia, T. (2005). Underwater acoustic sensor networks: research challenges. *Ad Hoc Networks*, 3(3), 257–279.
  - [6] Thorp, W. H. (1967). Analytic description of the low-frequency attenuation coefficient. *The Journal of the Acoustical Society of America*, 42(1), 270–270.

> **🎤 Speaker Notes (Slide 9):**  
> *"Our theoretical framework is grounded in established literature in underwater acoustics and robotics: Mackenzie's nine-term sound speed equation, Donald Ross's propeller source noise formulations, Wenz's ambient ocean noise models, and Yan & Cui's seminal Depth-Based Routing protocol."*

---

### Slide 10: Thank You
- **Top Banner:** Amrita Vishwa Vidyapeetham, Amaravati Campus
- **Title:** Thank You
- **Quote:** *"Transforming passive underwater sensor grids into intelligent, adaptive robotic swarms — bringing resilient acoustic surveillance and autonomous traffic governance to maritime harbors."*
- **Presenters:** M Ganesh | J Kiran | P Rahul
- **Footer:** 04-10-2026 | INTRODUCTION TO AI ROBOTICS (22AIE214) | Slide 10

> **🎤 Speaker Notes (Slide 10):**  
> *"Thank you Dr. Ravishankar Prakash Desai and the evaluation panel for your time and guidance. We are now open for any questions and discussion."*

---

## ❓ Frequently Asked Viva / Q&A Questions & Model Answers

1. **Q: Why did you choose Depth-Based Routing (DBR) instead of classic AODV or geographic routing?**  
   *A: Underwater, GPS signals attenuate within a few centimeters and cannot provide 3D node coordinates. Traditional ad-hoc routing protocols like AODV flood route discovery packets, incurring unacceptable latency and draining limited battery reserves in acoustic channels (speed of sound is only 1500 m/s). DBR requires only local depth sensor measurements (which are cheap and readily available via pressure sensors), greedily forwarding packets to shallower neighbors toward surface gateway buoys.*

2. **Q: What role does the AUV swarm play that fixed sensors cannot accomplish?**  
   *A: Fixed sensors are tethered to the seabed and cannot reposition when acoustic dead zones occur or when nearby sensors sleep (under the 70/30 duty cycle). The 8 mobile AUVs actively patrol sectors, act as data mules to bridge fragmented networks, physically intercept and shadow overspeed violators, and return autonomously to surface buoys to recharge when battery drops below 15%.*

3. **Q: How does the Monsoon scenario affect the acoustic physics and routing?**  
   *A: In the Monsoon scenario, sea states rise to Douglas 6–8 with gale winds and torrential rain. Using the Wenz ambient ocean noise model, ambient noise surges by +26.1 dB (from 65.8 dB to 91.9 dB). Because modem SNR drops below the 10 dB decoding threshold at longer distances, direct long-range links fail. DBR automatically adapts by finding multi-hop paths through mobile AUV relays, increasing the average hop count from 1.4 to 2.3 hops while preserving a 92.4% Packet Delivery Ratio.*

4. **Q: Why use a Multi-Task model instead of two separate neural networks?**  
   *A: A single vessel's radiated noise simultaneously reveals both its engine class and its physical mass displacement. By sharing 3 latent representation layers (128 → 64 → 32), `VesselMLP` learns joint acoustic embeddings, reduces total model parameters, and executes in under 0.8 milliseconds, allowing real-time inference during 3D simulation frames without thread blocking.*
