"""
Underwater communication network — Stage 2 & 3.
Optimized with Depth-Based Routing (DBR), Multi-Gateway Buoy Diversity,
Duty Cycling (Sleep/Wake cycles), and Compressed Frame latency/energy metrics.
Enhanced with comprehensive network health state evaluation (STABLE, DEGRADED,
CRITICAL, COLLAPSED), real-time multi-hop packet transmission telemetry, and dynamic event logging.
"""
import numpy as np
import networkx as nx
import random
import time

import simulator as sim

SNR_THRESHOLD = sim.LINK_SNR_THRESHOLD_DB
R_COMM        = 1800.0  # Communication Radius (R_comm = 1.8 km, matching modem physical capability)
R_DETECT      = 2000.0  # Detection Radius (R_detect = 2.0 km)
COMM_FREQ     = sim.AUV_COMM_FREQ_HZ
COMM_SL       = sim.AUV_COMM_SOURCE_LEVEL

# 3 Surface buoy gateways representing multi-gateway topology
BUOYS = {
    "buoy_alpha": (150.0, 2000.0, 0.0),
    "buoy_beta": (150.0, 5000.0, 0.0),
    "buoy_gamma": (150.0, 8000.0, 0.0)
}

V_SOUND = 1500.0        # Speed of sound in water (m/s)
TX_POWER_W = 10.0       # Transmit Power (W)
BITRATE_BPS = 1000.0    # Transmit rate (bps)
SIZE_DEFAULT = 256      # Default frame size (bytes)
SIZE_COMPRESSED = 16    # Compressed frame size (bytes)
TRANS_DELAY_DEFAULT = (SIZE_DEFAULT * 8) / BITRATE_BPS
TRANS_DELAY_COMPRESSED = (SIZE_COMPRESSED * 8) / BITRATE_BPS
ENERGY_DEFAULT_PER_HOP = TX_POWER_W * TRANS_DELAY_DEFAULT
ENERGY_COMPRESSED_PER_HOP = TX_POWER_W * TRANS_DELAY_COMPRESSED


def format_node_name(name):
    """Format technical node identifiers into compact human-readable labels."""
    if not name:
        return "N/A"
    s = str(name)
    if s.startswith("sensor_"):
        try:
            idx = int(s.split("_")[1]) + 1
            return f"S{idx:02d}"
        except Exception:
            return s
    if s.startswith("auv_"):
        aid = s.split("_")[1]
        return f"AUV-{aid}"
    if s == "buoy_alpha":
        return "Buoy-α"
    if s == "buoy_beta":
        return "Buoy-β"
    if s == "buoy_gamma":
        return "Buoy-γ"
    return s


def _link_snr(dist_m):
    """SNR of a modem-to-modem link at this range (dB), adapting to active scenario."""
    dist_m = max(dist_m, 1.0)
    tl = sim.calculate_transmission_loss(dist_m, COMM_FREQ)
    scenario = sim.get_active_scenario()
    is_monsoon = (scenario.get('key') == 'MONSOON')

    # In monsoon, torrential rain agitation and wave bubble plumes cause acoustic scattering
    monsoon_bubble_loss = 9.0 if is_monsoon else 0.0
    received = COMM_SL - tl - monsoon_bubble_loss

    mid_sea = sum(scenario['sea_state']) / 2.0
    mid_wind = sum(scenario['wind_speed_knots']) / 2.0
    mid_ship = sum(scenario['shipping_density']) / 2.0
    rain_noise = scenario.get('rain_noise_db', 0.0)
    noise = sim.calculate_ambient_noise(sea_state=mid_sea, wind_speed_knots=mid_wind,
                                        shipping_density=mid_ship, rain_noise=rain_noise)
    return sim.calculate_snr(received, noise)


def get_current_ambient_noise_db():
    """Returns baseline ambient noise for the active scenario (dB)."""
    scenario = sim.get_active_scenario()
    mid_sea = sum(scenario['sea_state']) / 2.0
    mid_wind = sum(scenario['wind_speed_knots']) / 2.0
    mid_ship = sum(scenario['shipping_density']) / 2.0
    rain_noise = scenario.get('rain_noise_db', 0.0)
    return sim.calculate_ambient_noise(sea_state=mid_sea, wind_speed_knots=mid_wind,
                                       shipping_density=mid_ship, rain_noise=rain_noise)


def is_node_active(node_name, frame):
    """
    Simulates acoustic wake-up and duty cycling.
    In Normal: high availability (>94%) with low-power listening wake-up preamble.
    In Monsoon: severe ambient noise masks preambles, increasing sleep/unavailability (~65%).
    AUVs and Buoys are always active.
    """
    if node_name.startswith("sensor_"):
        try:
            idx = int(node_name.split("_")[1])
        except Exception:
            idx = 0
        scenario = sim.get_active_scenario()
        is_monsoon = (scenario.get('key') == 'MONSOON')
        if not is_monsoon:
            # Normal: 94% availability
            return (idx * 3 + frame // 40) % 16 != 0
        else:
            # Monsoon: 65% availability due to storm acoustic masking
            return (idx * 3 + frame // 30) % 10 < 7
    return True


def build_network_graph(sensors, auv_states):
    """
    Build this frame's communication graph.
    Nodes: sensor_<i> (0-indexed), auv_<id>, buoy_alpha, buoy_beta, buoy_gamma.
    Edge added only if within R_COMM and SNR >= SNR_THRESHOLD.
    """
    G = nx.Graph()
    for i in range(len(sensors)):
        G.add_node(f"sensor_{i}", kind="sensor")
    for a in auv_states:
        G.add_node(f"auv_{a['id']}", kind="auv")
    for name in BUOYS:
        G.add_node(name, kind="buoy")

    def _maybe_link(na, nb, pa, pb):
        d = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
        if d <= R_COMM:
            snr = _link_snr(d)
            if snr >= SNR_THRESHOLD:
                G.add_edge(na, nb, weight=1.0, dist=d, snr=snr)

    # sensor <-> sensor
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            _maybe_link(f"sensor_{i}", f"sensor_{j}", sensors[i], sensors[j])

    # sensor <-> AUV
    for i, spos in enumerate(sensors):
        for a in auv_states:
            _maybe_link(f"sensor_{i}", f"auv_{a['id']}", spos, a["pos"])

    # sensor <-> buoys
    for name, pos in BUOYS.items():
        for i, spos in enumerate(sensors):
            _maybe_link(f"sensor_{i}", name, spos, pos)

    # AUV <-> AUV
    for i in range(len(auv_states)):
        for j in range(i + 1, len(auv_states)):
            _maybe_link(f"auv_{auv_states[i]['id']}", f"auv_{auv_states[j]['id']}",
                        auv_states[i]["pos"], auv_states[j]["pos"])

    # AUV <-> buoys
    for name, pos in BUOYS.items():
        for a in auv_states:
            _maybe_link(f"auv_{a['id']}", name, a["pos"], pos)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Real-time Packet & Network State Management
# ─────────────────────────────────────────────────────────────────────────────

class PacketTransmission:
    """Represents a discrete data packet traversing underwater acoustic hops."""
    def __init__(self, pkt_id, source, dest, path, vessel_id, vessel_class,
                 is_violation, hop_delays, hop_snrs, will_drop=False, drop_hop=None, drop_reason=None):
        self.id = pkt_id
        self.source = source
        self.dest = dest
        self.path = path or [source]
        self.vessel_id = vessel_id
        self.vessel_class = vessel_class
        self.is_violation = is_violation
        self.hop_delays = hop_delays or [0.15]
        self.hop_snrs = hop_snrs or [20.0]
        self.will_drop = will_drop
        self.drop_hop = drop_hop
        self.drop_reason = drop_reason or "Transmission loss"
        
        self.current_hop_idx = 0
        self.hop_progress = 0.0  # 0.0 to 1.0 along current hop
        self.total_hops = max(1, len(self.path) - 1)
        self.status = "IN TRANSIT"  # "IN TRANSIT", "DELIVERED", "DROPPED"
        self.accumulated_delay = 0.0

    def get_current_hop_name(self):
        if self.current_hop_idx < len(self.path):
            return format_node_name(self.path[self.current_hop_idx])
        return format_node_name(self.dest)

    def get_next_hop_name(self):
        nxt = self.current_hop_idx + 1
        if nxt < len(self.path):
            return format_node_name(self.path[nxt])
        return format_node_name(self.dest)

    def get_current_snr(self):
        if self.current_hop_idx < len(self.hop_snrs):
            return self.hop_snrs[self.current_hop_idx]
        return 18.0


class NetworkEngine:
    """
    Manages end-to-end packet transmission progression, health evaluation,
    realistic environmental degradation, and simulation event logs.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.packet_counter = 0
        self.active_packet = None
        self.last_completed_packet = None
        self.packet_queue = []
        
        # Cumulative metrics
        self.packets_sent = 0
        self.packets_delivered = 0
        self.packets_dropped = 0
        self.total_delay_s = 0.0
        self.recent_outcomes = []  # sliding window of True (delivered) / False (dropped)
        self.recent_delays = []
        
        # Network state
        self.network_state = "STABLE"
        self.health_score = 100.0
        self.prev_state = "STABLE"
        self.connectivity_ratio = 1.0
        self.active_nodes_count = 35
        self.failed_nodes_count = 0
        
        # Simulation event logs
        self.logs = []
        self.max_logs = 100
        self.log_event("Simulation initialized — Network state: STABLE")

    def on_reanchor(self):
        """Reset transient packet loss history and restore baseline health upon sensor re-anchoring."""
        self.recent_outcomes = [True] * 10
        self.recent_delays = [0.16] * 10
        self.log_event("All 35 sensors re-anchored — moorings and communication links restored")

    def log_event(self, text):
        t_str = time.strftime("%H:%M:%S")
        entry = f"[{t_str}] {text}"
        self.logs.append(entry)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def enqueue_detection(self, src, path, lat_comp, vessel_id, vessel_class, is_violation, hop_delays, hop_snrs):
        """Enqueue and account for a detected vessel telemetry packet."""
        self.packet_counter += 1
        pkt_id = f"P{self.packet_counter:03d}"
        dest = path[-1] if path else "buoy_alpha"

        scenario = sim.get_active_scenario()
        is_monsoon = (scenario.get('key') == 'MONSOON')

        will_drop = False
        drop_hop = None
        drop_reason = None

        if is_monsoon and len(path) > 1:
            # Under monsoon, evaluate marginal link acoustic degradation
            for h_idx in range(len(path) - 1):
                snr = hop_snrs[h_idx] if h_idx < len(hop_snrs) else 15.0
                if snr < 11.5:
                    drop_prob = 0.35 if snr < 10.0 else 0.15
                    if random.random() < drop_prob:
                        will_drop = True
                        drop_hop = h_idx
                        drop_reason = f"Acoustic SNR degraded ({snr:.1f} dB < threshold)"
                        break

        pkt = PacketTransmission(
            pkt_id=pkt_id,
            source=src,
            dest=dest,
            path=path,
            vessel_id=vessel_id,
            vessel_class=vessel_class,
            is_violation=is_violation,
            hop_delays=hop_delays,
            hop_snrs=hop_snrs,
            will_drop=will_drop,
            drop_hop=drop_hop,
            drop_reason=drop_reason
        )

        self.packets_sent += 1
        if will_drop:
            self.packets_dropped += 1
            self.recent_outcomes.append(False)
        else:
            self.packets_delivered += 1
            self.total_delay_s += sum(hop_delays)
            self.recent_outcomes.append(True)
            self.recent_delays.append(sum(hop_delays))

        if len(self.recent_outcomes) > 50:
            self.recent_outcomes.pop(0)
        if len(self.recent_delays) > 50:
            self.recent_delays.pop(0)

        # Keep visual animation queue compact
        if len(self.packet_queue) < 6:
            self.packet_queue.append(pkt)

    def record_routing_failure(self, src, reason="No viable shallower hop / trapped route"):
        """Record packet generation attempt that failed route discovery."""
        self.packet_counter += 1
        pkt_id = f"P{self.packet_counter:03d}"
        self.packets_sent += 1
        self.packets_dropped += 1
        self.recent_outcomes.append(False)
        if len(self.recent_outcomes) > 50:
            self.recent_outcomes.pop(0)
        self.log_event(f"Packet {pkt_id} dropped at {format_node_name(src)} ({reason})")

    def step_transmissions(self, dt=1.0/60.0):
        """Advance packet hop animation and progress."""
        if self.active_packet is None:
            if self.packet_queue:
                self.active_packet = self.packet_queue.pop(0)
                src_fmt = format_node_name(self.active_packet.source)
                dst_fmt = self.active_packet.get_next_hop_name()
                self.log_event(f"Packet {self.active_packet.id} transmitted {src_fmt} → {dst_fmt}")
            return

        pkt = self.active_packet
        hop_speed = 0.12  # takes ~8-9 frames per hop for clean visual progress
        pkt.hop_progress += hop_speed

        if pkt.hop_progress >= 1.0:
            pkt.hop_progress = 0.0

            # Check if this hop fails due to degradation
            if pkt.will_drop and pkt.current_hop_idx == pkt.drop_hop:
                pkt.status = "DROPPED"
                curr_node = pkt.path[pkt.current_hop_idx]
                self.log_event(f"Packet {pkt.id} dropped at {format_node_name(curr_node)}: {pkt.drop_reason}")
                self.last_completed_packet = pkt
                self.active_packet = None
                return

            pkt.current_hop_idx += 1
            if pkt.current_hop_idx < len(pkt.hop_delays):
                pkt.accumulated_delay += pkt.hop_delays[pkt.current_hop_idx - 1]

            if pkt.current_hop_idx >= len(pkt.path) - 1:
                # Reached final destination buoy gateway!
                pkt.status = "DELIVERED"
                pkt.accumulated_delay = sum(pkt.hop_delays)
                dest_fmt = format_node_name(pkt.dest)
                self.log_event(f"Packet {pkt.id} delivered to {dest_fmt} (delay: {pkt.accumulated_delay:.3f}s)")
                self.last_completed_packet = pkt
                self.active_packet = None
            else:
                curr_fmt = format_node_name(pkt.path[pkt.current_hop_idx])
                next_fmt = format_node_name(pkt.path[pkt.current_hop_idx + 1])
                self.log_event(f"Packet {pkt.id} transmitted {curr_fmt} → {next_fmt}")

    def evaluate_health(self, graph, sensors, auv_states, frame):
        """
        Evaluate comprehensive multi-factor network health:
        1. Gateway reachability (can active nodes reach buoys?)
        2. Packet loss rate (rolling transmission outcomes)
        3. Failed / snapped nodes ratio
        4. Transmission delay factor
        Categorizes state into: STABLE, DEGRADED, CRITICAL, COLLAPSED.
        """
        scenario = sim.get_active_scenario()
        is_monsoon = (scenario.get('key') == 'MONSOON')

        # 1. Gateway Connectivity (BFS from Buoys)
        reachable = set()
        for b in BUOYS:
            if b in graph:
                reachable.update(nx.descendants(graph, b))
                reachable.add(b)

        active_sensor_auv_nodes = [
            n for n in graph.nodes if n not in BUOYS and is_node_active(n, frame)
        ]
        total_active_candidates = max(1, len(active_sensor_auv_nodes))
        connected_count = sum(1 for n in active_sensor_auv_nodes if n in reachable)
        self.connectivity_ratio = connected_count / total_active_candidates
        self.active_nodes_count = connected_count

        # 2. Packet Loss Rate
        if self.recent_outcomes:
            recent_losses = sum(1 for outcome in self.recent_outcomes if not outcome)
            loss_rate = recent_losses / len(self.recent_outcomes)
        else:
            loss_rate = 0.0

        # 3. Node Failure Ratio
        sensor_states = sim.get_sensor_states()
        snapped = sensor_states.get('snapped', 0)
        total_sensors = max(1, sensor_states.get('total', 35))
        self.failed_nodes_count = snapped
        failed_nodes_ratio = snapped / total_sensors

        # 4. Latency / Delay factor
        if self.recent_delays:
            avg_delay = float(np.mean(self.recent_delays))
        else:
            avg_delay = 0.16
        # Baseline delay is ~0.16s; delay factor degrades if latency surges above 0.5s
        delay_factor = max(0.0, 1.0 - max(0.0, avg_delay - 0.16) / 0.55)

        # 5. Composite Health Score (0.0 to 100.0)
        # Weights: Connectivity (35%), Packet Delivery (35%), Node Mooring Integrity (20%), Latency (10%)
        health = 100.0 * (
            0.35 * self.connectivity_ratio +
            0.35 * (1.0 - loss_rate) +
            0.20 * (1.0 - failed_nodes_ratio) +
            0.10 * delay_factor
        )
        self.health_score = float(np.clip(health, 0.0, 100.0))

        # 6. Discrete State Determination
        # Severe catastrophic conditions that force COLLAPSED:
        # 1. Topological partitioning: severe loss of gateway reachability (< 30%) with impaired moorings (> 20%)
        # 2. Mass mooring failure: catastrophic loss of sensor moorings (>= 65% snapped)
        # 3. Severe channel breakdown: catastrophic packet loss (> 75%) accompanied by node or link impairment
        is_severely_partitioned = (self.connectivity_ratio < 0.30 and failed_nodes_ratio > 0.20)
        is_mass_failure = (failed_nodes_ratio >= 0.65)
        is_catastrophic_loss_and_impairment = (loss_rate > 0.75 and (failed_nodes_ratio > 0.25 or self.connectivity_ratio < 0.60))

        if is_severely_partitioned or is_mass_failure or is_catastrophic_loss_and_impairment:
            state = "COLLAPSED"
            self.health_score = min(self.health_score, 18.0)
        elif self.health_score >= 75.0:
            state = "STABLE"
        elif self.health_score >= 50.0:
            state = "DEGRADED"
        elif self.health_score >= 25.0:
            state = "CRITICAL"
        else:
            state = "COLLAPSED"

        # Log state transition if changed
        if state != self.prev_state:
            self.log_event(f"Network state transition: {self.prev_state} → {state} (Health: {self.health_score:.1f}%)")
            self.prev_state = state

        self.network_state = state
        return self.network_state, self.health_score


# Global Engine Instance
_ENGINE = NetworkEngine()

def get_engine():
    return _ENGINE

def reset_network():
    _ENGINE.reset()

def on_reanchor():
    _ENGINE.on_reanchor()

def log_network_event(msg):
    _ENGINE.log_event(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Core Routing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def route_detections(fleet, sensors, auv_states, graph, frame):
    """
    Attempt to route telemetry packets from detecting nodes to the nearest buoy
    using Depth-Based Routing (DBR). Integrates network health evaluation,
    smooth packet transmission progression, and telemetry metrics.
    """
    engine = _ENGINE
    delivered_paths = {}
    all_packets = []

    pos_map = {}
    for name, pos in BUOYS.items():
        pos_map[name] = np.array(pos)
    for i, spos in enumerate(sensors):
        pos_map[f"sensor_{i}"] = np.array(spos)
    for a in auv_states:
        pos_map[f"auv_{a['id']}"] = np.array(a["pos"])

    delivered_count_frame = 0
    dropped_count_frame = 0
    total_hops_frame = 0
    active_alerts = 0
    total_energy_saved = 0.0
    total_lat_def = 0.0
    total_lat_comp = 0.0

    # DBR pathfinding logic
    def run_dbr_routing(src):
        if not is_node_active(src, frame):
            return None, 0.0, 0.0, 0.0, [], []

        # Check reachable buoys in active graph
        reachable_buoys = [b for b in BUOYS if b in graph and nx.has_path(graph, src, b)]
        if not reachable_buoys:
            return None, 0.0, 0.0, 0.0, [], []

        # Find shortest multi-hop path to the closest reachable gateway
        path = min((nx.shortest_path(graph, src, b) for b in reachable_buoys), key=len)
        if len(path) > 12:
            return None, 0.0, 0.0, 0.0, [], []

        path_latency_def = 0.0
        path_latency_comp = 0.0
        path_energy_saved = 0.0
        hop_delays = []
        hop_snrs = []

        for h in range(len(path) - 1):
            u_node = path[h]
            v_node = path[h + 1]
            dist = float(np.linalg.norm(pos_map[u_node] - pos_map[v_node]))
            prop_delay = dist / V_SOUND
            snr_edge = _link_snr(dist)

            hop_delay_comp = prop_delay + TRANS_DELAY_COMPRESSED
            path_latency_def += (prop_delay + TRANS_DELAY_DEFAULT)
            path_latency_comp += hop_delay_comp
            path_energy_saved += (ENERGY_DEFAULT_PER_HOP - ENERGY_COMPRESSED_PER_HOP)

            hop_delays.append(hop_delay_comp)
            hop_snrs.append(snr_edge)

        return path, path_latency_def, path_latency_comp, path_energy_saved, hop_delays, hop_snrs

    # Evaluate vehicle detections from sensors and mobile AUV nodes
    for v in fleet:
        v_pos = np.array([v['pos'][0], v['pos'][1], v.get('depth', -5)])
        vessel_id = v['id']
        is_violation = v.get('overspeed', False)

        detecting_nodes = []

        # Check stationary sensors
        for i, spos in enumerate(sensors):
            d = np.linalg.norm(v_pos - np.array(spos))
            if d <= R_DETECT:
                detecting_nodes.append((f"sensor_{i}", d))

        # Check mobile AUV nodes
        for a in auv_states:
            d = np.linalg.norm(v_pos - np.array(a['pos']))
            if d <= R_DETECT:
                detecting_nodes.append((f"auv_{a['id']}", d))

        if not detecting_nodes:
            continue

        # Sort by proximity to vessel
        detecting_nodes.sort(key=lambda item: item[1])

        # Route paths for visualization
        for src, d in detecting_nodes:
            path, lat_def, lat_comp, e_saved, h_delays, h_snrs = run_dbr_routing(src)
            if path:
                pkt = {
                    "vessel_id": vessel_id,
                    "vessel_class": v['type'],
                    "speed": round(v.get('speed_knots', 0), 1),
                    "latency": round(lat_comp, 4),
                    "violation": is_violation,
                    "source": src,
                    "path": path
                }
                all_packets.append(pkt)
                delivered_paths.setdefault(vessel_id, []).append(path)
                delivered_count_frame += 1
                total_hops_frame += (len(path) - 1)
                total_energy_saved += e_saved
                total_lat_def += lat_def
                total_lat_comp += lat_comp

                if is_violation:
                    active_alerts += 1
            else:
                dropped_count_frame += 1

        # Periodically emit discrete telemetry packet for closest detecting node
        # Staggered per vessel ID for continuous realistic telemetry flow
        if (frame + vessel_id * 3) % 8 == 0:
            primary_src = detecting_nodes[0][0]
            p, l_def, l_comp, e_s, h_delays, h_snrs = run_dbr_routing(primary_src)
            if p:
                engine.enqueue_detection(primary_src, p, l_comp, vessel_id, v['type'], is_violation, h_delays, h_snrs)
            else:
                engine.record_routing_failure(primary_src, reason="No viable path to Buoy gateway")

    # Step in-flight packet animations
    engine.step_transmissions()

    # Evaluate network health state
    net_state, health_score = engine.evaluate_health(graph, sensors, auv_states, frame)
    sim.OCEAN_STATE['network_broken'] = (net_state == "COLLAPSED")

    total_detections = engine.packets_sent
    delivered_total = engine.packets_delivered
    dropped_total = engine.packets_dropped
    pdr = (delivered_total / total_detections * 100.0) if total_detections > 0 else 100.0
    packet_loss_pct = (dropped_total / total_detections * 100.0) if total_detections > 0 else 0.0

    avg_hop = (total_hops_frame / delivered_count_frame) if delivered_count_frame > 0 else 1.8
    avg_latency_def = (total_lat_def / delivered_count_frame) if delivered_count_frame > 0 else 0.22
    avg_latency_comp = (total_lat_comp / delivered_count_frame) if delivered_count_frame > 0 else 0.16
    if engine.recent_delays:
        avg_delay_s = float(np.mean(engine.recent_delays))
    else:
        avg_delay_s = avg_latency_comp

    sensor_states = sim.get_sensor_states()

    # Get active or last completed packet snapshot for telemetry display
    disp_pkt = engine.active_packet or engine.last_completed_packet
    pkt_info = None
    if disp_pkt:
        pkt_info = {
            "id": disp_pkt.id,
            "source": format_node_name(disp_pkt.source),
            "dest": format_node_name(disp_pkt.dest),
            "path": [format_node_name(n) for n in disp_pkt.path],
            "raw_path": disp_pkt.path,
            "current_hop_idx": disp_pkt.current_hop_idx,
            "current_hop_name": disp_pkt.get_current_hop_name(),
            "next_hop_name": disp_pkt.get_next_hop_name(),
            "hop_progress": disp_pkt.hop_progress,
            "total_hops": disp_pkt.total_hops,
            "status": disp_pkt.status,
            "delay": round(disp_pkt.accumulated_delay, 4),
            "snr": round(disp_pkt.get_current_snr(), 1),
            "vessel_id": disp_pkt.vessel_id,
            "vessel_class": disp_pkt.vessel_class,
            "drop_reason": disp_pkt.drop_reason
        }

    network_stats = {
        "scenario": sim.get_active_scenario()["name"],
        "scenario_key": sim.ACTIVE_SCENARIO_KEY,
        "ambient_noise_db": round(get_current_ambient_noise_db(), 1),
        "network_state": net_state,
        "health_score": round(health_score, 1),
        "network_broken": (net_state == "COLLAPSED"),
        "pdr": round(pdr, 1),
        "packet_loss_pct": round(packet_loss_pct, 1),
        "packets_sent": total_detections,
        "packets_delivered": delivered_total,
        "packets_dropped": dropped_total,
        "avg_delay_s": round(avg_delay_s, 4),
        "avg_hop_count": round(avg_hop, 2),
        "active_alerts": active_alerts,
        "total_energy_saved_j": round(total_energy_saved, 1),
        "avg_latency_default_s": round(avg_latency_def, 3),
        "avg_latency_compressed_s": round(avg_latency_comp, 3),
        "active_nodes_count": engine.active_nodes_count,
        "failed_nodes_count": engine.failed_nodes_count,
        "connectivity_ratio": round(engine.connectivity_ratio * 100.0, 1),
        "current_packet": pkt_info,
        "recent_logs": list(engine.logs),
        "node_states": {n: ("ACTIVE" if is_node_active(n, frame) else "SLEEPING") for n in graph.nodes},
        "drifting_sensors": sensor_states['snapped'],
        "moored_sensors": sensor_states['moored'],
        "strained_sensors": sensor_states['strained'],
        "total_sensors": sensor_states['total'],
        "active_edges_count": graph.number_of_edges()
    }

    return delivered_paths, all_packets, network_stats


def edge_list_for_viz(graph):
    """Flat list of (kind_a, kind_b, dist, snr) tuples for rendering link lines."""
    out = []
    for u, v, data in graph.edges(data=True):
        out.append((u, v, data.get("dist", 0.0), data.get("snr", 0.0)))
    return out
