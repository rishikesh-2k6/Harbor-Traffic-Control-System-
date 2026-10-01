"""
Underwater communication network — Stage 2 & 3.
Optimized with Depth-Based Routing (DBR), Multi-Gateway Buoy Diversity,
Duty Cycling (Sleep/Wake cycles), and Compressed Frame latency/energy metrics.
"""
import numpy as np
import networkx as nx

import simulator as sim

SNR_THRESHOLD = sim.LINK_SNR_THRESHOLD_DB
R_COMM        = 1500.0  # Communication Radius (R_comm = 1.5 km)
R_DETECT      = 2000.0  # Detection Radius (R_detect = 2.0 km)
COMM_FREQ     = sim.AUV_COMM_FREQ_HZ
COMM_SL       = sim.AUV_COMM_SOURCE_LEVEL

# 3 Surface buoy gateways representing multi-gateway topology
BUOYS = {
    "buoy_alpha": (150.0, 2000.0, 0.0),
    "buoy_beta": (150.0, 5000.0, 0.0),
    "buoy_gamma": (150.0, 8000.0, 0.0)
}

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
    Simulates duty cycling sleep/wake states to conserve battery.
    Sensors have a 70% active, 30% sleeping duty cycle.
    """
    if node_name.startswith("sensor_"):
        try:
            idx = int(node_name.split("_")[1])
        except Exception:
            idx = 0
        # Duty cycle cycle: active for 35 frames, sleeping for 15 frames
        cycle = (idx + frame // 35) % 10
        return cycle < 7
    # AUVs and Buoy gateways are always awake
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

    # sensor <-> sensor (peer-to-peer acoustic links between deployed sensors)
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            _maybe_link(f"sensor_{i}", f"sensor_{j}", sensors[i], sensors[j])

    # sensor <-> AUV
    for i, spos in enumerate(sensors):
        for a in auv_states:
            _maybe_link(f"sensor_{i}", f"auv_{a['id']}", spos, a["pos"])

    # sensor <-> buoys (direct gateway link if within acoustic range)
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


def route_detections(fleet, sensors, auv_states, graph, frame):
    """
    Attempt to route telemetry packets from detecting nodes to the nearest buoy
    using Depth-Based Routing (DBR). Handles node duty cycles, and calculates
    compression metrics (Default vs Compressed).
    """
    delivered_paths = {}
    all_packets = []
    
    pos_map = {}
    for name, pos in BUOYS.items():
        pos_map[name] = np.array(pos)
    for i, spos in enumerate(sensors):
        pos_map[f"sensor_{i}"] = np.array(spos)
    for a in auv_states:
        pos_map[f"auv_{a['id']}"] = np.array(a["pos"])

    V_SOUND = 1500.0        # Speed of sound in water (m/s)
    TX_POWER_W = 10.0       # Transmit Power (W)
    BITRATE_BPS = 1000.0    # Transmit rate (bps)

    # Frame Sizes (bytes)
    SIZE_DEFAULT = 256
    SIZE_COMPRESSED = 16

    # Transmission Delays per hop (seconds)
    TRANS_DELAY_DEFAULT = (SIZE_DEFAULT * 8) / BITRATE_BPS
    TRANS_DELAY_COMPRESSED = (SIZE_COMPRESSED * 8) / BITRATE_BPS

    # Energy per hop (Joules)
    ENERGY_DEFAULT_PER_HOP = TX_POWER_W * TRANS_DELAY_DEFAULT
    ENERGY_COMPRESSED_PER_HOP = TX_POWER_W * TRANS_DELAY_COMPRESSED

    delivered_count = 0
    dropped_count = 0
    total_hops = 0
    active_alerts = 0
    total_energy_saved = 0.0

    total_lat_def = 0.0
    total_lat_comp = 0.0

    # DBR pathfinding logic
    def run_dbr_routing(src):
        # If the detecting source node is asleep, packet is instantly dropped
        if not is_node_active(src, frame):
            return None, 0.0, 0.0, 0.0

        path = [src]
        curr = src
        path_latency_def = 0.0
        path_latency_comp = 0.0
        path_energy_saved = 0.0

        while curr not in BUOYS:
            if curr not in graph:
                return None, 0.0, 0.0, 0.0

            # Find active neighbors
            active_nbrs = [v for v in graph.neighbors(curr) if is_node_active(v, frame)]
            if not active_nbrs:
                return None, 0.0, 0.0, 0.0

            # Check if any buoy is directly reachable
            buoy_nbrs = [v for v in active_nbrs if v in BUOYS]
            if buoy_nbrs:
                # Pick the closest buoy gateway
                next_hop = min(buoy_nbrs, key=lambda b: np.linalg.norm(pos_map[curr] - pos_map[b]))
            else:
                # Otherwise, select shallower neighbors (closer to depth 0, i.e., larger z)
                curr_z = pos_map[curr][2]
                shallower_nbrs = [v for v in active_nbrs if pos_map[v][2] > curr_z]
                if not shallower_nbrs:
                    return None, 0.0, 0.0, 0.0 # Trapped in depth local maximum

                # Greedy choice: pick neighbor closest to surface (highest z)
                next_hop = max(shallower_nbrs, key=lambda v: pos_map[v][2])

            if next_hop in path:
                # Loop detected
                return None, 0.0, 0.0, 0.0

            # Add to path & compute latency/energy
            dist = float(np.linalg.norm(pos_map[curr] - pos_map[next_hop]))
            prop_delay = dist / V_SOUND

            path_latency_def += prop_delay + TRANS_DELAY_DEFAULT
            path_latency_comp += prop_delay + TRANS_DELAY_COMPRESSED
            path_energy_saved += (ENERGY_DEFAULT_PER_HOP - ENERGY_COMPRESSED_PER_HOP)

            path.append(next_hop)
            curr = next_hop

            # Safeguard hop limit
            if len(path) > 12:
                return None, 0.0, 0.0, 0.0

        return path, path_latency_def, path_latency_comp, path_energy_saved

    for v in fleet:
        v_pos = np.array([v['pos'][0], v['pos'][1], v.get('depth', -5)])
        vessel_id = v['id']
        
        # Check stationary sensors
        for i, spos in enumerate(sensors):
            d = np.linalg.norm(v_pos - np.array(spos))
            if d <= R_DETECT:
                src = f"sensor_{i}"
                path, lat_def, lat_comp, e_saved = run_dbr_routing(src)
                if path:
                    pkt = {
                        "vessel_id": vessel_id,
                        "vessel_class": v['type'],
                        "speed": round(v.get('speed_knots', 0), 1),
                        "latency": round(lat_comp, 4), # report compressed latency by default
                        "violation": v.get('overspeed', False),
                        "source": src,
                        "path": path
                    }
                    all_packets.append(pkt)
                    delivered_paths.setdefault(vessel_id, []).append(path)
                    delivered_count += 1
                    total_hops += (len(path) - 1)
                    total_energy_saved += e_saved
                    total_lat_def += lat_def
                    total_lat_comp += lat_comp

                    if pkt["violation"]:
                        active_alerts += 1
                    
                    vio_str = "VIOLATION" if pkt["violation"] else "NORMAL"
                    print(f"[DBR Net] Frame {frame:04d} | Vessel {vessel_id} ({pkt['vessel_class']}) detected by {src} -> DBR Path: {path} | Latency: {lat_comp:.4f}s | Energy Saved: {e_saved:.1f}J")
                else:
                    dropped_count += 1

        # Check mobile AUV nodes
        for a in auv_states:
            d = np.linalg.norm(v_pos - np.array(a['pos']))
            if d <= R_DETECT:
                src = f"auv_{a['id']}"
                path, lat_def, lat_comp, e_saved = run_dbr_routing(src)
                if path:
                    pkt = {
                        "vessel_id": vessel_id,
                        "vessel_class": v['type'],
                        "speed": round(v.get('speed_knots', 0), 1),
                        "latency": round(lat_comp, 4),
                        "violation": v.get('overspeed', False),
                        "source": src,
                        "path": path
                    }
                    all_packets.append(pkt)
                    delivered_paths.setdefault(vessel_id, []).append(path)
                    delivered_count += 1
                    total_hops += (len(path) - 1)
                    total_energy_saved += e_saved
                    total_lat_def += lat_def
                    total_lat_comp += lat_comp

                    if pkt["violation"]:
                        active_alerts += 1
                    
                    vio_str = "VIOLATION" if pkt["violation"] else "NORMAL"
                    print(f"[DBR Net] Frame {frame:04d} | Vessel {vessel_id} ({pkt['vessel_class']}) detected by {src} -> DBR Path: {path} | Latency: {lat_comp:.4f}s | Energy Saved: {e_saved:.1f}J")
                else:
                    dropped_count += 1

    total_detections = delivered_count + dropped_count
    pdr = (delivered_count / total_detections) * 100.0 if total_detections > 0 else 100.0
    avg_hop = (total_hops / delivered_count) if delivered_count > 0 else 0.0
    
    avg_latency_def = (total_lat_def / delivered_count) if delivered_count > 0 else 0.0
    avg_latency_comp = (total_lat_comp / delivered_count) if delivered_count > 0 else 0.0

    sensor_states = sim.get_sensor_states()
    network_broken = (pdr < 30.0 and total_detections > 0) or (sensor_states['snapped'] >= 10)
    sim.OCEAN_STATE['network_broken'] = network_broken

    network_stats = {
        "scenario": sim.get_active_scenario()["name"],
        "scenario_key": sim.ACTIVE_SCENARIO_KEY,
        "ambient_noise_db": round(get_current_ambient_noise_db(), 1),
        "pdr": round(pdr, 1),
        "avg_hop_count": round(avg_hop, 2),
        "active_alerts": active_alerts,
        "total_energy_saved_j": round(total_energy_saved, 1),
        "avg_latency_default_s": round(avg_latency_def, 3),
        "avg_latency_compressed_s": round(avg_latency_comp, 3),
        "node_states": {n: ("ACTIVE" if is_node_active(n, frame) else "SLEEPING") for n in graph.nodes},
        "network_broken": network_broken,
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
