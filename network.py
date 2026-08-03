"""
Underwater communication network — Stage 1.

Sensors are passive (no modem of their own — matches the project brief: they
only "receive/detect", they don't relay). AUVs carry acoustic modems and can
talk to nearby sensors, to each other, and to the fixed surface buoy. A link
exists only if it is physically plausible: within hardware range AND the
acoustic SNR (reusing the same transmission-loss / ambient-noise model
simulator.py already uses for vessel detection) clears a threshold — so
topology genuinely reflects distance, attenuation and noise, not just a
radius check.

Routing here is per-frame shortest-path over whatever the current topology
allows. Stage 2 replaces this with a churn-aware adaptive routing table and
a swarm controller that actively repositions AUVs to preserve connectivity.
"""
import numpy as np
import networkx as nx

import simulator as sim

SNR_THRESHOLD = sim.LINK_SNR_THRESHOLD_DB
MAX_RANGE     = sim.AUV_COMM_RANGE_M
COMM_FREQ     = sim.AUV_COMM_FREQ_HZ
COMM_SL       = sim.AUV_COMM_SOURCE_LEVEL


def _link_snr(dist_m):
    """SNR of a modem-to-modem link at this range (dB)."""
    dist_m = max(dist_m, 1.0)
    tl = sim.calculate_transmission_loss(dist_m, COMM_FREQ)
    received = COMM_SL - tl
    noise = sim.calculate_ambient_noise(sea_state=3, wind_speed_knots=10,
                                        shipping_density=0.4)
    return sim.calculate_snr(received, noise)


def build_network_graph(sensors, auv_states, buoy_pos=None):
    """
    Build this frame's comm graph.
    Nodes: sensor_<i> (0-indexed), auv_<id>, buoy.
    Edge added only if within MAX_RANGE and SNR >= SNR_THRESHOLD.
    """
    if buoy_pos is None:
        buoy_pos = sim.BUOY_POS

    G = nx.Graph()
    for i in range(len(sensors)):
        G.add_node(f"sensor_{i}", kind="sensor")
    for a in auv_states:
        G.add_node(f"auv_{a['id']}", kind="auv")
    G.add_node("buoy", kind="buoy")

    def _maybe_link(na, nb, pa, pb):
        d = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
        if d <= MAX_RANGE:
            snr = _link_snr(d)
            if snr >= SNR_THRESHOLD:
                G.add_edge(na, nb, weight=1.0, dist=d, snr=snr)

    # sensor <-> AUV
    for i, spos in enumerate(sensors):
        for a in auv_states:
            _maybe_link(f"sensor_{i}", f"auv_{a['id']}", spos, a["pos"])

    # AUV <-> AUV
    for i in range(len(auv_states)):
        for j in range(i + 1, len(auv_states)):
            _maybe_link(f"auv_{auv_states[i]['id']}", f"auv_{auv_states[j]['id']}",
                        auv_states[i]["pos"], auv_states[j]["pos"])

    # AUV <-> buoy
    for a in auv_states:
        _maybe_link(f"auv_{a['id']}", "buoy", a["pos"], buoy_pos)

    return G


def route_detections(active_sensor_indices, graph):
    """
    For each sensor that produced a detection this frame, attempt to route it
    to the buoy through the current topology.

    Returns (delivered, undelivered, paths):
      delivered   -- sensor indices with a live path to the buoy this frame
      undelivered -- sensor indices with no current path (dropped this frame)
      paths       -- {sensor_index: [node, node, ..., "buoy"]} for visualization
    """
    delivered, undelivered, paths = [], [], {}
    for i in active_sensor_indices:
        node = f"sensor_{i}"
        if node not in graph:
            undelivered.append(i)
            continue
        try:
            path = nx.shortest_path(graph, node, "buoy")
            delivered.append(i)
            paths[i] = path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            undelivered.append(i)
    return delivered, undelivered, paths


def edge_list_for_viz(graph):
    """Flat list of (kind_a, kind_b, dist, snr) tuples for rendering link lines."""
    out = []
    for u, v, data in graph.edges(data=True):
        out.append((u, v, data.get("dist", 0.0), data.get("snr", 0.0)))
    return out
