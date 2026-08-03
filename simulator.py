import numpy as np
import pandas as pd
import random
import os
import tempfile
import torch
import auv

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RHO             = 1025
G               = 9.81
SAFE_DISTANCE   = 450
SENSING_RANGE   = 2000.0        # metres — hydrophone detection radius (R_detect)
MAP_SIZE        = 10000         # 10 km x 10 km

# ── AUV network layer (Stage 1) ──────────────────────────────────────────────
NUM_AUVS               = 8
AUV_RADIUS_M            = 8.0     # collision-shape radius (visualization scale)
AUV_MASS_KG             = 500.0
AUV_COMM_RANGE_M        = 1800    # hard cutoff — beyond this, modems can't hear each other
AUV_COMM_FREQ_HZ        = 12000   # typical underwater acoustic modem carrier (~12 kHz)
AUV_COMM_SOURCE_LEVEL   = 178.0   # dB re 1 uPa @ 1 m — modem transmit power
LINK_SNR_THRESHOLD_DB   = 10.0    # minimum SNR for a "reliable" comm link
BUOY_POS                = (150.0, 5000.0, 0.0)   # fixed surface relay buoy near harbor mouth

# ── Zone Boundaries ──────────────────────────────────────────────────────────
LOC_X                   = 5000   # Line of Control — splits Outer Yard / Inner Zone
HARBOR_X                = 700    # ships docked inside this X are "in harbor"
OUTER_YARD_SPEED_LIMIT  = 10.0   # knots — max speed in the Outer Yard
INNER_ZONE_SPEED_LIMIT  = 6.0    # knots — max speed in the Inner / Patrol Zone

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
_TMP            = tempfile.gettempdir()

# The ONE growing CSV file that accumulates every real sensor detection
COLLECTED_FILE  = os.path.join(SCRIPT_DIR, "collected_data.csv")
# Live inference snapshot written by mlmodel.py
LIVE_FILE       = os.path.join(_TMP, "stage3_live_sensor_data.csv")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Physics Engine
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sound_speed(T, S, D):
    """Mackenzie (1981) — m/s."""
    return (1448.96 + 4.591*T - 5.304e-2*T**2 + 2.374e-4*T**3
            + 1.340*(S - 35) + 1.630e-2*D + 1.675e-7*D**2
            - 1.025e-2*T*(S - 35) - 7.139e-13*T*D**3)


def calculate_thorp_absorption(f_hz):
    """Thorp (1967) — dB/km."""
    f = f_hz / 1000.0
    return (0.11*f**2/(1+f**2) + 44*f**2/(4100+f**2)
            + 2.75e-4*f**2 + 0.003)


def calculate_transmission_loss(range_m, f_hz):
    """Spherical spreading + Thorp absorption — dB."""
    range_m = max(range_m, 1.0)
    return (20*np.log10(range_m)
            + calculate_thorp_absorption(f_hz) * (range_m / 1000.0))


def calculate_source_level(speed_knots, weight_kg, base_sl, prop_diameter):
    """Ross (1976) — dB re 1 uPa @ 1 m."""
    speed_knots = max(speed_knots, 1.0)
    return (base_sl
            + 55*np.log10(speed_knots / 10.0)
            + 20*np.log10(prop_diameter / 3.0)
            + random.gauss(0, 2))


def calculate_doppler_shift(f_source, v_radial, sound_speed):
    denom = sound_speed - v_radial
    if denom <= 0:
        denom = 1.0
    return f_source * (sound_speed / denom)


def calculate_ambient_noise(sea_state, wind_speed_knots, shipping_density):
    """Wenz (1962) incoherent combination — dB re 1 uPa."""
    wind_speed_knots = max(wind_speed_knots, 1.0)
    w  = 44 + 20*np.log10(wind_speed_knots)
    sh = 50 + 10*shipping_density + random.uniform(-3, 3)
    bio= random.uniform(35, 45)
    return 10*np.log10(10**(w/10) + 10**(sh/10) + 10**(bio/10))


def calculate_snr(received_level, noise_level):
    return received_level - noise_level


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Seabed terrain
# ─────────────────────────────────────────────────────────────────────────────

def get_seabed_depth(x, y):
    slope = -60 * (x / float(MAP_SIZE))
    wave  = 7*np.sin(x/1600.0) + 5*np.cos(y/1200.0)
    return np.minimum(slope + wave - 5, -2)


def get_current_vector(x, y, z, t):
    """
    Depth-dependent, time-varying water current field — m/s.

    Real harbor currents are strongest near the surface (wind + tidal driven)
    and weaken toward the seabed. Modeled as a slowly-oscillating tidal term
    plus large spatial gyres, scaled down with depth. Not a fluid solver —
    just enough structure that AUV drift is spatially and temporally coherent
    rather than pure noise.
    """
    depth_frac     = np.clip(-z / 60.0, 0.0, 1.0)     # 0 at surface, 1 near seabed
    surface_factor = 1.0 - 0.75 * depth_frac          # currents weaken with depth

    tphase = t / 400.0
    vx = surface_factor * (0.35*np.sin(y/2200.0 + tphase) + 0.15*np.cos(x/3000.0 - tphase*0.6))
    vy = surface_factor * (0.30*np.cos(x/2500.0 - tphase*0.8) + 0.12*np.sin(y/1800.0 + tphase*0.4))
    vz = 0.03*np.sin((x + y)/4000.0 + tphase*0.3)     # weak vertical heave

    return vx, vy, vz


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Vehicle profiles
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_TYPES = {
    'Cargo Ship': {
        'weight':             (80000, 200000),
        'speed':              (5, 12),
        'freq_hz':            (30, 60),
        'sl_base':            (185, 195),
        'propeller_diameter': (5.0, 8.0),
        'num_blades':         (4, 6),
        'draft':              (8, 14),
        'color': 'orange', 'marker': 's'
    },
    'Tanker': {
        'weight':             (150000, 400000),
        'speed':              (3, 8),
        'freq_hz':            (20, 45),
        'sl_base':            (190, 200),
        'propeller_diameter': (6.0, 10.0),
        'num_blades':         (4, 6),
        'draft':              (12, 20),
        'color': 'darkred', 'marker': 'D'
    },
    'Cruiser': {
        'weight':             (30000, 80000),
        'speed':              (8, 18),
        'freq_hz':            (200, 350),
        'sl_base':            (160, 175),
        'propeller_diameter': (2.5, 4.0),
        'num_blades':         (3, 5),
        'draft':              (4, 8),
        'color': 'green', 'marker': 'P'
    },
    'Ferry': {
        'weight':             (20000, 60000),
        'speed':              (12, 22),
        'freq_hz':            (150, 300),
        'sl_base':            (155, 170),
        'propeller_diameter': (2.0, 3.5),
        'num_blades':         (3, 5),
        'draft':              (3, 6),
        'color': 'purple', 'marker': 'h'
    },
    'Speedboat': {
        'weight':             (1000, 10000),
        'speed':              (20, 45),
        'freq_hz':            (1000, 2500),
        'sl_base':            (145, 165),
        'propeller_diameter': (0.3, 0.8),
        'num_blades':         (2, 4),
        'draft':              (0.5, 2),
        'color': 'blue', 'marker': '>'
    },
    'Fishing Vessel': {
        'weight':             (5000, 30000),
        'speed':              (6, 14),
        'freq_hz':            (80, 180),
        'sl_base':            (150, 165),
        'propeller_diameter': (1.0, 2.5),
        'num_blades':         (2, 4),
        'draft':              (2, 5),
        'color': 'teal', 'marker': '*'
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Zone helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_vessel_zone(vessel):
    """Return the zone the vessel is currently in."""
    x = float(vessel['pos'][0])
    if x > LOC_X:
        return 'outer_yard'
    elif x > HARBOR_X:
        return 'inner_zone'
    else:
        return 'harbor'


def get_vessel_direction(vessel):
    """
    Infer direction from the ship's travel angle.
    Angle in degrees: 0° = +X (outbound), 180° = -X (inbound toward harbor).
    """
    angle = vessel.get('angle', 180.0) % 360
    # Heading roughly toward harbor (left / -X direction)
    if 90 < angle <= 270:
        return 'INBOUND'
    return 'OUTBOUND'


def is_overspeeding(vessel, zone):
    """True if the vessel's current measured speed exceeds the zone limit."""
    speed_knots = vessel.get('speed_knots', 0.0)
    if zone == 'outer_yard':
        return speed_knots > OUTER_YARD_SPEED_LIMIT
    elif zone == 'inner_zone':
        return speed_knots > INNER_ZONE_SPEED_LIMIT
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Sensor layout
# ─────────────────────────────────────────────────────────────────────────────

def generate_sensors(num_sensors=35):
    sensors = []
    for _ in range(num_sensors):
        sx = random.uniform(400, MAP_SIZE - 400)
        sy = random.uniform(400, MAP_SIZE - 400)
        bed = float(get_seabed_depth(sx, sy))
        z_type = random.choice(['surface', 'mid', 'bed'])
        if z_type == 'surface':
            sz = -2
        elif z_type == 'mid':
            sz = bed / 2
        else:
            sz = bed + 2
        sensors.append((sx, sy, sz))
    return sensors


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Fleet generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_fleet():
    """
    All ships spawn on the right side (X > 7500) heading INBOUND so they
    pass through Outer Yard → LOC → Inner Zone naturally.
    6 ship types: Cargo, Tanker, Cruiser, Ferry, Speedboat, Fishing Vessel.
    """
    ship_counter = 1
    fleet = []

    spawn_configs = [
        ('Cargo Ship',      3, -12,  (8500, 9500)),
        ('Tanker',          2, -16,  (8000, 9500)),
        ('Cruiser',         4,  -5,  (7500, 9500)),
        ('Ferry',           3,  -4,  (7800, 9500)),
        ('Speedboat',       6,  -1,  (7500, 9800)),
        ('Fishing Vessel',  4,  -3,  (7500, 9500)),
    ]

    for ship_type, count, depth, x_range in spawn_configs:
        cfg = VEHICLE_TYPES[ship_type]
        for _ in range(count):
            sx = random.uniform(*x_range)
            sy = random.uniform(800, MAP_SIZE - 800)
            spd = random.uniform(*cfg['speed']) * 0.514
            fleet.append({
                "id":            ship_counter,
                "type":          ship_type,
                "pos":           np.array([sx, sy]),
                "depth":         depth,
                "speed_mps":     spd,
                "current_speed": spd,
                "angle":         180.0,
                "state":         "INBOUND",
                "timer":         0,
                "has_cargo":     False,
                "tms_command":   "MOVE",
                "true_weight":   random.uniform(*cfg['weight']),
                "speed_knots":   random.uniform(*cfg['speed']),
                "base_freq":     random.uniform(*cfg['freq_hz']),
                "base_sl":       random.uniform(*cfg['sl_base']),
                "prop_diameter": random.uniform(*cfg['propeller_diameter']),
                "zone":          "outer_yard",
                "direction":     "INBOUND",
                "overspeed":     False,
                "halt_time":     0,
                "dock_slot":     None,
                "path_x": [], "path_y": []
            })
            ship_counter += 1

    return fleet


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – Physics helper (single detection event)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sensor_row(v, sx, sy, sz, range_3d, range_2d, sensor_id, frame):
    """
    Run the full acoustic physics chain for ONE sensor-vessel detection.
    Returns a dict containing the 17 noisy feature columns PLUS
    the 4 ground-truth target columns (for supervised training).
    """
    local_temp       = 15.0 + random.uniform(-3, 3)
    local_salinity   = 35.0 + random.uniform(-0.5, 0.5)
    local_depth_abs  = abs(sz) + random.uniform(-5, 5)
    sea_state        = random.randint(2, 5)
    wind_speed       = random.uniform(5, 20)
    shipping_density = random.uniform(0.2, 0.6)

    sound_speed  = calculate_sound_speed(local_temp, local_salinity, local_depth_abs)
    source_level = calculate_source_level(v['speed_knots'], v['true_weight'],
                                          v['base_sl'], v['prop_diameter'])
    tl           = calculate_transmission_loss(range_3d, v['base_freq'])
    received_lvl = source_level - tl
    ambient_nl   = calculate_ambient_noise(sea_state, wind_speed, shipping_density)
    snr          = calculate_snr(received_lvl, ambient_nl)

    # Doppler — radial velocity component towards the sensor
    rad      = np.radians(v['angle'])
    dx_u     = (sx - v['pos'][0]) / (range_2d + 1e-9)
    dy_u     = (sy - v['pos'][1]) / (range_2d + 1e-9)
    v_radial = v['speed_knots'] * 0.514 * (np.cos(rad)*dx_u + np.sin(rad)*dy_u)

    freq_obs     = calculate_doppler_shift(v['base_freq'], v_radial, sound_speed)
    doppler_shift = freq_obs - v['base_freq']
    toa          = range_3d / sound_speed
    bearing      = np.degrees(np.arctan2(v['pos'][1] - sy, v['pos'][0] - sx)) % 360

    # 20 % realistic measurement noise
    rl   = received_lvl  + random.gauss(0, 0.9)  + random.uniform(-0.4, 0.4)
    sl   = source_level  + random.gauss(0, 1.2)  + random.uniform(-0.6, 0.6)
    tlv  = tl            + random.gauss(0, 0.6)  + random.uniform(-0.3, 0.3)
    fhz  = freq_obs      * (1 + random.gauss(0, 0.016) + random.uniform(-0.006, 0.006))
    dsh  = doppler_shift + random.gauss(0, abs(doppler_shift)*0.03 + 0.4)
    snrv = snr           + random.gauss(0, 0.7)  + random.uniform(-0.3, 0.3)
    amb  = ambient_nl    + random.gauss(0, 0.4)
    spm  = sound_speed   + random.gauss(0, 1.6)  + random.uniform(-0.6, 0.6)
    toav = toa           * (1 + random.gauss(0, 0.012) + random.uniform(-0.004, 0.004))
    brng = (bearing      + random.gauss(0, 1.6)  + random.uniform(-0.8, 0.8)) % 360
    tmp  = local_temp    + random.gauss(0, 0.16)
    sal  = local_salinity + random.gauss(0, 0.08)
    sdep = local_depth_abs + random.gauss(0, 0.3)

    return {
        # Identifiers
        "Timestamp":             frame,
        "Ship_ID":               v['id'],
        "Sensor_ID":             sensor_id,
        # ── Acoustic features (sensor measurements, with noise) ──
        "Received_Level_dB":     round(rl,   2),
        "Source_Level_dB":       round(sl,   2),
        "Transmission_Loss_dB":  round(tlv,  2),
        "Frequency_Hz":          round(fhz,  2),
        "Doppler_Shift_Hz":      round(dsh,  2),
        "SNR_dB":                round(snrv, 2),
        "Ambient_Noise_dB":      round(amb,  2),
        # ── Propagation features ──
        "Sound_Speed_mps":       round(spm,  2),
        "Time_of_Arrival_s":     round(toav, 6),
        "Bearing_deg":           round(brng, 2),
        # ── Environmental features ──
        "Water_Temp_C":          round(tmp,  2),
        "Salinity_ppt":          round(sal,  2),
        "Sensor_Depth_m":        round(sdep, 1),
        "Sea_State":             sea_state,
        "Wind_Speed_knots":      round(wind_speed + random.gauss(0, 0.3), 1),
        # ── Ground truth (used for supervised training) ──
        "Actual_Type":           v['type'],
        "Actual_Weight_kg":      int(v['true_weight']),
        "Actual_Speed_knots":    round(v['speed_knots'], 2),
        "Actual_Depth_m":        round(v.get('depth', -5), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – Dynamic sensor data collection  (called every animation frame)
# ─────────────────────────────────────────────────────────────────────────────

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collect_sensor_detections(fleet, sensors, frame):
    """
    For every sensor × vessel pair within SENSING_RANGE, build one physics row
    and APPEND it to the growing collected_data.csv.
    Now also includes AUVs as dynamic sensor nodes.

    Returns
    -------
    sensor_active : list[bool]   — True if that sensor detected anything this frame
    total_rows    : int          — total rows accumulated in collected_data.csv so far
    """
    new_rows     = []
    sensor_active = [False] * len(sensors)
    
    if not fleet:
        return sensor_active, 0

    # Extract vessel coordinates (N, 3)
    ship_xyz = np.array([[v['pos'][0], v['pos'][1], v.get('depth', -5)]
                         for v in fleet])   # (N, 3)
    
    # 1. Process stationary sensors
    if sensors:
        sensor_xyz = np.array(sensors)
        # Vectorized distance using NumPy broadcasting
        # Shape: (M_sens, 1, 3) - (1, N, 3) -> (M_sens, N, 3)
        diff_3d = sensor_xyz[:, np.newaxis, :] - ship_xyz[np.newaxis, :, :]
        dists_3d = np.linalg.norm(diff_3d, axis=2)
        
        diff_2d = sensor_xyz[:, np.newaxis, :2] - ship_xyz[np.newaxis, :, :2]
        dists_2d = np.linalg.norm(diff_2d, axis=2)
        
        for i, (sx, sy, sz) in enumerate(sensors):
            for j, v in enumerate(fleet):
                if dists_2d[i, j] < SENSING_RANGE:        # Real detection!
                    sensor_active[i] = True
                    row = _build_sensor_row(
                        v, sx, sy, sz,
                        range_3d=float(dists_3d[i, j]),
                        range_2d=float(dists_2d[i, j]),
                        sensor_id=i + 1,
                        frame=frame
                    )
                    new_rows.append(row)

    # 2. Process mobile robot nodes (AUVs)
    rob_states = auv.get_auv_states()
    if rob_states:
        rob_xyz = np.array([r['pos'] for r in rob_states])
        # Vectorized distance using NumPy broadcasting
        diff_3d_rob = rob_xyz[:, np.newaxis, :] - ship_xyz[np.newaxis, :, :]
        dists_3d_rob = np.linalg.norm(diff_3d_rob, axis=2)
        
        diff_2d_rob = rob_xyz[:, np.newaxis, :2] - ship_xyz[np.newaxis, :, :2]
        dists_2d_rob = np.linalg.norm(diff_2d_rob, axis=2)
        
        for i, r in enumerate(rob_states):
            rx, ry, rz = r['pos']
            for j, v in enumerate(fleet):
                if dists_2d_rob[i, j] < SENSING_RANGE:
                    # Mobile sensors have sensor IDs starting after the stationary sensors
                    sensor_id = len(sensors) + r['id']
                    row = _build_sensor_row(
                        v, rx, ry, rz,
                        range_3d=float(dists_3d_rob[i, j]),
                        range_2d=float(dists_2d_rob[i, j]),
                        sensor_id=sensor_id,
                        frame=frame
                    )
                    new_rows.append(row)

    # Atomic append: write to a temp file, then merge
    if new_rows:
        new_df    = pd.DataFrame(new_rows)
        file_exists = os.path.exists(COLLECTED_FILE)
        tmp_path  = COLLECTED_FILE + ".tmp"
        try:
            if file_exists:
                # Read existing, concatenate, write back atomically
                existing = pd.read_csv(COLLECTED_FILE)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_csv(tmp_path, index=False)
            os.replace(tmp_path, COLLECTED_FILE)
        except Exception:
            # Fallback: simple append (less safe but won't crash the simulation)
            new_df.to_csv(COLLECTED_FILE, mode='a',
                          header=not file_exists, index=False)

    # Return total row count for status display
    total_rows = 0
    try:
        if os.path.exists(COLLECTED_FILE):
            total_rows = sum(1 for _ in open(COLLECTED_FILE)) - 1  # -1 for header
    except Exception:
        pass

    return sensor_active, max(total_rows, 0)