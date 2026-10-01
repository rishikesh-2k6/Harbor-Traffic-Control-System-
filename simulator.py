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
# Environmental Scenarios Configuration
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = {
    'NORMAL': {
        'name': 'Normal Season',
        'key': 'NORMAL',
        'badge': '☀️ Normal Season',
        'description': 'Calm waters, moderate winds, low acoustic background noise, nominal modem range.',
        'sea_state': (2, 3),              # Douglas scale 2-3 (slight to moderate waves)
        'wind_speed_knots': (8.0, 16.0),  # 8-16 knots
        'temp_c': (18.0, 22.0),           # 18-22 °C
        'salinity_ppt': (34.5, 35.5),     # 34.5 - 35.5 ppt (open sea salinity)
        'rain_noise_db': 0.0,             # No rain impact noise
        'shipping_density': (0.3, 0.5),   # Baseline harbor traffic noise index
        'current_mult': 1.0,              # Baseline tidal current (~0.35 m/s peak)
        'ocean_color': '#AED9E0',         # Azure calm water
    },
    'MONSOON': {
        'name': 'Monsoon Season',
        'key': 'MONSOON',
        'badge': '⛈️ Monsoon Season',
        'description': 'Severe sea state, gale-force winds, torrential rain noise, turbulent currents, acoustic SNR degradation.',
        'sea_state': (6, 8),              # Douglas scale 6-8 (very rough to high seas)
        'wind_speed_knots': (36.0, 52.0), # 36-52 knots (near gale to storm force)
        'temp_c': (12.0, 15.0),           # 12-15 °C (cooler freshwater runoff)
        'salinity_ppt': (30.0, 32.5),     # 30-32.5 ppt (freshwater dilution)
        'rain_noise_db': 15.0,            # +15 dB torrential sea surface droplet noise (Wenz/Nystuen)
        'shipping_density': (0.2, 0.4),   # Storm shipping density
        'current_mult': 2.6,              # Storm surge & wind-driven currents (~0.9-1.2 m/s peak)
        'ocean_color': '#4A6572',         # Dark turbulent stormy water
    }
}

ACTIVE_SCENARIO_KEY = 'NORMAL'

def set_active_scenario(key):
    global ACTIVE_SCENARIO_KEY
    if key in SCENARIOS:
        ACTIVE_SCENARIO_KEY = key
        print(f"[Environment] Switched scenario to: {SCENARIOS[key]['name']}")
        return True
    return False

def get_active_scenario():
    return SCENARIOS.get(ACTIVE_SCENARIO_KEY, SCENARIOS['NORMAL'])


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


def calculate_ambient_noise(sea_state, wind_speed_knots, shipping_density, rain_noise=0.0):
    """Wenz (1962) incoherent combination — dB re 1 uPa, with rain noise enhancement."""
    wind_speed_knots = max(wind_speed_knots, 1.0)
    w  = 44 + 20*np.log10(wind_speed_knots) + rain_noise
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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2b – Non-Deterministic Ocean Current & Wave Physics Engine
# ─────────────────────────────────────────────────────────────────────────────

OCEAN_STATE = {
    'intensity_mps': 0.35,          # instantaneous effective current speed (m/s)
    'target_intensity': 0.35,       # stochastic drift target
    'squall_timer': 0,              # frames remaining in sudden violent squall
    'squall_surge': 0.0,            # added speed during squall (m/s)
    'surge_water_level': 0.0,       # mean water level displacement (m)
    'wave_height': 0.3,             # significant wave height peak-to-trough (m)
    'gust_heading_rad': np.radians(225.0), # direction of storm surge flow
    'vortices': [                   # wandering turbulent gyres [x, y, spin]
        [3200.0, 4200.0, 1.4],
        [7200.0, 6200.0, -1.2],
        [4800.0, 2400.0, 1.6]
    ],
    'snapped_count': 0,             # sensors broken away
    'network_broken': False
}

def step_ocean_environment(frame, dt=1.0/60.0):
    """
    Step the non-deterministic ocean current, water level disruption,
    and wave field state forward by dt.
    """
    scenario = get_active_scenario()
    is_monsoon = (scenario.get('key') == 'MONSOON')

    if not is_monsoon:
        # Normal Season: calm, predictable, gentle tidal breathing
        t_phase = frame * 0.02
        OCEAN_STATE['intensity_mps'] = 0.35 + 0.04 * np.sin(t_phase)
        OCEAN_STATE['target_intensity'] = 0.35
        OCEAN_STATE['squall_timer'] = 0
        OCEAN_STATE['squall_surge'] = 0.0
        OCEAN_STATE['surge_water_level'] = 0.15 * np.sin(t_phase * 0.5)
        OCEAN_STATE['wave_height'] = 0.35 + 0.08 * np.cos(t_phase * 0.7)
        return OCEAN_STATE

    # Monsoon Season: Non-deterministic stochastic current intensity and water level disruptions
    # 1. Stochastic intensity drift-diffusion
    if frame % 30 == 0:
        # Pick new stochastic target intensity between 2.6 and 4.6 m/s
        OCEAN_STATE['target_intensity'] = random.uniform(2.8, 4.6)

    # Ornstein-Uhlenbeck drift towards stochastic target + random Gaussian turbulence
    dI = 0.08 * (OCEAN_STATE['target_intensity'] - OCEAN_STATE['intensity_mps']) + random.gauss(0, 0.16)
    OCEAN_STATE['intensity_mps'] = float(np.clip(OCEAN_STATE['intensity_mps'] + dI, 1.8, 6.8))

    # 2. Sudden Gale Squall Bursts (non-deterministic random severe surges)
    if OCEAN_STATE['squall_timer'] > 0:
        OCEAN_STATE['squall_timer'] -= 1
        if OCEAN_STATE['squall_timer'] == 0:
            OCEAN_STATE['squall_surge'] = 0.0
    else:
        # ~2.5% chance per frame to trigger sudden violent current surge
        if random.random() < 0.025:
            OCEAN_STATE['squall_timer'] = random.randint(45, 120)
            OCEAN_STATE['squall_surge'] = random.uniform(1.8, 3.4)
            print(f"[Monsoon Gale] Severe Squall Surge struck! Current spiking by +{OCEAN_STATE['squall_surge']:.2f} m/s!")

    # 3. Disrupted Water Level (storm surge + chaotic oscillation)
    t_phase = frame * 0.03
    base_surge = 3.2 + 1.8 * np.sin(t_phase * 0.4) + 0.9 * np.cos(t_phase * 0.8)
    squall_bonus = (OCEAN_STATE['squall_surge'] * 0.45) if OCEAN_STATE['squall_timer'] > 0 else 0.0
    OCEAN_STATE['surge_water_level'] = float(base_surge + squall_bonus + random.gauss(0, 0.08))

    # Significant wave height peak-to-trough (5.0m to 8.5m+)
    OCEAN_STATE['wave_height'] = float(5.2 + 1.6 * np.sin(t_phase * 0.6) + squall_bonus * 0.8)

    # 4. Wandering turbulent vortices
    for v in OCEAN_STATE['vortices']:
        v[0] += random.uniform(-18.0, 18.0)
        v[1] += random.uniform(-18.0, 18.0)
        v[0] = float(np.clip(v[0], 1000.0, MAP_SIZE - 1000.0))
        v[1] = float(np.clip(v[1], 1000.0, MAP_SIZE - 1000.0))
        v[2] += random.gauss(0, 0.03)

    return OCEAN_STATE


def trigger_monsoon_squall():
    """Immediately trigger a violent non-deterministic current burst."""
    OCEAN_STATE['squall_timer'] = random.randint(60, 140)
    OCEAN_STATE['squall_surge'] = random.uniform(2.5, 4.2)
    print(f"[Manual Trigger] Violent Monsoon Squall Injected! Surge: +{OCEAN_STATE['squall_surge']:.2f} m/s")
    return OCEAN_STATE['squall_surge']


def get_monsoon_current_info():
    """Return live telemetry snapshot of the ocean current and wave dynamics."""
    total_speed = OCEAN_STATE['intensity_mps'] + (OCEAN_STATE['squall_surge'] if OCEAN_STATE['squall_timer'] > 0 else 0.0)
    return {
        'intensity_mps': round(float(total_speed), 2),
        'base_intensity': round(float(OCEAN_STATE['intensity_mps']), 2),
        'squall_active': OCEAN_STATE['squall_timer'] > 0,
        'squall_surge': round(float(OCEAN_STATE['squall_surge']), 2),
        'surge_level': round(float(OCEAN_STATE['surge_water_level']), 2),
        'wave_height': round(float(OCEAN_STATE['wave_height']), 2),
        'vortices': OCEAN_STATE['vortices'],
        'snapped_count': OCEAN_STATE.get('snapped_count', 0),
        'network_broken': OCEAN_STATE.get('network_broken', False)
    }


def get_water_surface_height(x, y, t):
    """
    Returns water surface elevation (Z in meters) at (x, y) at time t.
    In Normal season: flat calm surface (~0m, gentle ripple).
    In Monsoon season: severely disrupted water levels with storm surge,
    rolling chaotic waves, and turbulent crests.
    """
    scenario = get_active_scenario()
    is_monsoon = (scenario.get('key') == 'MONSOON')

    if not is_monsoon:
        return 0.35 * np.sin(x/850.0 + t*1.2) * np.cos(y/1100.0 + t*0.9)

    surge = OCEAN_STATE['surge_water_level']
    # Multi-harmonic chaotic storm wave superposition
    w1 = 2.4 * np.sin(x/620.0 + y/820.0 - t*3.5)
    w2 = 1.9 * np.cos(x/440.0 - y/640.0 + t*4.2)
    w3 = 1.2 * np.sin(x/280.0 + t*5.1) * np.cos(y/310.0 - t*4.6)
    w4 = 0.7 * np.sin((x - y)/160.0 + t*6.8)  # high-frequency foaming wave chop

    squall_extra = (OCEAN_STATE['squall_surge'] * 0.35) if OCEAN_STATE['squall_timer'] > 0 else 0.0
    return surge + w1 + w2 + w3 + w4 + squall_extra


def get_current_vector(x, y, z, t):
    """
    Non-deterministic, depth-dependent, time-varying water current field — m/s.
    In Normal season: nominal tidal current (~0.35 m/s peak).
    In Monsoon season: violent turbulent storm surge currents (~2.2 - 6.5 m/s)
    with wandering vortex gyres and vertical heave.
    """
    depth_frac     = np.clip(-z / 60.0, 0.0, 1.0)     # 0 at surface, 1 near seabed
    surface_factor = 1.0 - 0.65 * depth_frac          # currents strongest near surface

    scenario = get_active_scenario()
    is_monsoon = (scenario.get('key') == 'MONSOON')

    if not is_monsoon:
        tphase = t / 400.0
        vx = surface_factor * (0.35*np.sin(y/2200.0 + tphase) + 0.15*np.cos(x/3000.0 - tphase*0.6))
        vy = surface_factor * (0.30*np.cos(x/2500.0 - tphase*0.8) + 0.12*np.sin(y/1800.0 + tphase*0.4))
        vz = 0.02 * np.sin((x + y)/4000.0 + tphase*0.3)
        return float(vx), float(vy), float(vz)

    # Monsoon season: highly non-deterministic current field
    speed_mult = OCEAN_STATE['intensity_mps']
    if OCEAN_STATE['squall_timer'] > 0:
        speed_mult += OCEAN_STATE['squall_surge']

    tphase = t * 0.8
    # 1. Storm surge flow generally westward / southwestward through the harbor
    base_vx = -0.55 + 0.30 * np.sin(y/1900.0 + tphase*0.4) + 0.20 * np.cos(x/2600.0 - tphase*0.3)
    base_vy = 0.32 * np.cos(x/2100.0 + tphase*0.35) - 0.25 * np.sin(y/1600.0 - tphase*0.25)

    # 2. Multi-scale chaotic spatial turbulence
    turb_vx = 0.25 * np.sin(x/700.0 + y/900.0 + tphase*0.9) + 0.15 * np.cos(y/550.0 - tphase*1.1)
    turb_vy = 0.25 * np.cos(x/850.0 - y/750.0 + tphase*1.0) + 0.15 * np.sin(x/500.0 + tphase*0.8)

    # 3. Wandering vortex gyres (swirling ocean eddies)
    vortex_vx, vortex_vy = 0.0, 0.0
    for vx_c, vy_c, v_spin in OCEAN_STATE['vortices']:
        dx_v = (x - vx_c)
        dy_v = (y - vy_c)
        dist = np.sqrt(dx_v*dx_v + dy_v*dy_v + 1e5)
        # Bounded vortex swirl velocity (0.35 m/s peak)
        vortex_vx += -v_spin * (dy_v / dist) * 0.30
        vortex_vy +=  v_spin * (dx_v / dist) * 0.30

    # 4. Vertical current heave (upwelling / downwelling dragging sensors)
    vz = surface_factor * 0.35 * np.sin((x*1.3 + y)/1600.0 + tphase*1.2) * (speed_mult / 3.0)

    total_vx = surface_factor * (base_vx + turb_vx + vortex_vx) * (speed_mult / 1.5)
    total_vy = surface_factor * (base_vy + turb_vy + vortex_vy) * (speed_mult / 1.5)

    return float(total_vx), float(total_vy), float(vz)


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
# SECTION 5 – Sensor layout & Moored Hydrodynamic Physics
# ─────────────────────────────────────────────────────────────────────────────

class MooredSensor:
    """
    Physical representation of a seabed-moored acoustic sensor node.
    Equipped with anchor on the seabed, elastic tether cable, hydrodynamic drag,
    mechanical strain fatigue tracking, and drift physics when driven away by harsh currents.
    """
    def __init__(self, sensor_id, anchor_x, anchor_y, target_depth):
        self.id = int(sensor_id)
        bed_z = float(get_seabed_depth(anchor_x, anchor_y))
        self.anchor = np.array([anchor_x, anchor_y, bed_z], dtype=float)
        self.initial_pos = np.array([anchor_x, anchor_y, float(target_depth)], dtype=float)
        self.pos = self.initial_pos.copy()
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.target_depth = float(target_depth)
        self.status = "MOORED"   # "MOORED", "STRAINED", "SNAPPED" (drifting)
        # Structural breaking velocity threshold (m/s)
        self.break_threshold = random.uniform(2.2, 3.4)
        self.strain_accum = 0.0
        self.tether_length = abs(target_depth - bed_z)
        self.trail = []

    def __getitem__(self, idx):
        return float(self.pos[idx])

    def __len__(self):
        return 3

    def __iter__(self):
        return iter(self.pos)

    def reanchor(self):
        """Restore sensor to initial seabed mooring state."""
        self.status = "MOORED"
        self.pos = self.initial_pos.copy()
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.strain_accum = 0.0
        self.trail = []

    def update(self, frame, dt, is_monsoon):
        t = frame * dt
        cvx, cvy, cvz = get_current_vector(self.pos[0], self.pos[1], self.pos[2], t)
        current_vec = np.array([cvx, cvy, cvz], dtype=float)
        current_speed = float(np.linalg.norm(current_vec[:2]))

        bed_z = float(get_seabed_depth(self.pos[0], self.pos[1]))
        water_surf_z = float(get_water_surface_height(self.pos[0], self.pos[1], t))

        if not is_monsoon:
            # Normal Season: tether holds firm, small gentle sway around mooring
            if self.status != "SNAPPED":
                self.status = "MOORED"
                self.strain_accum = max(0.0, self.strain_accum - dt * 2.0)
                sway_t = t * 1.5 + self.id
                sway = np.array([6.0 * np.sin(sway_t), 6.0 * np.cos(sway_t * 0.8), 0.3 * np.sin(sway_t * 2.0)])
                self.pos = self.initial_pos + sway
                self.vel = np.array([0.0, 0.0, 0.0])
                return

        # Monsoon Season:
        if self.status in ("MOORED", "STRAINED"):
            if current_speed > 1.3:
                self.status = "STRAINED"
                self.strain_accum += dt * (current_speed / 1.2)
            else:
                self.status = "MOORED"
                self.strain_accum = max(0.0, self.strain_accum - dt * 0.6)

            # Snap trigger: instantaneous violent current spike OR cumulative mechanical fatigue
            if current_speed >= self.break_threshold or self.strain_accum > 12.0:
                self.status = "SNAPPED"
                print(f"[Monsoon Hazard] Sensor #{self.id} mooring SNAPPED by harsh current ({current_speed:.2f} m/s)! Sensor is DRIVEN AWAY!")
            else:
                # Moored elastic sway in current direction
                sway_dist = min(current_speed * 40.0, 220.0)
                dir_c = current_vec[:2] / (current_speed + 1e-6)
                self.pos[0] = self.anchor[0] + dir_c[0] * sway_dist
                self.pos[1] = self.anchor[1] + dir_c[1] * sway_dist
                self.pos[2] = np.clip(self.initial_pos[2] + cvz * 6.0, bed_z + 1.0, water_surf_z - 1.0)
                return

        # If SNAPPED / DRIFTING:
        # The sensor is DRIVEN AWAY by the harsh water currents!
        VISUAL_DRIFT_MULT = 14.0
        target_vel = current_vec * random.uniform(0.85, 1.15) * VISUAL_DRIFT_MULT
        # Inertial lag
        self.vel += (target_vel - self.vel) * min(dt * 4.0, 1.0)
        self.pos += self.vel * dt

        # Disrupted water level heave
        if self.target_depth > -5.0:
            # Surface sensor: rides undulating wave peaks & troughs
            self.pos[2] = water_surf_z - random.uniform(0.6, 1.8)
        else:
            # Subsurface sensor: heaved vertically by water currents
            self.pos[2] = np.clip(self.pos[2] + cvz * 8.0 * dt, bed_z + 2.0, water_surf_z - 1.5)

        # Soft wrap / bounce if drifting past map boundaries
        if self.pos[0] < 50:
            self.pos[0] = MAP_SIZE - 200
        elif self.pos[0] > MAP_SIZE - 50:
            self.pos[0] = 200
        if self.pos[1] < 50:
            self.pos[1] = MAP_SIZE - 200
        elif self.pos[1] > MAP_SIZE - 50:
            self.pos[1] = 200

        # Maintain drift trail history
        if len(self.trail) == 0 or np.linalg.norm(self.pos[:2] - self.trail[-1][:2]) > 90.0:
            self.trail.append(self.pos.copy())
            if len(self.trail) > 12:
                self.trail.pop(0)

_active_sensors = []

def generate_sensors(num_sensors=35):
    global _active_sensors
    _active_sensors = []
    for i in range(num_sensors):
        sx = random.uniform(600, MAP_SIZE - 600)
        sy = random.uniform(600, MAP_SIZE - 600)
        bed = float(get_seabed_depth(sx, sy))
        z_type = random.choice(['surface', 'mid', 'bed'])
        if z_type == 'surface':
            sz = -2.0
        elif z_type == 'mid':
            sz = bed / 2.0
        else:
            sz = bed + 2.0
        _active_sensors.append(MooredSensor(sensor_id=i+1, anchor_x=sx, anchor_y=sy, target_depth=sz))
    return _active_sensors

def step_sensors(frame, dt=1.0/60.0):
    scenario = get_active_scenario()
    is_monsoon = (scenario.get('key') == 'MONSOON')
    snapped = 0
    for s in _active_sensors:
        s.update(frame, dt, is_monsoon)
        if s.status == "SNAPPED":
            snapped += 1
    OCEAN_STATE['snapped_count'] = snapped
    return _active_sensors

def reset_all_sensors():
    for s in _active_sensors:
        s.reanchor()
    OCEAN_STATE['snapped_count'] = 0
    print("[Sensors] All 35 sensors re-anchored to seabed moorings.")

def get_sensor_states():
    moored = sum(1 for s in _active_sensors if s.status == "MOORED")
    strained = sum(1 for s in _active_sensors if s.status == "STRAINED")
    snapped = sum(1 for s in _active_sensors if s.status == "SNAPPED")
    return {
        'total': len(_active_sensors),
        'moored': moored,
        'strained': strained,
        'snapped': snapped
    }


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
    scenario = get_active_scenario()
    local_temp       = random.uniform(*scenario['temp_c']) + random.uniform(-0.4, 0.4)
    local_salinity   = random.uniform(*scenario['salinity_ppt']) + random.uniform(-0.15, 0.15)
    local_depth_abs  = abs(sz) + random.uniform(-5, 5)
    sea_state        = random.randint(*scenario['sea_state'])
    wind_speed       = random.uniform(*scenario['wind_speed_knots'])
    shipping_density = random.uniform(*scenario['shipping_density'])
    rain_noise       = scenario.get('rain_noise_db', 0.0)

    sound_speed  = calculate_sound_speed(local_temp, local_salinity, local_depth_abs)
    source_level = calculate_source_level(v['speed_knots'], v['true_weight'],
                                          v['base_sl'], v['prop_diameter'])
    tl           = calculate_transmission_loss(range_3d, v['base_freq'])
    received_lvl = source_level - tl
    ambient_nl   = calculate_ambient_noise(sea_state, wind_speed, shipping_density, rain_noise=rain_noise)
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
        sensor_xyz = np.array([s.pos for s in sensors], dtype=float)
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