import numpy as np
import pandas as pd
import os
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# Constants (mirror simulator.py — no circular import)
# ─────────────────────────────────────────────────────────────────────────────
SAFE_DISTANCE           = 400
LOC_X                   = 5000    # Line of Control
HARBOR_X                = 700
OUTER_YARD_SPEED_LIMIT  = 10.0    # knots
INNER_ZONE_SPEED_LIMIT  = 6.0     # knots

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDS_FILE = os.path.join(tempfile.gettempdir(), "stage3_live_predictions.csv")

# Dock slot definitions  (Y-range centres for each ship type)
DOCK_SLOTS = {
    'Cargo Ship':     [('Dock-A', 2000.0)],
    'Tanker':         [('Dock-T1', 2700.0), ('Dock-T2', 3000.0)],
    'Speedboat':      [('Dock-C', 4125.0)],
    'Fishing Vessel': [('Dock-V1', 5200.0), ('Dock-V2', 5500.0),
                       ('Dock-V3', 5800.0), ('Dock-V4', 6100.0)],
    'Ferry':          [('Dock-F', 6825.0)],
    'Cruiser':        [('Dock-B1', 7500.0), ('Dock-B2', 7850.0),
                       ('Dock-B3', 8150.0), ('Dock-B4', 8450.0)],
}
# Target X for ships when docked (just inside harbor edge)
DOCK_X = 300.0

_last_preds_mtime = 0.0
_dock_assignments: dict = {}   # ship_id → (slot_name, dock_y)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_zone(v):
    x = float(v['pos'][0])
    if x > LOC_X:
        return 'outer_yard'
    elif x > HARBOR_X:
        return 'inner_zone'
    return 'harbor'


def _get_direction(v):
    angle = v.get('angle', 180.0) % 360
    return 'INBOUND' if 90 < angle <= 270 else 'OUTBOUND'


def _knots(v):
    """Return current speed in knots."""
    spd_mps = v.get('current_speed', 0.0)
    return spd_mps / 0.514


def _assign_dock(v):
    """Assign (or recall) a dock slot for this vessel. Returns (slot_name, y)."""
    vid = v['id']
    if vid in _dock_assignments:
        return _dock_assignments[vid]

    ship_type = v.get('pred_type', v['type'])
    # Fallback key resolution
    if ship_type not in DOCK_SLOTS:
        ship_type = v['type']

    slots = DOCK_SLOTS[ship_type]
    used_ys = {y for _, y in _dock_assignments.values()}
    for name, y in slots:
        if y not in used_ys:
            _dock_assignments[vid] = (name, y)
            return (name, y)
    # All slots full — use first slot anyway
    _dock_assignments[vid] = slots[0]
    return slots[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main traffic manager — called every animation frame
# ─────────────────────────────────────────────────────────────────────────────

def manage_traffic_from_csv(fleet):
    global _last_preds_mtime

    # ── Load latest ML predictions ──────────────────────────────────────────
    if os.path.exists(PREDS_FILE):
        try:
            current_mtime = os.path.getmtime(PREDS_FILE)
            if current_mtime > _last_preds_mtime:
                df = pd.read_csv(PREDS_FILE)
                predictions = {row['Ship_ID']: {
                    'Pred_Type':      row['Pred_Type'],
                    'Pred_Weight_kg': row['Pred_Weight_kg']
                } for _, row in df.iterrows()}
                _last_preds_mtime = current_mtime

                for v in fleet:
                    if v['id'] in predictions:
                        v['pred_type']   = predictions[v['id']]['Pred_Type']
                        v['pred_weight'] = predictions[v['id']]['Pred_Weight_kg']
                    else:
                        v.setdefault('pred_type',   v['type'])
                        v.setdefault('pred_weight', int(v.get('true_weight', 0)))
        except (pd.errors.EmptyDataError, OSError, PermissionError, KeyError):
            pass

    violations = []   # list of dicts for dashboard

    for v in fleet:
        v.setdefault('pred_type',   v['type'])
        v.setdefault('pred_weight', int(v.get('true_weight', 0)))

        zone      = _get_zone(v)
        direction = _get_direction(v)
        v['zone']      = zone
        v['direction'] = direction

        # Overspeed check
        speed_knots = _knots(v)
        limit = OUTER_YARD_SPEED_LIMIT if zone == 'outer_yard' else INNER_ZONE_SPEED_LIMIT
        v['overspeed'] = (speed_knots > limit) and (v.get('current_speed', 0) > 0)
        if v['overspeed']:
            violations.append({
                'id':    v['id'],
                'type':  v['type'],
                'zone':  zone,
                'speed': round(speed_knots, 1),
                'limit': limit,
            })

        # ── OUTER YARD ZONE ──────────────────────────────────────────────────
        if zone == 'outer_yard':
            _handle_outer_yard(v)

        # ── INNER ZONE (PATROL ZONE) ─────────────────────────────────────────
        elif zone == 'inner_zone':
            _handle_inner_zone(v)

        # ── HARBOR (already docked) ──────────────────────────────────────────
        else:
            v['state']         = 'DOCKED'
            v['tms_command']   = 'DOCKED'
            v['current_speed'] = 0
            v['halt_time']     = v.get('halt_time', 0) + 1

    return fleet, violations


# ─────────────────────────────────────────────────────────────────────────────
# Outer Yard handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_outer_yard(v):
    """
    Apply entry commands based on ML-predicted ship type.
      Cargo Ship / Tanker       -> HALT          (heavy vessels wait at LOC)
      Speedboat / Fishing Vessel-> CONTINUE       (light vessels pass freely)
      Cruiser / Ferry           -> SLOW DOWN      (passenger vessels reduce speed)
    """
    pred = v.get('pred_type', v['type'])

    if pred in ('Cargo Ship', 'Tanker'):
        v['tms_command']   = 'HALT'
        v['current_speed'] = 0
        v['state']         = 'HALTED_AT_LOC'
        v['halt_time']     = v.get('halt_time', 0) + 1

    elif pred in ('Speedboat', 'Fishing Vessel'):
        v['tms_command']   = 'CONTINUE'
        v['current_speed'] = v['speed_mps']
        v['state']         = 'CROSSING'
        v['halt_time']     = 0
        _aim_at(v, LOC_X - 100, v['pos'][1])

    elif pred in ('Cruiser', 'Ferry'):
        v['tms_command']   = 'SLOW DOWN'
        v['current_speed'] = v['speed_mps'] * 0.5
        v['state']         = 'SLOWING'
        v['halt_time']     = 0
        _aim_at(v, LOC_X - 100, v['pos'][1])

    else:
        v['tms_command']   = 'HOLD'
        v['current_speed'] = 0
        v['state']         = 'AWAITING_ID'
        v['halt_time']     = v.get('halt_time', 0) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Inner Zone handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_inner_zone(v):
    """
    Smart parking: assign a dock slot, then route the vessel there.
    States: ROUTING → DOCKING → DOCKED
    """
    slot_name, dock_y = _assign_dock(v)
    v['dock_slot'] = slot_name

    target = np.array([DOCK_X, dock_y])
    dist   = np.linalg.norm(v['pos'] - target)

    if dist < 100:
        # Close enough — begin docking
        v['state']         = 'DOCKING'
        v['tms_command']   = 'DOCKING'
        v['current_speed'] = 0
        v['halt_time']     = v.get('halt_time', 0) + 1
    else:
        v['state']       = 'ROUTING'
        v['tms_command'] = f'ROUTE→{slot_name}'
        # Speed governed by zone limit
        speed_cap = INNER_ZONE_SPEED_LIMIT * 0.514   # m/s
        v['current_speed'] = min(v['speed_mps'], speed_cap)
        _aim_at(v, target[0], target[1])
        v['halt_time'] = 0


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _aim_at(v, tx, ty):
    """Point vessel toward (tx, ty)."""
    dy = ty - v['pos'][1]
    dx = tx - v['pos'][0]
    v['angle'] = np.degrees(np.arctan2(dy, dx))