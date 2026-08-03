"""
AUV swarm physics — Stage 1.

Uses PyBullet (headless/DIRECT mode) as the rigid-body physics backend so
AUV-AUV collisions ("dashing" into each other) resolve through real contact
forces rather than hand-rolled avoidance math. Matplotlib in main.py remains
the single visual layer; this module only produces positions/velocities for
it to draw.

Movement here is a simple waypoint patrol — just enough to demonstrate real
collision response and current-drift under PyBullet. Stage 2 replaces the
control force with a swarm/connectivity-aware controller.
"""
import random
import numpy as np
import pybullet as p

import simulator as sim

MAP           = sim.MAP_SIZE
AUV_RADIUS    = sim.AUV_RADIUS_M
AUV_MASS      = sim.AUV_MASS_KG

MAX_CONTROL_FORCE = 900.0    # N — bounded thrust, real AUV thrusters are limited
WAYPOINT_TOL      = 250.0    # metres — close enough to pick a new patrol point
CURRENT_FORCE_GAIN = 0.6     # how strongly ambient current pushes the hull

_client = None
_bodies = []   # pybullet body ids
_meta   = []   # per-AUV bookkeeping dicts


def _random_patrol_point():
    x = random.uniform(800, MAP - 800)
    y = random.uniform(800, MAP - 800)
    bed = float(sim.get_seabed_depth(x, y))
    z = random.uniform(bed + 8, -5)     # stay clear of seabed and surface
    return np.array([x, y, z])


def init_physics():
    """Connect a headless PyBullet world. Call once at boot."""
    global _client
    _client = p.connect(p.DIRECT)
    p.setGravity(0, 0, 0, physicsClientId=_client)   # AUVs are neutrally buoyant
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/60.0, numSubSteps=2,
                                physicsClientId=_client)
    return _client


def spawn_auvs(n=None):
    """Create the AUV swarm as spherical rigid bodies. Returns per-AUV state list."""
    global _bodies, _meta
    if n is None:
        n = sim.NUM_AUVS

    _bodies, _meta = [], []
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=AUV_RADIUS,
                                 physicsClientId=_client)

    for i in range(n):
        x = random.uniform(1000, MAP - 1000)
        y = random.uniform(1000, MAP - 1000)
        bed = float(sim.get_seabed_depth(x, y))
        z = random.uniform(bed + 8, -5)

        body = p.createMultiBody(baseMass=AUV_MASS,
                                  baseCollisionShapeIndex=col,
                                  basePosition=[x, y, z],
                                  physicsClientId=_client)
        # Water drag (damping) + soft, non-elastic bump on contact
        p.changeDynamics(body, -1,
                          linearDamping=0.85,
                          angularDamping=0.9,
                          restitution=0.35,
                          lateralFriction=0.2,
                          physicsClientId=_client)
        _bodies.append(body)
        _meta.append({
            "id":          i + 1,
            "waypoint":    _random_patrol_point(),
            "battery_pct": 100.0,   # placeholder — Stage 2 energy model
        })

    return get_auv_states()


def get_auv_states():
    states = []
    for body, meta in zip(_bodies, _meta):
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=_client)
        vel, _ = p.getBaseVelocity(body, physicsClientId=_client)
        states.append({
            "id":          meta["id"],
            "pos":         np.array(pos),
            "vel":         np.array(vel),
            "battery_pct": meta["battery_pct"],
        })
    return states


def step_auvs(frame, dt=1.0/60.0):
    """
    Advance the AUV swarm by one physics step:
      1. Waypoint-patrol control force (bounded thrust).
      2. External current-drift force sampled from simulator.get_current_vector().
      3. PyBullet integrates motion and resolves AUV-AUV collisions natively.
    Returns the resulting per-AUV state list (id, pos, vel, battery_pct).
    """
    t = frame * dt

    for body, meta in zip(_bodies, _meta):
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=_client)
        pos = np.array(pos)

        # ── Waypoint patrol control ──────────────────────────────────────
        to_wp = meta["waypoint"] - pos
        dist  = np.linalg.norm(to_wp)
        if dist < WAYPOINT_TOL:
            meta["waypoint"] = _random_patrol_point()
            to_wp = meta["waypoint"] - pos
            dist  = np.linalg.norm(to_wp)
        control_force = (to_wp / (dist + 1e-6)) * MAX_CONTROL_FORCE

        # ── Ambient current drift ────────────────────────────────────────
        cvx, cvy, cvz = sim.get_current_vector(pos[0], pos[1], pos[2], t)
        current_force = np.array([cvx, cvy, cvz]) * AUV_MASS * CURRENT_FORCE_GAIN

        total_force = control_force + current_force
        p.applyExternalForce(body, -1, total_force.tolist(), pos.tolist(),
                              p.WORLD_FRAME, physicsClientId=_client)

    p.stepSimulation(physicsClientId=_client)

    # Soft boundary clamp (keep the swarm inside the mapped harbor volume)
    for body in _bodies:
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=_client)
        bed = float(sim.get_seabed_depth(pos[0], pos[1]))
        clamped = (
            float(np.clip(pos[0], 10, MAP - 10)),
            float(np.clip(pos[1], 10, MAP - 10)),
            float(np.clip(pos[2], bed + 3, -2)),
        )
        if clamped != pos:
            p.resetBasePositionAndOrientation(body, clamped, orn, physicsClientId=_client)

    return get_auv_states()


def close_physics():
    global _client
    if _client is not None:
        p.disconnect(physicsClientId=_client)
        _client = None
