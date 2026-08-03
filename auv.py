"""
AUV swarm physics — Stage 1.

Standalone numpy-based kinematics and behavior model. Replaces PyBullet to guarantee
compatibility and fast execution without compiled C++ binary requirements on Windows.
Tracks 3D coordinates, zone-based waypoint patrol, battery drain, charging routines,
and active vessel interception (dispatch).
"""
import random
import numpy as np
import simulator as sim

class RobotNode:
    def __init__(self, node_id, initial_zone=None):
        self.id = node_id
        # Patrol zones: "Outer Yard", "Channel", "Inner Harbor"
        self.patrol_zone = initial_zone or random.choice(["Outer Yard", "Channel", "Inner Harbor"])
        self.status = "PATROLLING"  # "PATROLLING", "DISPATCHED", "CHARGING"
        self.battery_pct = random.uniform(85.0, 100.0)
        
        # Position initialization based on zone
        self.pos = self._generate_random_position_in_zone(self.patrol_zone)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.waypoint = self._generate_waypoint_in_zone(self.patrol_zone)
        
        # Physical constraints
        self.max_speed = 10.0  # m/s
        self.max_force = 900.0  # N
        self.mass = 500.0       # kg
        self.drag_coeff = 0.85  # drag damping
        self.waypoint_tol = 200.0 # meters tolerance to waypoint
        self.low_battery_threshold = 15.0
        self.full_battery_threshold = 95.0
        self.battery_drain_rate = 0.015 # percentage per frame
        self.charge_rate = 0.2         # percentage per frame when docked at buoy
        
        # Dispatch tracking
        self.target_vessel = None
        self.intercept_hold_timer = 0
        
    def _generate_random_position_in_zone(self, zone):
        x, y, z = 0.0, 0.0, 0.0
        if zone == "Outer Yard":
            x = random.uniform(5200, 9500)
            y = random.uniform(1000, 9000)
        elif zone == "Channel":
            x = random.uniform(1200, 4800)
            y = random.uniform(3000, 7000)
        else: # Inner Harbor
            x = random.uniform(200, 700)
            y = random.uniform(1500, 8500)
            
        bed = float(sim.get_seabed_depth(x, y))
        z = random.uniform(bed + 5.0, -5.0)
        return np.array([x, y, z])
        
    def _generate_waypoint_in_zone(self, zone):
        return self._generate_random_position_in_zone(zone)
        
    def get_state(self):
        return {
            "id": self.id,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "battery_pct": round(self.battery_pct, 1),
            "status": self.status,
            "waypoint": self.waypoint.copy(),
            "patrol_zone": self.patrol_zone
        }
        
    def update(self, frame, dt, active_violations=None, fleet=None):
        t = frame * dt
        
        # 1. Update battery
        if self.status == "CHARGING":
            # If close to charging buoy, charge battery
            dist_to_buoy = np.linalg.norm(self.pos - np.array(sim.BUOY_POS))
            if dist_to_buoy < 250.0:
                self.battery_pct = min(100.0, self.battery_pct + self.charge_rate)
                if self.battery_pct >= self.full_battery_threshold:
                    self.status = "PATROLLING"
                    self.waypoint = self._generate_waypoint_in_zone(self.patrol_zone)
            # Move towards charging station
            self.waypoint = np.array(sim.BUOY_POS)
        else:
            # Drain battery based on velocity
            speed_ratio = np.linalg.norm(self.vel) / self.max_speed
            drain = self.battery_drain_rate * (1.0 + speed_ratio)
            self.battery_pct = max(0.0, self.battery_pct - drain)
            
            # Check low battery trigger
            if self.battery_pct < self.low_battery_threshold:
                self.status = "CHARGING"
                self.waypoint = np.array(sim.BUOY_POS)
                self.target_vessel = None
                
        # 2. Dispatch / Patrolling State Machine
        if self.status == "PATROLLING":
            # Check for violations to dispatch to
            if active_violations and fleet:
                best_vessel = None
                min_dist = float('inf')
                for vio in active_violations:
                    v_match = next((v for v in fleet if v['id'] == vio['id']), None)
                    if v_match:
                        vessel_pos_3d = np.array([v_match['pos'][0], v_match['pos'][1], v_match.get('depth', -5)])
                        dist = np.linalg.norm(self.pos - vessel_pos_3d)
                        if dist < min_dist:
                            vx = v_match['pos'][0]
                            v_zone = "Outer Yard" if vx > sim.LOC_X else ("Channel" if vx > sim.HARBOR_X else "Inner Harbor")
                            if v_zone == self.patrol_zone:
                                min_dist = dist
                                best_vessel = v_match
                                
                if best_vessel:
                    self.status = "DISPATCHED"
                    self.target_vessel = best_vessel
                    self.intercept_hold_timer = 0
            
            # Check waypoint arrival
            dist_to_wp = np.linalg.norm(self.pos - self.waypoint)
            if dist_to_wp < self.waypoint_tol:
                self.waypoint = self._generate_waypoint_in_zone(self.patrol_zone)
                
        elif self.status == "DISPATCHED":
            if not self.target_vessel or self.target_vessel.get('dock_slot') is not None:
                self.status = "PATROLLING"
                self.target_vessel = None
                self.waypoint = self._generate_waypoint_in_zone(self.patrol_zone)
            else:
                target_pos_3d = np.array([self.target_vessel['pos'][0], self.target_vessel['pos'][1], self.target_vessel.get('depth', -5)])
                self.waypoint = target_pos_3d
                dist_to_target = np.linalg.norm(self.pos - target_pos_3d)
                
                if dist_to_target < self.waypoint_tol:
                    self.intercept_hold_timer += 1
                    if self.intercept_hold_timer > 180:
                        self.status = "PATROLLING"
                        self.target_vessel = None
                        self.waypoint = self._generate_waypoint_in_zone(self.patrol_zone)
                        
        # 3. Physics Integration (Forces -> Velocity -> Position)
        if self.status == "CHARGING" and np.linalg.norm(self.pos - np.array(sim.BUOY_POS)) < 150.0:
            self.vel = np.array([0.0, 0.0, 0.0])
        elif self.status == "DISPATCHED" and self.intercept_hold_timer > 0:
            self.vel = np.array([0.0, 0.0, 0.0])
        else:
            to_wp = self.waypoint - self.pos
            dist = np.linalg.norm(to_wp)
            dir_v = to_wp / (dist + 1e-6)
            
            control_force = dir_v * self.max_force
            cvx, cvy, cvz = sim.get_current_vector(self.pos[0], self.pos[1], self.pos[2], t)
            current_force = np.array([cvx, cvy, cvz]) * self.mass * 0.6
            drag_force = -self.drag_coeff * self.vel * np.linalg.norm(self.vel)
            
            total_force = control_force + current_force + drag_force
            acc = total_force / self.mass
            
            self.vel += acc * dt
            speed = np.linalg.norm(self.vel)
            if speed > self.max_speed:
                self.vel = (self.vel / speed) * self.max_speed
                
            self.pos += self.vel * dt
            
        # 4. Soft boundary clamp
        bed = float(sim.get_seabed_depth(self.pos[0], self.pos[1]))
        self.pos[0] = np.clip(self.pos[0], 100.0, sim.MAP_SIZE - 100.0)
        self.pos[1] = np.clip(self.pos[1], 100.0, sim.MAP_SIZE - 100.0)
        self.pos[2] = np.clip(self.pos[2], bed + 3.0, -2.0)

_robots = []

def init_physics():
    global _robots
    _robots = []
    print("[Robot Physics] Initialized NumPy-based robot kinematics engine.")
    return True

def spawn_auvs(n=None):
    global _robots
    if n is None:
        n = sim.NUM_AUVS
    _robots = []
    zones = ["Outer Yard", "Channel", "Inner Harbor"]
    for i in range(n):
        zone = zones[i % len(zones)]
        _robots.append(RobotNode(node_id=i + 1, initial_zone=zone))
    return get_auv_states()

def get_auv_states():
    return [r.get_state() for r in _robots]

def step_auvs(frame, dt=1.0/60.0, active_violations=None, fleet=None):
    for r in _robots:
        r.update(frame, dt, active_violations=active_violations, fleet=fleet)
    return get_auv_states()

def close_physics():
    global _robots
    _robots = []
    print("[Robot Physics] Physics engine shut down.")
