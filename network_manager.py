import numpy as np
import random
import math

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_COMM_RANGE = 1200.0    # meters — hydrophone acoustic link radius for AUVs
DEFAULT_ROBOT_SPEED = 2.5      # m/s — average underwater cruising speed
LOC_X = 5000                   # Line of Control boundary

# ─────────────────────────────────────────────────────────────────────────────
# 1. RobotNode — Autonomous Underwater Vehicle (AUV) / Mobile Robot Node
# ─────────────────────────────────────────────────────────────────────────────
class RobotNode:
    """
    Represents a mobile underwater robot (AUV) capable of patrolling harbor zones,
    maintaining acoustic communication links, and relaying vessel telemetry.
    """
    def __init__(self, robot_id, name, pos_3d, assigned_zone="outer_yard",
                 comm_range=DEFAULT_COMM_RANGE, speed_mps=DEFAULT_ROBOT_SPEED):
        self.id = int(robot_id)
        self.name = str(name)
        self.pos = np.array(pos_3d, dtype=float)  # [x, y, z]
        self.assigned_zone = assigned_zone
        self.comm_range = float(comm_range)
        self.speed_mps = float(speed_mps)

        # Operational State
        self.battery = 100.0                  # Percentage (0 - 100%)
        self.discharge_rate = 0.005           # % consumed per step
        self.status = "PATROLLING"            # PATROLLING | RELAYING | RECHARGING | IDLE
        self.sensor_connected = False
        self.active_links_count = 0
        self.connected_vessels = []
        self.connected_sensors = []

        # Patrol Route / Movement
        self.patrol_waypoints = []
        self.waypoint_index = 0
        self.target_pos = None

    def set_patrol_waypoints(self, waypoints):
        """Set a list of 3D waypoints for cyclic patrol movement."""
        self.patrol_waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        if self.patrol_waypoints:
            self.waypoint_index = 0
            self.target_pos = self.patrol_waypoints[0]

    def update_position(self, speed_mult=1.0):
        """Move toward current waypoint and consume battery."""
        if self.battery <= 5.0:
            self.status = "RECHARGING"
            return

        if self.patrol_waypoints and self.target_pos is not None:
            direction = self.target_pos - self.pos
            dist = np.linalg.norm(direction)

            if dist < 20.0:  # Waypoint reached
                self.waypoint_index = (self.waypoint_index + 1) % len(self.patrol_waypoints)
                self.target_pos = self.patrol_waypoints[self.waypoint_index]
            else:
                step_dist = self.speed_mps * speed_mult
                unit_vec = direction / (dist + 1e-9)
                self.pos += unit_vec * min(step_dist, dist)

        self.consume_battery()

    def consume_battery(self, custom_rate=None):
        rate = custom_rate if custom_rate is not None else self.discharge_rate
        self.battery = max(0.0, self.battery - rate)

    def distance_to(self, target_pos_3d):
        return float(np.linalg.norm(self.pos - np.array(target_pos_3d, dtype=float)))

    def can_communicate_with(self, target_pos_3d, max_range=None):
        r = max_range if max_range is not None else self.comm_range
        return self.distance_to(target_pos_3d) <= r

    def get_zone(self):
        x = float(self.pos[0])
        if x > LOC_X:
            return "outer_yard"
        elif x > 700:
            return "inner_zone"
        return "harbor"

    def get_telemetry(self):
        """Serialize state for dashboard_data.json."""
        return {
            "id": self.id,
            "name": self.name,
            "x": round(float(self.pos[0]), 1),
            "y": round(float(self.pos[1]), 1),
            "z": round(float(self.pos[2]), 1),
            "zone": self.get_zone(),
            "battery": round(self.battery, 1),
            "status": self.status,
            "comm_range": self.comm_range,
            "sensor_connected": self.sensor_connected,
            "active_links": self.active_links_count
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CommLink — Acoustic Peer-to-Peer Link Model
# ─────────────────────────────────────────────────────────────────────────────
class CommLink:
    """
    Represents an underwater acoustic communication link between nodes
    (Robot ↔ Sensor, Robot ↔ Robot, Robot ↔ Vessel).
    """
    def __init__(self, link_id, source_id, source_type, source_pos,
                 target_id, target_type, target_pos):
        self.link_id = str(link_id)
        self.source_id = source_id
        self.source_type = str(source_type)
        self.source_pos = np.array(source_pos, dtype=float)
        self.target_id = target_id
        self.target_type = str(target_type)
        self.target_pos = np.array(target_pos, dtype=float)

        self.distance_m = float(np.linalg.norm(self.source_pos - self.target_pos))
        self.snr_db = 0.0
        self.status = "DISCONNECTED"
        self.throughput_kbps = 0.0

        self.evaluate_link()

    def evaluate_link(self, sound_speed=1500.0, ambient_noise=45.0):
        """Evaluate acoustic link quality based on transmission loss and distance."""
        dist = max(self.distance_m, 1.0)
        # Spherical spreading loss + absorption approximation
        tl = 20 * math.log10(dist) + 0.001 * dist
        tx_power = 175.0  # dB re 1 uPa @ 1m
        rx_level = tx_power - tl
        self.snr_db = round(rx_level - ambient_noise, 1)

        if self.snr_db > 12.0:
            self.status = "ACTIVE"
            self.throughput_kbps = round(min(64.0, max(8.0, 64.0 * (self.snr_db / 30.0))), 1)
        elif self.snr_db > 3.0:
            self.status = "DEGRADED"
            self.throughput_kbps = round(max(2.0, 16.0 * (self.snr_db / 12.0)), 1)
        else:
            self.status = "DISCONNECTED"
            self.throughput_kbps = 0.0

    def get_telemetry(self):
        """Serialize link state for dashboard_data.json."""
        return {
            "id": self.link_id,
            "source": f"{self.source_type.title()}-{self.source_id}",
            "target": f"{self.target_type.title()}-{self.target_id}",
            "source_id": self.source_id,
            "target_id": self.target_id,
            "distance_m": round(self.distance_m, 1),
            "snr_db": self.snr_db,
            "status": self.status,
            "throughput_kbps": self.throughput_kbps
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. NetworkManager — Multi-Robot Swarm & Network Controller
# ─────────────────────────────────────────────────────────────────────────────
class NetworkManager:
    """
    Manages the underwater robot fleet, computes dynamic acoustic communication links,
    and aggregates network telemetry for real-time dashboard export.
    """
    def __init__(self):
        self.robots = []
        self.active_links = []
        self.deploy_default_robots()

    def deploy_default_robots(self):
        """Deploy default fleet of 6 AUV robot nodes across key harbor zones."""
        self.robots = []

        # 2 Robots in Outer Yard (X: 5500 - 9000)
        r1 = RobotNode(101, "AUV-Alpha", [6500, 3000, -15], assigned_zone="outer_yard")
        r1.set_patrol_waypoints([[6500, 2000, -15], [8500, 3500, -15], [7500, 5000, -15], [6000, 3500, -15]])

        r2 = RobotNode(102, "AUV-Bravo", [7500, 7000, -12], assigned_zone="outer_yard")
        r2.set_patrol_waypoints([[7500, 6000, -12], [9000, 7500, -12], [8000, 9000, -12], [6500, 7500, -12]])

        # 2 Robots along LOC Wall (X ~ 5000)
        r3 = RobotNode(103, "AUV-Charlie", [5000, 2500, -10], assigned_zone="loc_boundary")
        r3.set_patrol_waypoints([[5000, 1000, -10], [5000, 4500, -10]])

        r4 = RobotNode(104, "AUV-Delta", [5000, 7500, -10], assigned_zone="loc_boundary")
        r4.set_patrol_waypoints([[5000, 5500, -10], [5000, 9000, -10]])

        # 2 Robots in Inner Zone (X: 1000 - 4500)
        r5 = RobotNode(105, "AUV-Echo", [3000, 3500, -8], assigned_zone="inner_zone")
        r5.set_patrol_waypoints([[2000, 2500, -8], [4000, 4500, -8], [2500, 5500, -8]])

        r6 = RobotNode(106, "AUV-Foxtrot", [2500, 7500, -8], assigned_zone="inner_zone")
        r6.set_patrol_waypoints([[1500, 6500, -8], [3500, 8500, -8], [2000, 9000, -8]])

        self.robots = [r1, r2, r3, r4, r5, r6]

    def update_network_state(self, sensors, fleet, frame, speed_mult=1.0):
        """
        Main animation step update:
        1. Updates robot positions along patrol routes.
        2. Computes links between Robots ↔ Sensors, Robots ↔ Robots, and Robots ↔ Vessels.
        """
        # Step 1: Update robot movement
        for robot in self.robots:
            robot.update_position(speed_mult=speed_mult)
            robot.sensor_connected = False
            robot.active_links_count = 0
            robot.connected_sensors = []
            robot.connected_vessels = []

        self.active_links = []

        # Step 2: Robot ↔ Sensor links
        if sensors:
            for robot in self.robots:
                for idx, (sx, sy, sz) in enumerate(sensors):
                    sensor_id = idx + 1
                    dist = robot.distance_to([sx, sy, sz])
                    if dist <= robot.comm_range:
                        link_id = f"link_R{robot.id}_S{sensor_id}"
                        link = CommLink(link_id, robot.id, "robot", robot.pos,
                                        sensor_id, "sensor", [sx, sy, sz])
                        if link.status in ("ACTIVE", "DEGRADED"):
                            self.active_links.append(link)
                            robot.sensor_connected = True
                            robot.active_links_count += 1
                            robot.connected_sensors.append(sensor_id)

        # Step 3: Robot ↔ Robot inter-node links
        num_robots = len(self.robots)
        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                r1 = self.robots[i]
                r2 = self.robots[j]
                dist = r1.distance_to(r2.pos)
                if dist <= max(r1.comm_range, r2.comm_range):
                    link_id = f"link_R{r1.id}_R{r2.id}"
                    link = CommLink(link_id, r1.id, "robot", r1.pos,
                                    r2.id, "robot", r2.pos)
                    if link.status in ("ACTIVE", "DEGRADED"):
                        self.active_links.append(link)
                        r1.active_links_count += 1
                        r2.active_links_count += 1

        # Step 4: Robot ↔ Vessel links
        if fleet:
            for robot in self.robots:
                for v in fleet:
                    v_pos = [v['pos'][0], v['pos'][1], v.get('depth', -5)]
                    dist = robot.distance_to(v_pos)
                    if dist <= robot.comm_range:
                        link_id = f"link_R{robot.id}_V{v['id']}"
                        link = CommLink(link_id, robot.id, "robot", robot.pos,
                                        v['id'], "vessel", v_pos)
                        if link.status in ("ACTIVE", "DEGRADED"):
                            self.active_links.append(link)
                            robot.connected_vessels.append(v['id'])
                            robot.active_links_count += 1

    def get_network_json_data(self):
        """Construct JSON dictionary for export to dashboard_data.json."""
        robot_telemetry = [r.get_telemetry() for r in self.robots]
        link_telemetry = [l.get_telemetry() for l in self.active_links]

        active_count = sum(1 for l in self.active_links if l.status == "ACTIVE")
        degraded_count = sum(1 for l in self.active_links if l.status == "DEGRADED")
        total_links = len(self.active_links)

        avg_throughput = (np.mean([l.throughput_kbps for l in self.active_links])
                          if self.active_links else 0.0)

        return {
            "robots": robot_telemetry,
            "network_links": link_telemetry,
            "network_stats": {
                "total_robots": len(self.robots),
                "active_links_count": active_count,
                "degraded_links_count": degraded_count,
                "total_links": total_links,
                "avg_throughput_kbps": round(float(avg_throughput), 1),
                "packet_delivery_rate": round(0.95 if total_links > 0 else 0.0, 2)
            }
        }
