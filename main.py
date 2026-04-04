import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')                                      # MUST be before pyplot import

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D                      # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
import json, os

import simulator as sim
import mlmodel as ml
import traffic_algo as tms

# ─────────────────────────────────────────────────────────────────────────────
# 0. Dashboard JSON path
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dashboard_data.json")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Boot
# ─────────────────────────────────────────────────────────────────────────────
sensors = sim.generate_sensors()
fleet   = sim.generate_fleet()
print(f"[Boot] {len(sensors)} sensors deployed | {len(fleet)} vessels spawned")
print("[Boot] All ships spawn on the right — passing Outer Yard → LOC → Inner Zone.")
print("[Boot] ML model will auto-train once 100 detections accumulate.")

MAP   = sim.MAP_SIZE        # 10 000 m
SPEED_MULT = 8              # visual speed multiplier — ships traverse the map at a comfortable pace
LOC_X = sim.LOC_X          # 5 000 m  — Line of Control

# ─────────────────────────────────────────────────────────────────────────────
# 2. Design tokens
# ─────────────────────────────────────────────────────────────────────────────
C_WIN           = '#F0F4F8'
C_PANEL         = '#FFFFFF'
C_BORDER        = '#D9E2EC'
C_GRID          = '#E8EEF4'
C_TITLE         = '#1A2B4C'
C_SUBTITLE      = '#4A6FA5'
C_OCEAN_TOP     = '#AED9E0'
C_SENSOR_IDLE   = '#0077B6'
C_SENSOR_ACTIVE = '#D62828'
C_CRANE         = '#E07B39'
C_HARBOR        = '#C44B4B'
C_LOC           = '#FF6B35'   # LOC wall colour
C_OUTER_FLOOR   = '#D4E6F1'   # outer yard floor tint
C_INNER_FLOOR   = '#D5F5E3'   # inner zone floor tint

TYPE_COLORS = {
    'Cargo Ship':     '#E07B39',
    'Tanker':         '#C0392B',
    'Cruiser':        '#2E9E52',
    'Ferry':          '#9B59B6',
    'Speedboat':      '#3A7DD1',
    'Fishing Vessel': '#16A085',
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Tkinter root window
# ─────────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("⚓ Harbor Traffic Control — Outer Yard | LOC | Inner Zone")
root.configure(bg=C_WIN)
root.state('zoomed')

# ── Top status bar ──────────────────────────────────────────────────────────
top_bar = tk.Frame(root, bg='#1A2B4C', height=40)
top_bar.pack(side=tk.TOP, fill=tk.X)

tk.Label(top_bar,
         text="  ⚓  Harbor Traffic Control System  |  Outer Yard  —  LOC  —  Inner Zone",
         font=('Segoe UI', 11, 'bold'),
         bg='#1A2B4C', fg='#FFFFFF').pack(side=tk.LEFT, padx=10, pady=7)

frame_var      = tk.StringVar(value="Frame: 0000")
sensor_var     = tk.StringVar(value="Active Sensors: 0")
violation_var  = tk.StringVar(value="Violations: 0")

def reset_view():
    ax.view_init(elev=28, azim=-55)
    canvas.draw_idle()

tk.Button(top_bar, text="⟲ Reset View", bg='#3A4C6C', fg='white',
          font=('Segoe UI', 9, 'bold'), bd=0, padx=8, cursor='hand2',
          command=reset_view).pack(side=tk.RIGHT, padx=14)

tk.Label(top_bar, textvariable=violation_var,
         font=('Segoe UI', 9, 'bold'), bg='#1A2B4C', fg='#FF6B6B').pack(side=tk.RIGHT, padx=14)
tk.Label(top_bar, textvariable=sensor_var,
         font=('Segoe UI', 9), bg='#1A2B4C', fg='#90CAF9').pack(side=tk.RIGHT, padx=14)
tk.Label(top_bar, textvariable=frame_var,
         font=('Segoe UI', 9), bg='#1A2B4C', fg='#90CAF9').pack(side=tk.RIGHT, padx=6)

# ── Bottom status bar ─────────────────────────────────────────────────────
bot_bar = tk.Frame(root, bg='#E2EAF4', height=24)
bot_bar.pack(side=tk.BOTTOM, fill=tk.X)

status_var = tk.StringVar(value="Simulation running…")
tk.Label(bot_bar, textvariable=status_var,
         font=('Segoe UI', 8), bg='#E2EAF4', fg='#4A6FA5').pack(side=tk.LEFT, padx=10)
tk.Label(bot_bar, text="LOC @ X=5000 m  |  Outer Yard (X>5000)  |  Inner Patrol Zone (X<5000)",
         font=('Segoe UI', 8), bg='#E2EAF4', fg='#4A6FA5').pack(side=tk.RIGHT, padx=10)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Matplotlib figure embedded in Tkinter
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor=C_PANEL)
ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')

ax.set_facecolor(C_PANEL)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = True
    pane.set_facecolor(C_PANEL)
    pane.set_edgecolor(C_BORDER)
    pane.set_alpha(1.0)

ax.xaxis._axinfo['grid'].update(color=C_GRID, linewidth=0.7)
ax.yaxis._axinfo['grid'].update(color=C_GRID, linewidth=0.7)
ax.zaxis._axinfo['grid'].update(color=C_GRID, linewidth=0.7)
ax.grid(True)

ax.set_xlim(0, MAP); ax.set_ylim(0, MAP); ax.set_zlim(-60, 70)
ax.tick_params(colors='#555555', labelsize=7.5)
for lbl in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label):
    lbl.set_color('#333333')
ax.set_xlabel("X — Easting (m)",  labelpad=10, fontsize=9)
ax.set_ylabel("Y — Northing (m)", labelpad=10, fontsize=9)
ax.set_zlabel("Depth (m)",         labelpad=8,  fontsize=9)
ax.view_init(elev=28, azim=-55)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Seabed & ocean surface
# ─────────────────────────────────────────────────────────────────────────────
Xg = np.linspace(0, MAP, 35)
Yg = np.linspace(0, MAP, 35)
Xg, Yg = np.meshgrid(Xg, Yg)
Zg = sim.get_seabed_depth(Xg, Yg)

ax.plot_surface(Xg, Yg, Zg, cmap='GnBu', alpha=0.65, edgecolor='none')
ax.plot_surface(Xg, Yg, np.zeros_like(Zg),
                color=C_OCEAN_TOP, alpha=0.18, edgecolor='none')

# ─────────────────────────────────────────────────────────────────────────────
# 6. Zone floor shading
# ─────────────────────────────────────────────────────────────────────────────
# Outer Yard floor (right half): X in [LOC_X, MAP]
Xo = np.array([[LOC_X, MAP], [LOC_X, MAP]])
Yo = np.array([[0, 0], [MAP, MAP]])
ax.plot_surface(Xo, Yo, np.zeros_like(Xo) - 0.5,
                color='#F0A500', alpha=0.08, edgecolor='none')

# Inner Zone floor (left half): X in [0, LOC_X]
Xi = np.array([[0, LOC_X], [0, LOC_X]])
Yi = np.array([[0, 0], [MAP, MAP]])
ax.plot_surface(Xi, Yi, np.zeros_like(Xi) - 0.5,
                color='#1A8C5B', alpha=0.08, edgecolor='none')

# ─────────────────────────────────────────────────────────────────────────────
# 7. LOC wall  (vertical curtain at X = LOC_X)
# ─────────────────────────────────────────────────────────────────────────────
loc_verts = [
    [(LOC_X, 0,   -5), (LOC_X, MAP, -5),
     (LOC_X, MAP,  60), (LOC_X, 0,   60)]
]
loc_poly = Poly3DCollection(loc_verts, alpha=0.18)
loc_poly.set_facecolor(C_LOC)
loc_poly.set_edgecolor(C_LOC)
ax.add_collection3d(loc_poly)

# LOC edge lines (top + bottom of curtain)
ax.plot([LOC_X, LOC_X], [0, MAP], [60, 60],
        color=C_LOC, linewidth=2.5, linestyle='--', alpha=0.85)
ax.plot([LOC_X, LOC_X], [0, MAP], [-5, -5],
        color=C_LOC, linewidth=1.5, linestyle=':', alpha=0.60)

# LOC label
ax.text(LOC_X + 80, MAP * 0.5, 65,
        "◀  LOC — Line of Control  ▶",
        color=C_LOC, fontsize=9, fontweight='bold', alpha=0.95)

# Zone labels
ax.text(LOC_X + 600, MAP * 0.85, 55,
        "OUTER YARD ZONE", color='#B07000',
        fontsize=10, fontweight='bold', alpha=0.80)
ax.text(200, MAP * 0.85, 55,
        "INNER / PATROL ZONE", color='#1A6B45',
        fontsize=10, fontweight='bold', alpha=0.80)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Harbor infrastructure + color-coded dock zones
# ─────────────────────────────────────────────────────────────────────────────
HARBOR_X_VIZ = 700

def draw_box(axis, x0, y0, z0, dx, dy, dz, color, alpha=0.88):
    verts = [
        [(x0,y0,z0),(x0+dx,y0,z0),(x0+dx,y0+dy,z0),(x0,y0+dy,z0)],
        [(x0,y0,z0+dz),(x0+dx,y0,z0+dz),(x0+dx,y0+dy,z0+dz),(x0,y0+dy,z0+dz)],
        [(x0,y0,z0),(x0+dx,y0,z0),(x0+dx,y0,z0+dz),(x0,y0,z0+dz)],
        [(x0,y0+dy,z0),(x0+dx,y0+dy,z0),(x0+dx,y0+dy,z0+dz),(x0,y0+dy,z0+dz)],
        [(x0,y0,z0),(x0,y0+dy,z0),(x0,y0+dy,z0+dz),(x0,y0,z0+dz)],
        [(x0+dx,y0,z0),(x0+dx,y0+dy,z0),(x0+dx,y0+dy,z0+dz),(x0+dx,y0,z0+dz)],
    ]
    poly = Poly3DCollection(verts, alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor('#BBBBBB')
    axis.add_collection3d(poly)

def draw_dock_zone(y0, y1, color, name, ship_type_label):
    """Draw a colored floor patch for a dock zone and label it."""
    DX = HARBOR_X_VIZ
    verts = [[(0,y0,0.3),(DX,y0,0.3),(DX,y1,0.3),(0,y1,0.3)]]
    poly = Poly3DCollection(verts, alpha=0.38)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    ax.add_collection3d(poly)
    # Border lines
    ax.plot([0, DX], [y0, y0], [0.4, 0.4], color=color, linewidth=1.8, alpha=0.9)
    ax.plot([0, DX], [y1, y1], [0.4, 0.4], color=color, linewidth=1.8, alpha=0.9)
    # Zone label (center)
    mid_y = (y0 + y1) / 2
    ax.text(30, mid_y, 6, f"{name}\n{ship_type_label}",
            color=color, fontsize=6.5, fontweight='bold', alpha=0.95,
            ha='left', va='center')

# Harbor water surface
X_h, Y_h = np.meshgrid(np.linspace(0, HARBOR_X_VIZ, 2), np.linspace(0, MAP, 2))
ax.plot_surface(X_h, Y_h, np.zeros_like(X_h), color=C_HARBOR, alpha=0.07)
ax.text(30, MAP/2, 38, "HARBOR", color=C_HARBOR,
        fontsize=11, fontweight='bold', alpha=0.80)

# ── Colored dock zone floor patches (ordered bottom→top in Y) ──────────────
dock_zones = [
    (1550, 2450, '#E07B39', 'DOCK A', 'Cargo Ship'),
    (2600, 3100, '#C0392B', 'DOCK T', 'Tanker'),
    (3300, 4950, '#3A7DD1', 'DOCK C', 'Speedboat'),
    (5100, 6200, '#16A085', 'DOCK V', 'Fishing Vessel'),
    (6400, 7250, '#9B59B6', 'DOCK F', 'Ferry'),
    (7400, 8550, '#2E9E52', 'DOCK B', 'Cruiser'),
]
for y0d, y1d, dc, dn, dt in dock_zones:
    draw_dock_zone(y0d, y1d, dc, dn, dt)

# ── Structural buildings per dock ─────────────────────────────────────────
dock_buildings = [
    (40, 1600, 0, 380, 800, 20, '#A0856C'),    # Cargo
    (40, 2650, 0, 380, 400, 24, '#8B2020'),    # Tanker
    (40, 3350, 0, 300, 550, 14, '#2B5FA8'),    # Speedboat (lower)
    (40, 4000, 0, 300, 500, 14, '#2B5FA8'),
    (40, 4550, 0, 300, 350, 14, '#2B5FA8'),
    (40, 5150, 0, 320, 950, 16, '#0E6B5E'),    # Fishing
    (40, 6450, 0, 380, 750, 20, '#6C3483'),    # Ferry
    (40, 7450, 0, 400, 1050, 22, '#1A5C35'),   # Cruiser
]
for b in dock_buildings:
    draw_box(ax, *b)

# Cranes at major cargo/tanker docks
for cx, cy in [(440, 1800), (440, 2750)]:
    ax.plot([cx, cx],       [cy, cy], [0, 40],   color=C_CRANE, linewidth=4)
    ax.plot([cx, cx+200],   [cy, cy], [40, 40],  color=C_CRANE, linewidth=4)
    ax.plot([cx+200,cx+200],[cy, cy], [40, 10],  color=C_CRANE, linewidth=2, linestyle='--')

# Lighthouse (visible landmark at top of harbor)
lx, ly = 350, 9200
ax.plot([lx, lx], [ly, ly], [0, 55], color='#E0E0E0', linewidth=7, solid_capstyle='round')
ax.scatter([lx], [ly], [58], color='#FFD700', s=700, marker='o',
           edgecolors='#FFA500', linewidths=3, alpha=0.97, zorder=10)
ax.scatter([lx], [ly], [58], color='white', s=200, marker='o', zorder=11)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Sensor nodes
# ─────────────────────────────────────────────────────────────────────────────
SENSING_RANGE = sim.SENSING_RANGE
sensor_glow   = []
sensor_core   = []

for sx, sy, sz in sensors:
    bed_z = float(sim.get_seabed_depth(sx, sy))
    sg = ax.scatter(sx, sy, sz, color=C_SENSOR_IDLE, marker='o', s=320,
                    edgecolors='#90E0EF', linewidths=1.6, alpha=0.30, zorder=4)
    sc = ax.scatter(sx, sy, sz, color=C_SENSOR_IDLE, marker='o', s=110,
                    edgecolors='#023E8A', linewidths=1.0, zorder=5)
    ax.plot([sx, sx], [sy, sy], [bed_z, sz],
            color='#ADE8F4', linestyle=':', linewidth=0.9, alpha=0.55)
    sensor_glow.append(sg)
    sensor_core.append(sc)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Vessel artists
# ─────────────────────────────────────────────────────────────────────────────
v_plots = [ax.plot([], [], [], 'o',
                   markeredgecolor='#FFFFFF', markeredgewidth=1.2)[0]
           for _ in fleet]
v_texts = [ax.text(0, 0, 0, '', fontsize=7, fontweight='bold', color='#111111')
           for _ in fleet]

SIZE_MAP = {'Cargo Ship': 28, 'Tanker': 32, 'Cruiser': 18, 'Ferry': 22, 'Speedboat': 10, 'Fishing Vessel': 13}

# ─────────────────────────────────────────────────────────────────────────────
# 11. Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_proxies = [
    Line2D([0],[0], marker='s', color='none', markerfacecolor='#E07B39',
           markersize=11, label='Cargo  (HALT)',         linestyle='None', markeredgecolor='#555'),
    Line2D([0],[0], marker='D', color='none', markerfacecolor='#C0392B',
           markersize=11, label='Tanker (HALT)',         linestyle='None', markeredgecolor='#555'),
    Line2D([0],[0], marker='P', color='none', markerfacecolor='#2E9E52',
           markersize=11, label='Cruiser (SLOW)',        linestyle='None', markeredgecolor='#555'),
    Line2D([0],[0], marker='h', color='none', markerfacecolor='#9B59B6',
           markersize=11, label='Ferry  (SLOW)',         linestyle='None', markeredgecolor='#555'),
    Line2D([0],[0], marker='>', color='none', markerfacecolor='#3A7DD1',
           markersize=9,  label='Speedboat (CONTINUE)',  linestyle='None', markeredgecolor='#555'),
    Line2D([0],[0], marker='*', color='none', markerfacecolor='#16A085',
           markersize=11, label='Fishing (CONTINUE)',    linestyle='None', markeredgecolor='#555'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=C_SENSOR_IDLE,
           markersize=9,  label='Sensor — idle',         linestyle='None', markeredgecolor='#023E8A'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=C_SENSOR_ACTIVE,
           markersize=9,  label='Sensor — active',       linestyle='None', markeredgecolor='#7D0000'),
    Line2D([0],[0], color=C_LOC, linewidth=2, linestyle='--', label='LOC — Line of Control'),
]
leg = ax.legend(handles=legend_proxies,
                loc='upper right',
                fancybox=True, framealpha=0.95,
                labelcolor='#111111',
                facecolor='#FFFFFF',
                edgecolor=C_BORDER,
                fontsize=8,
                title='  Fleet & Zones',
                title_fontsize=9,
                borderpad=0.9, labelspacing=0.6)
leg.get_title().set_color(C_TITLE)
leg.get_title().set_fontweight('bold')

# ─────────────────────────────────────────────────────────────────────────────
# 12. Animation loop
# ─────────────────────────────────────────────────────────────────────────────
def master_loop(frame):
    global fleet

    frame_var.set(f"Frame: {frame:04d}")

    # ── DYNAMIC UWSN PIPELINE ──────────────────────────────────────────────
    sensor_active, total_detections = sim.collect_sensor_detections(
        fleet, sensors, frame
    )
    ml_status, ml_is_live = ml.update_model_from_collected_data()

    # Traffic manager now returns (fleet, violations)
    result = tms.manage_traffic_from_csv(fleet)
    fleet, violations = result if isinstance(result, tuple) else (result, [])
    # ──────────────────────────────────────────────────────────────────────

    # Sensor blink visuals + per-vessel hit counting
    ship_xyz           = np.array([[v['pos'][0], v['pos'][1], v['depth']] for v in fleet])
    active_count       = 0
    vessel_sensor_hits = np.zeros(len(fleet), dtype=int)

    for i, (sx, sy, sz) in enumerate(sensors):
        dists = np.linalg.norm(ship_xyz - np.array([sx, sy, sz]), axis=1)
        hits  = dists < SENSING_RANGE
        vessel_sensor_hits[hits] += 1

        active = sensor_active[i]
        if active:
            active_count += 1
            blink_on   = (frame % 6) < 3
            fg         = C_SENSOR_ACTIVE if blink_on else '#FF6B6B'
            edge       = '#7D0000'  if blink_on else C_SENSOR_ACTIVE
            glow_alpha = 0.90 if blink_on else 0.50
        else:
            fg         = C_SENSOR_IDLE
            edge       = '#023E8A'
            glow_alpha = 0.30

        sensor_glow[i].set_facecolor(fg)
        sensor_glow[i].set_edgecolor(edge)
        sensor_glow[i].set_alpha(glow_alpha)
        sensor_core[i].set_facecolor(fg)
        sensor_core[i].set_edgecolor(edge)

    sensor_var.set(f"Active Sensors: {active_count} / {len(sensors)}")
    violation_var.set(f"⚠ Violations: {len(violations)}")

    ml_prefix = "ML LIVE" if ml_is_live else "Collecting"
    status_var.set(
        f"Frame {frame:04d}  |  {ml_prefix}: {total_detections} detections  |  "
        f"{active_count} sensors  |  Violations: {len(violations)}"
    )

    # Vessel movement
    dashboard_vessels = []
    for idx, v in enumerate(fleet):
        # Move only if speed > 0 (apply visual multiplier)
        if v['current_speed'] > 0:
            rad = np.radians(v['angle'])
            v['pos'] += np.array([np.cos(rad), np.sin(rad)]) * v['current_speed'] * SPEED_MULT

            # Boundary: clamp Y; X stays in [0, MAP]
            if not (0 < v['pos'][0] < MAP):
                v['angle'] = 180 - v['angle']
                v['pos'][0] = np.clip(v['pos'][0], 10, MAP - 10)
            if not (0 < v['pos'][1] < MAP):
                v['angle'] = -v['angle']
                v['pos'][1] = np.clip(v['pos'][1], 10, MAP - 10)

        cfg    = sim.VEHICLE_TYPES[v['type']]
        m_size = SIZE_MAP.get(v['type'], 12)

        # Overspeed ships flash red-orange regardless of type
        color = '#FF2222' if v.get('overspeed') else TYPE_COLORS[v['type']]

        v_plots[idx].set_data_3d([v['pos'][0]], [v['pos'][1]], [v['depth']])
        v_plots[idx].set_color(color)
        v_plots[idx].set_marker(cfg['marker'])
        v_plots[idx].set_markersize(m_size)

        # Label: TYPE-ID [ZONE] [DIR]
        zone_tag = {'outer_yard': 'OY', 'inner_zone': 'IZ', 'harbor': 'H'}.get(
            v.get('zone', 'outer_yard'), '?')
        dir_tag  = v.get('direction', 'IN')[:2]
        v_texts[idx].set_position((v['pos'][0], v['pos'][1]))
        v_texts[idx].set_3d_properties(22)
        v_texts[idx].set_text(f"{v['type'][:5]}-{v['id']} [{zone_tag}][{dir_tag}]")

        n_sensors = int(vessel_sensor_hits[idx])
        dashboard_vessels.append({
            "id":              v['id'],
            "type":            v['type'],
            "pred_type":       v.get('pred_type', '?'),
            "pred_weight":     int(v.get('pred_weight', 0)),
            "state":           v['state'],
            "cmd":             v['tms_command'],
            "zone":            v.get('zone', 'outer_yard'),
            "direction":       v.get('direction', 'INBOUND'),
            "overspeed":       v.get('overspeed', False),
            "halt_time":       v.get('halt_time', 0),
            "dock_slot":       v.get('dock_slot'),
            "speed_knots":     round(v.get('speed_knots', 0), 1),
            "x":               round(float(v['pos'][0]), 1),
            "y":               round(float(v['pos'][1]), 1),
            "sensor_detected": n_sensors > 0,
            "sensor_count":    n_sensors,
        })

    try:
        with open(DASHBOARD_JSON, 'w') as _f:
            json.dump({
                "frame":          frame,
                "active_sensors": active_count,
                "total_sensors":  len(sensors),
                "ml_status":      ml_status,
                "violations":     violations,
                "vessels":        dashboard_vessels,
            }, _f)
    except Exception:
        pass

    return v_plots + v_texts + sensor_glow + sensor_core

# ─────────────────────────────────────────────────────────────────────────────
# 13. Run
# ─────────────────────────────────────────────────────────────────────────────
ani = FuncAnimation(fig, master_loop, frames=6000, interval=1, blit=True)
root.mainloop()