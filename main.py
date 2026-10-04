import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')                                      # MUST be before pyplot import

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D                      # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import numpy as np
import json, os, time, webbrowser, http.server, socketserver, threading

LAST_DASHBOARD_WRITE_TIME = 0.0

import simulator as sim
import mlmodel as ml
import traffic_algo as tms
import auv
import network

# ─────────────────────────────────────────────────────────────────────────────
# 0. Dashboard JSON path & HTTP Server
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dashboard_data.json")
DASHBOARD_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dashboard_data.js")

DASHBOARD_PORT = 8000

def _run_dashboard_server():
    global DASHBOARD_PORT
    base_dir = os.path.dirname(os.path.abspath(__file__))
    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=base_dir, **kwargs)
        def log_message(self, format, *args):
            pass

    for p in range(8000, 8025):
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", p), QuietHTTPRequestHandler)
            DASHBOARD_PORT = p
            print(f"[Dashboard Server] Serving live harbor telemetry at http://127.0.0.1:{p}/dashboard.html")
            httpd.serve_forever()
            break
        except OSError:
            continue

_server_thread = threading.Thread(target=_run_dashboard_server, daemon=True)
_server_thread.start()

def open_dashboard():
    """Opens live web dashboard in default web browser asynchronously."""
    url = f"http://127.0.0.1:{DASHBOARD_PORT}/dashboard.html"
    print(f"[Dashboard] Opening live web dashboard: {url}")
    def _open():
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[Dashboard] Error opening browser: {e}")
    threading.Thread(target=_open, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Boot
# ─────────────────────────────────────────────────────────────────────────────
sensors = sim.generate_sensors()
fleet   = sim.generate_fleet()
print(f"[Boot] {len(sensors)} sensors deployed | {len(fleet)} vessels spawned")
print("[Boot] All ships spawn on the right - passing Outer Yard -> LOC -> Inner Zone.")
print("[Boot] ML model will auto-train once 100 detections accumulate.")
auv.init_physics()
auv.spawn_auvs()

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
# 2b. Dedicated Desktop Network Operations Dashboard Widget
# ─────────────────────────────────────────────────────────────────────────────
class NetworkDashboard(tk.Frame):
    """
    Dedicated Desktop Network Operations Dashboard.
    Visualizes multi-hop topology, live packet hop telemetry,
    quantitative metrics, and dynamically streaming simulation logs.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, bg='#0B132B', width=380, bd=1, relief='solid', **kwargs)
        self.pack_propagate(False)

        # Header: State badge + Health score & meter bar
        hdr_frame = tk.Frame(self, bg='#111C38', bd=1, relief='ridge', padx=10, pady=8)
        hdr_frame.pack(fill='x', padx=8, pady=(8, 4))

        tk.Label(hdr_frame, text="📡 NETWORK STATUS & HEALTH", font=('Segoe UI', 10, 'bold'),
                 bg='#111C38', fg='#64FFDA').pack(anchor='w')

        badge_row = tk.Frame(hdr_frame, bg='#111C38')
        badge_row.pack(fill='x', pady=(4, 2))

        self.state_badge = tk.Label(badge_row, text="STABLE", font=('Segoe UI', 10, 'bold'),
                                    bg='#065F46', fg='#34D399', padx=10, pady=3, relief='solid', bd=1)
        self.state_badge.pack(side=tk.LEFT)

        self.health_score_lbl = tk.Label(badge_row, text="Health: 100.0%", font=('Segoe UI', 9, 'bold'),
                                         bg='#111C38', fg='#F8FAFC')
        self.health_score_lbl.pack(side=tk.RIGHT, padx=4)

        self.meter_canvas = tk.Canvas(hdr_frame, height=8, bg='#1E293B', highlightthickness=0)
        self.meter_canvas.pack(fill='x', pady=(4, 4))
        self.meter_fill = self.meter_canvas.create_rectangle(0, 0, 340, 8, fill='#10B981', width=0)

        self.env_details_lbl = tk.Label(hdr_frame, text="Regime: Normal Season | Noise: 64.8 dB",
                                        font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8')
        self.env_details_lbl.pack(anchor='w')

        # Collapse Alert Panel (hidden until actual COLLAPSED state is triggered)
        self.collapse_alert = tk.Label(
            self,
            text="🚨 SENSOR NETWORK COLLAPSED 🚨\nGateway connectivity severed under severe conditions!\nRe-anchor sensors or revert to Normal Season.",
            font=('Segoe UI', 8, 'bold'), bg='#7F1D1D', fg='#FCA5A5',
            bd=2, relief='ridge', padx=8, pady=6, justify='center'
        )
        self._collapse_shown = False

        # Section 1: Hop Progression & Packet Flow Visualizer
        self.topo_frame = tk.Frame(self, bg='#111C38', bd=1, relief='ridge', padx=8, pady=6)
        self.topo_frame.pack(fill='x', padx=8, pady=4)

        tk.Label(self.topo_frame, text="HOP PROGRESSION & PACKET FLOW", font=('Segoe UI', 8, 'bold'),
                 bg='#111C38', fg='#38BDF8').pack(anchor='w')

        self.topo_canvas = tk.Canvas(self.topo_frame, height=66, bg='#070D1A',
                                     highlightthickness=1, highlightbackground='#1E293B')
        self.topo_canvas.pack(fill='x', pady=(4, 2))

        # Section 2: Current Packet Telemetry
        pkt_frame = tk.Frame(self, bg='#111C38', bd=1, relief='ridge', padx=8, pady=6)
        pkt_frame.pack(fill='x', padx=8, pady=4)

        tk.Label(pkt_frame, text="CURRENT PACKET TELEMETRY", font=('Segoe UI', 8, 'bold'),
                 bg='#111C38', fg='#38BDF8').pack(anchor='w')

        grid_pkt = tk.Frame(pkt_frame, bg='#111C38')
        grid_pkt.pack(fill='x', pady=(4, 2))

        tk.Label(grid_pkt, text="Packet ID:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=0, column=0, sticky='w')
        self.pkt_id_lbl = tk.Label(grid_pkt, text="--", font=('Segoe UI', 8, 'bold'), bg='#111C38', fg='#F8FAFC')
        self.pkt_id_lbl.grid(row=0, column=1, sticky='w', padx=(4, 12))

        tk.Label(grid_pkt, text="Status:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=0, column=2, sticky='w')
        self.pkt_status_lbl = tk.Label(grid_pkt, text="IDLE", font=('Segoe UI', 8, 'bold'), bg='#111C38', fg='#38BDF8')
        self.pkt_status_lbl.grid(row=0, column=3, sticky='w', padx=(4, 0))

        tk.Label(grid_pkt, text="Source:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=1, column=0, sticky='w')
        self.pkt_src_lbl = tk.Label(grid_pkt, text="--", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.pkt_src_lbl.grid(row=1, column=1, sticky='w', padx=(4, 12))

        tk.Label(grid_pkt, text="Current Hop:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=1, column=2, sticky='w')
        self.pkt_hop_lbl = tk.Label(grid_pkt, text="--", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.pkt_hop_lbl.grid(row=1, column=3, sticky='w', padx=(4, 0))

        tk.Label(grid_pkt, text="Total Hops:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=2, column=0, sticky='w')
        self.pkt_hops_lbl = tk.Label(grid_pkt, text="--", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.pkt_hops_lbl.grid(row=2, column=1, sticky='w', padx=(4, 12))

        tk.Label(grid_pkt, text="Delay / Latency:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=2, column=2, sticky='w')
        self.pkt_delay_lbl = tk.Label(grid_pkt, text="--", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.pkt_delay_lbl.grid(row=2, column=3, sticky='w', padx=(4, 0))

        tk.Label(grid_pkt, text="Link SNR:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=3, column=0, sticky='w')
        self.pkt_snr_lbl = tk.Label(grid_pkt, text="--", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.pkt_snr_lbl.grid(row=3, column=1, sticky='w', padx=(4, 12))

        tk.Label(grid_pkt, text="Tally (Del/Drop):", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=3, column=2, sticky='w')
        self.pkt_tally_lbl = tk.Label(grid_pkt, text="0 / 0", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.pkt_tally_lbl.grid(row=3, column=3, sticky='w', padx=(4, 0))

        # Section 3: Compact Quantitative Network Metrics
        metrics_frame = tk.Frame(self, bg='#111C38', bd=1, relief='ridge', padx=8, pady=6)
        metrics_frame.pack(fill='x', padx=8, pady=4)

        tk.Label(metrics_frame, text="NETWORK METRICS", font=('Segoe UI', 8, 'bold'),
                 bg='#111C38', fg='#38BDF8').pack(anchor='w')

        m_grid = tk.Frame(metrics_frame, bg='#111C38')
        m_grid.pack(fill='x', pady=(4, 2))

        tk.Label(m_grid, text="Packets Sent:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=0, column=0, sticky='w')
        self.m_sent = tk.Label(m_grid, text="0", font=('Segoe UI', 8, 'bold'), bg='#111C38', fg='#F8FAFC')
        self.m_sent.grid(row=0, column=1, sticky='w', padx=(4, 14))

        tk.Label(m_grid, text="Delivered:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=0, column=2, sticky='w')
        self.m_deliv = tk.Label(m_grid, text="0", font=('Segoe UI', 8, 'bold'), bg='#111C38', fg='#10B981')
        self.m_deliv.grid(row=0, column=3, sticky='w', padx=(4, 0))

        tk.Label(m_grid, text="Packets Dropped:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=1, column=0, sticky='w')
        self.m_drop = tk.Label(m_grid, text="0", font=('Segoe UI', 8, 'bold'), bg='#111C38', fg='#EF4444')
        self.m_drop.grid(row=1, column=1, sticky='w', padx=(4, 14))

        tk.Label(m_grid, text="Packet Loss %:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=1, column=2, sticky='w')
        self.m_loss = tk.Label(m_grid, text="0.0%", font=('Segoe UI', 8, 'bold'), bg='#111C38', fg='#EF4444')
        self.m_loss.grid(row=1, column=3, sticky='w', padx=(4, 0))

        tk.Label(m_grid, text="Average Delay:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=2, column=0, sticky='w')
        self.m_delay = tk.Label(m_grid, text="0.000 s", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.m_delay.grid(row=2, column=1, sticky='w', padx=(4, 14))

        tk.Label(m_grid, text="Avg Hop Count:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=2, column=2, sticky='w')
        self.m_hop = tk.Label(m_grid, text="0.0", font=('Segoe UI', 8), bg='#111C38', fg='#F8FAFC')
        self.m_hop.grid(row=2, column=3, sticky='w', padx=(4, 0))

        tk.Label(m_grid, text="Active Nodes:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=3, column=0, sticky='w')
        self.m_active = tk.Label(m_grid, text="35 / 35", font=('Segoe UI', 8), bg='#111C38', fg='#10B981')
        self.m_active.grid(row=3, column=1, sticky='w', padx=(4, 14))

        tk.Label(m_grid, text="Failed / Snapped:", font=('Segoe UI', 8), bg='#111C38', fg='#94A3B8').grid(row=3, column=2, sticky='w')
        self.m_failed = tk.Label(m_grid, text="0", font=('Segoe UI', 8), bg='#111C38', fg='#EF4444')
        self.m_failed.grid(row=3, column=3, sticky='w', padx=(4, 0))

        # Section 4: Live Monospace Simulation Event Log
        log_frame = tk.Frame(self, bg='#111C38', bd=1, relief='ridge', padx=8, pady=6)
        log_frame.pack(fill='both', expand=True, padx=8, pady=(4, 8))

        tk.Label(log_frame, text="SIMULATION EVENT LOG", font=('Segoe UI', 8, 'bold'),
                 bg='#111C38', fg='#38BDF8').pack(anchor='w')

        text_wrap = tk.Frame(log_frame, bg='#060C18')
        text_wrap.pack(fill='both', expand=True, pady=(4, 0))

        self.log_text = tk.Text(text_wrap, bg='#060C18', fg='#94A3B8', font=('Consolas', 8),
                                bd=0, height=8, wrap=tk.WORD)
        self.log_scroll = ttk.Scrollbar(text_wrap, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scroll.set)

        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill='both', expand=True)

        self.log_text.tag_configure('delivered', foreground='#34D399')
        self.log_text.tag_configure('dropped', foreground='#F87171')
        self.log_text.tag_configure('degraded', foreground='#FBBF24')
        self.log_text.tag_configure('trans', foreground='#38BDF8')
        self.log_text.tag_configure('squall', foreground='#F472B6')
        self.log_text.tag_configure('collapsed', foreground='#EF4444', font=('Consolas', 8, 'bold'))

        self._rendered_logs_count = 0

    def reset(self):
        """Reset dashboard display to initial clean state."""
        self.state_badge.config(text="STABLE", bg='#065F46', fg='#34D399')
        self.health_score_lbl.config(text="Health: 100.0%")
        self.meter_canvas.coords(self.meter_fill, 0, 0, 340, 8)
        self.meter_canvas.itemconfig(self.meter_fill, fill='#10B981')
        if self._collapse_shown:
            self.collapse_alert.pack_forget()
            self._collapse_shown = False
        self.topo_canvas.delete('all')
        self.topo_canvas.create_text(180, 33, text="Awaiting Acoustic Packet Transmission...", fill='#64748B', font=('Segoe UI', 8))
        self.pkt_id_lbl.config(text="--")
        self.pkt_status_lbl.config(text="IDLE", fg='#38BDF8')
        self.pkt_src_lbl.config(text="--")
        self.pkt_hop_lbl.config(text="--")
        self.pkt_hops_lbl.config(text="--")
        self.pkt_delay_lbl.config(text="--")
        self.pkt_snr_lbl.config(text="--")
        self.pkt_tally_lbl.config(text="0 / 0")
        self.m_sent.config(text="0")
        self.m_deliv.config(text="0")
        self.m_drop.config(text="0")
        self.m_loss.config(text="0.0%")
        self.m_delay.config(text="0.000 s")
        self.m_hop.config(text="0.0")
        self.m_active.config(text="35 / 35")
        self.m_failed.config(text="0")
        self.log_text.delete('1.0', tk.END)
        self._rendered_logs_count = 0

    def draw_topology(self, pkt_info):
        self.topo_canvas.delete('all')
        if not pkt_info or not pkt_info.get('path'):
            self.topo_canvas.create_text(180, 33, text="Awaiting Acoustic Packet Transmission...", fill='#64748B', font=('Segoe UI', 8))
            return

        path = pkt_info['path']
        N = len(path)
        if N < 2:
            return

        margin_x = 32
        w_avail = 360 - 2 * margin_x
        dx = w_avail / (N - 1)
        y = 26

        curr_idx = pkt_info.get('current_hop_idx', 0)
        prog = pkt_info.get('hop_progress', 0.0)
        status = pkt_info.get('status', 'IN TRANSIT')

        # Links
        for i in range(N - 1):
            x1 = margin_x + i * dx
            x2 = margin_x + (i + 1) * dx
            if status == 'DROPPED' and i == curr_idx:
                line_color = '#EF4444'
                self.topo_canvas.create_line(x1 + 16, y, x2 - 16, y, fill=line_color, width=2, dash=(3, 2), arrow=tk.LAST)
                self.topo_canvas.create_text((x1 + x2) / 2, y - 9, text="✖", fill='#EF4444', font=('Segoe UI', 8, 'bold'))
            elif i < curr_idx or (status == 'DELIVERED' and i <= curr_idx):
                line_color = '#10B981'
                self.topo_canvas.create_line(x1 + 16, y, x2 - 16, y, fill=line_color, width=2, arrow=tk.LAST)
            elif i == curr_idx:
                line_color = '#00E5FF'
                self.topo_canvas.create_line(x1 + 16, y, x2 - 16, y, fill=line_color, width=2, dash=(4, 2), arrow=tk.LAST)
            else:
                line_color = '#334155'
                self.topo_canvas.create_line(x1 + 16, y, x2 - 16, y, fill=line_color, width=1.5, arrow=tk.LAST)

        # Nodes
        for i, node_name in enumerate(path):
            x = margin_x + i * dx
            if i == 0:
                box_fill = '#0F2847'
                outline = '#00B4D8'
                tag = f"SRC: {node_name}"
                fg = '#90E0EF'
            elif i == N - 1:
                if status == 'DELIVERED':
                    box_fill = '#064E3B'
                    outline = '#10B981'
                    fg = '#A7F3D0'
                else:
                    box_fill = '#3B2904'
                    outline = '#F59E0B'
                    fg = '#FDE68A'
                tag = f"DEST: {node_name}"
            else:
                if status == 'DROPPED' and i == curr_idx:
                    box_fill = '#450A0A'
                    outline = '#EF4444'
                    fg = '#FCA5A5'
                elif i <= curr_idx:
                    box_fill = '#1E293B'
                    outline = '#38BDF8'
                    fg = '#BAE6FD'
                else:
                    box_fill = '#111827'
                    outline = '#475569'
                    fg = '#94A3B8'
                tag = f"H{i}: {node_name}"

            self.topo_canvas.create_rectangle(x - 20, y - 11, x + 20, y + 11, fill=box_fill, outline=outline, width=1.5)
            self.topo_canvas.create_text(x, y, text=tag, fill=fg, font=('Segoe UI', 6, 'bold'))

        # Packet indicator
        if status == 'IN TRANSIT' and curr_idx < N - 1:
            x_from = margin_x + curr_idx * dx
            x_to = margin_x + (curr_idx + 1) * dx
            xp = (x_from + 16) + prog * ((x_to - 16) - (x_from + 16))
            self.topo_canvas.create_oval(xp - 4, y - 4, xp + 4, y + 4, fill='#FF1744', outline='#FFFFFF', width=1.5)
            self.topo_canvas.create_text(xp, y + 15, text=f"● {pkt_info['id']}", fill='#FF80AB', font=('Segoe UI', 6, 'bold'))
        elif status == 'DELIVERED':
            x_dest = margin_x + (N - 1) * dx
            self.topo_canvas.create_text(x_dest, y + 16, text="✓ DELIVERED", fill='#34D399', font=('Segoe UI', 6, 'bold'))
        elif status == 'DROPPED':
            x_drop = margin_x + min(curr_idx, N - 1) * dx
            self.topo_canvas.create_text(x_drop, y + 16, text="✖ DROPPED", fill='#F87171', font=('Segoe UI', 6, 'bold'))

    def update_dashboard(self, stats):
        if not stats:
            return

        state = stats.get('network_state', 'STABLE')
        health = stats.get('health_score', 100.0)

        # 1. State badge
        state_colors = {
            'STABLE': ('#065F46', '#34D399'),
            'DEGRADED': ('#854D0E', '#FDE047'),
            'CRITICAL': ('#9A3412', '#FB923C'),
            'COLLAPSED': ('#7F1D1D', '#FCA5A5'),
        }
        bg_col, fg_col = state_colors.get(state, ('#065F46', '#34D399'))
        self.state_badge.config(text=state, bg=bg_col, fg=fg_col)
        self.health_score_lbl.config(text=f"Health: {health:.1f}%")

        # Meter
        meter_w = 340.0 * (health / 100.0)
        meter_col = '#10B981' if health >= 75 else ('#FBBF24' if health >= 50 else ('#F97316' if health >= 25 else '#EF4444'))
        self.meter_canvas.coords(self.meter_fill, 0, 0, meter_w, 8)
        self.meter_canvas.itemconfig(self.meter_fill, fill=meter_col)

        scen_name = stats.get('scenario', 'Normal Season')
        amb_noise = stats.get('ambient_noise_db', 65.0)
        self.env_details_lbl.config(text=f"Regime: {scen_name} | Noise: {amb_noise:.1f} dB")

        # Collapse banner
        if state == 'COLLAPSED':
            if not self._collapse_shown:
                self.collapse_alert.pack(fill='x', padx=8, pady=2, before=self.topo_frame)
                self._collapse_shown = True
        else:
            if self._collapse_shown:
                self.collapse_alert.pack_forget()
                self._collapse_shown = False

        # 2. Topology
        pkt = stats.get('current_packet')
        self.draw_topology(pkt)

        # 3. Packet Telemetry
        if pkt:
            self.pkt_id_lbl.config(text=pkt.get('id', '--'))
            st = pkt.get('status', 'IDLE')
            st_color = '#10B981' if st == 'DELIVERED' else ('#EF4444' if st == 'DROPPED' else '#38BDF8')
            self.pkt_status_lbl.config(text=st, fg=st_color)
            self.pkt_src_lbl.config(text=pkt.get('source', '--'))
            self.pkt_hop_lbl.config(text=f"{pkt.get('current_hop_name', '--')}")
            self.pkt_hops_lbl.config(text=f"{pkt.get('total_hops', 1)} hops")
            self.pkt_delay_lbl.config(text=f"{pkt.get('delay', 0.0):.3f} s")
            self.pkt_snr_lbl.config(text=f"{pkt.get('snr', 0.0):.1f} dB")
        deliv = stats.get('packets_delivered', 0)
        drop = stats.get('packets_dropped', 0)
        self.pkt_tally_lbl.config(text=f"{deliv} / {drop}")

        # 4. Metrics
        self.m_sent.config(text=str(stats.get('packets_sent', 0)))
        self.m_deliv.config(text=str(deliv))
        self.m_drop.config(text=str(drop))
        loss_pct = stats.get('packet_loss_pct', 0.0)
        self.m_loss.config(text=f"{loss_pct:.1f}%")
        self.m_delay.config(text=f"{stats.get('avg_delay_s', 0.0):.3f} s")
        self.m_hop.config(text=f"{stats.get('avg_hop_count', 0.0):.1f}")
        self.m_active.config(text=f"{stats.get('active_nodes_count', 0)} nodes")
        self.m_failed.config(text=f"{stats.get('failed_nodes_count', 0)} sensors")

        # 5. Simulation Logs
        all_logs = stats.get('recent_logs', [])
        if len(all_logs) > self._rendered_logs_count:
            new_entries = all_logs[self._rendered_logs_count:]
            for entry in new_entries:
                tag = None
                if 'delivered' in entry.lower():
                    tag = 'delivered'
                elif 'dropped' in entry.lower():
                    tag = 'dropped'
                elif 'degraded' in entry.lower() or 'strained' in entry.lower():
                    tag = 'degraded'
                elif 'squall' in entry.lower():
                    tag = 'squall'
                elif 'collapsed' in entry.lower():
                    tag = 'collapsed'
                elif 'transmitted' in entry.lower():
                    tag = 'trans'
                
                if tag:
                    self.log_text.insert(tk.END, entry + "\n", tag)
                else:
                    self.log_text.insert(tk.END, entry + "\n")
            self._rendered_logs_count = len(all_logs)
            self.log_text.see(tk.END)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tkinter root window & Menu Controls
# ─────────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("⚓ Harbor Traffic Control System — Underwater Sensor Network (UWSN)")
root.configure(bg=C_WIN)
try:
    root.state('zoomed')
except Exception:
    root.geometry("1400x850")

# Simulation execution state control
is_paused = False

def set_pause(paused):
    global is_paused
    is_paused = paused
    if 'btn_pause' in globals():
        btn_pause.config(text="▶ Resume" if is_paused else "⏸ Pause")
    status_var.set("Simulation paused" if is_paused else "Simulation running…")

def toggle_pause():
    set_pause(not is_paused)

def set_season(scenario_key):
    sim.set_active_scenario(scenario_key)
    sc = sim.get_active_scenario()
    season_var.set(f"{sc['badge']}")
    status_var.set(f"Switched scenario to: {sc['name']} ({sc['badge']})")
    network.log_network_event(f"Season switched to: {sc['name']}")

def toggle_scenario():
    curr = sim.ACTIVE_SCENARIO_KEY
    new_key = 'MONSOON' if curr == 'NORMAL' else 'NORMAL'
    set_season(new_key)

def trigger_squall_action():
    sim.set_active_scenario('MONSOON')
    sc = sim.get_active_scenario()
    season_var.set(f"{sc['badge']}")
    surge = sim.trigger_monsoon_squall()
    status_var.set(f"🌪️ Violent Gale Squall injected (+{surge:.2f} m/s)! Harsh currents surging!")

def reanchor_action():
    global _net_banner_visible
    sim.reset_all_sensors()
    sim.OCEAN_STATE['network_broken'] = False
    sim.OCEAN_STATE['squall_timer'] = 0.0
    network.on_reanchor()
    if '_net_banner_visible' in globals() and _net_banner_visible:
        net_broken_banner.place_forget()
        _net_banner_visible = False
    status_var.set("⚓ All 35 sensors re-anchored to seabed moorings. Network restored.")


def reset_simulation():
    global fleet, is_paused, _net_banner_visible
    sim.reset_all_sensors()
    sim.OCEAN_STATE['network_broken'] = False
    sim.OCEAN_STATE['squall_timer'] = 0.0
    network.reset_network()
    fleet = sim.generate_fleet()
    auv.init_physics()
    auv.spawn_auvs()
    if 'net_dashboard' in globals():
        net_dashboard.reset()
    if '_net_banner_visible' in globals() and _net_banner_visible:
        net_broken_banner.place_forget()
        _net_banner_visible = False
    status_var.set("Simulation reset to initial state.")
    canvas.draw_idle()

def reset_view():
    ax.view_init(elev=28, azim=-55)
    canvas.draw_idle()

_net_dash_visible = True
def toggle_net_dashboard():
    global _net_dash_visible
    if _net_dash_visible:
        net_dashboard.pack_forget()
        _net_dash_visible = False
    else:
        net_dashboard.pack(side=tk.RIGHT, fill=tk.Y)
        _net_dash_visible = True

def show_about_dialog():
    messagebox.showinfo(
        "About Harbor Traffic Control & UWSN Simulation",
        "Harbor Traffic Control System with Underwater Sensor Network (UWSN)\n\n"
        "• Key Capabilities:\n"
        "  - Multi-hop acoustic depth-based routing (DBR)\n"
        "  - Mobile autonomous underwater vehicle (AUV) relays\n"
        "  - Real-time Normal vs Monsoon environmental effects\n"
        "  - Stochastic wave surge, turbulence, and mooring strain physics\n"
        "  - Live desktop network topology & telemetry dashboard\n\n"
        "Status evaluation assesses gateway reachability, packet loss,\n"
        "hop availability, and mooring integrity:\n"
        "STABLE (≥75%) | DEGRADED (50–74%) | CRITICAL (20–49%) | COLLAPSED (<20%)."
    )

# ── Menubar ──────────────────────────────────────────────────────────────────
menubar = tk.Menu(root)

# 1. Simulation Menu
sim_menu = tk.Menu(menubar, tearoff=0)
sim_menu.add_command(label="▶  Start / Resume", command=lambda: set_pause(False), accelerator="Ctrl+S")
sim_menu.add_command(label="⏸  Pause", command=lambda: set_pause(True), accelerator="Space")
sim_menu.add_command(label="↺  Reset Simulation", command=reset_simulation, accelerator="Ctrl+R")
sim_menu.add_separator()
sim_menu.add_command(label="Exit", command=lambda: on_closing())
menubar.add_cascade(label="Simulation", menu=sim_menu)

# 2. Season Menu
season_menu = tk.Menu(menubar, tearoff=0)
season_menu.add_command(label="☀️  Normal Season", command=lambda: set_season('NORMAL'))
season_menu.add_command(label="⛈️  Monsoon Season", command=lambda: set_season('MONSOON'))
season_menu.add_separator()
season_menu.add_command(label="🌪️  Trigger Gale Squall Surge", command=trigger_squall_action)
season_menu.add_command(label="⚓  Re-Anchor Sensors", command=reanchor_action)
menubar.add_cascade(label="Season", menu=season_menu)

# 3. View Menu
view_menu = tk.Menu(menubar, tearoff=0)
view_menu.add_command(label="⟲  Reset 3D View Angle", command=reset_view)
view_menu.add_command(label="📊  Live Web Dashboard", command=open_dashboard)
view_menu.add_separator()
view_menu.add_command(label="Toggle Network Dashboard Panel", command=toggle_net_dashboard)
menubar.add_cascade(label="View", menu=view_menu)

# 4. Help Menu
help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="About UWSN Simulation", command=show_about_dialog)
menubar.add_cascade(label="Help", menu=help_menu)

root.config(menu=menubar)

# Keyboard shortcuts
root.bind("<space>", lambda e: toggle_pause())
root.bind("<Control-s>", lambda e: set_pause(False))
root.bind("<Control-r>", lambda e: reset_simulation())

# ── Top Toolbar ──────────────────────────────────────────────────────────────
top_bar = tk.Frame(root, bg='#1A2B4C', height=40)
top_bar.pack(side=tk.TOP, fill=tk.X)

tk.Label(top_bar,
         text=" ⚓ Harbor Traffic Control System ",
         font=('Segoe UI', 11, 'bold'),
         bg='#1A2B4C', fg='#FFFFFF').pack(side=tk.LEFT, padx=(8, 4), pady=6)

# Simulation controls
btn_pause = tk.Button(top_bar, text="⏸ Pause", bg='#2B4C6F', fg='white',
                      activebackground='#3A608F', activeforeground='white',
                      font=('Segoe UI', 8, 'bold'), bd=0, padx=8, pady=3, cursor='hand2',
                      command=toggle_pause)
btn_pause.pack(side=tk.LEFT, padx=3)

btn_reset = tk.Button(top_bar, text="↺ Reset", bg='#2B4C6F', fg='white',
                      activebackground='#3A608F', activeforeground='white',
                      font=('Segoe UI', 8, 'bold'), bd=0, padx=8, pady=3, cursor='hand2',
                      command=reset_simulation)
btn_reset.pack(side=tk.LEFT, padx=3)

# Season indicator badge & toggle
season_var = tk.StringVar(value=f"{sim.get_active_scenario()['badge']}")
tk.Label(top_bar, textvariable=season_var,
         font=('Segoe UI', 8, 'bold'), bg='#0F2847', fg='#FFD166',
         padx=8, pady=2, relief='groove', bd=1).pack(side=tk.LEFT, padx=6)

tk.Button(top_bar, text="⇄ Switch Season", bg='#2B4C6F', fg='white',
          activebackground='#3A608F', activeforeground='white',
          font=('Segoe UI', 8, 'bold'), bd=0, padx=8, pady=3, cursor='hand2',
          command=toggle_scenario).pack(side=tk.LEFT, padx=3)

tk.Button(top_bar, text="🌪️ Gale Squall", bg='#D62828', fg='white',
          activebackground='#E63946', activeforeground='white',
          font=('Segoe UI', 8, 'bold'), bd=0, padx=8, pady=3, cursor='hand2',
          command=trigger_squall_action).pack(side=tk.LEFT, padx=3)

tk.Button(top_bar, text="⚓ Re-Anchor Sensors", bg='#0077B6', fg='white',
          activebackground='#0096C7', activeforeground='white',
          font=('Segoe UI', 8, 'bold'), bd=0, padx=8, pady=3, cursor='hand2',
          command=reanchor_action).pack(side=tk.LEFT, padx=3)

# Right-aligned buttons and counters
tk.Button(top_bar, text="📊 Web Dashboard", bg='#0077B6', fg='white',
          activebackground='#0096C7', activeforeground='white',
          font=('Segoe UI', 8, 'bold'), bd=0, padx=8, pady=3, cursor='hand2',
          command=open_dashboard).pack(side=tk.RIGHT, padx=8)

tk.Button(top_bar, text="⟲ Reset View", bg='#3A4C6C', fg='white',
          font=('Segoe UI', 8, 'bold'), bd=0, padx=8, pady=3, cursor='hand2',
          command=reset_view).pack(side=tk.RIGHT, padx=4)

frame_var      = tk.StringVar(value="Frame: 0000")
sensor_var     = tk.StringVar(value="Active Sensors: 0")
violation_var  = tk.StringVar(value="Violations: 0")

tk.Label(top_bar, textvariable=violation_var,
         font=('Segoe UI', 8, 'bold'), bg='#1A2B4C', fg='#FF6B6B').pack(side=tk.RIGHT, padx=6)
tk.Label(top_bar, textvariable=sensor_var,
         font=('Segoe UI', 8), bg='#1A2B4C', fg='#90CAF9').pack(side=tk.RIGHT, padx=6)
tk.Label(top_bar, textvariable=frame_var,
         font=('Segoe UI', 8), bg='#1A2B4C', fg='#90CAF9').pack(side=tk.RIGHT, padx=4)

# ── Dynamic Monsoon Environment & Water Current HUD Bar ─────────────────────
hud_bar = tk.Frame(root, bg='#0B1B30', height=28)
hud_bar.pack(side=tk.TOP, fill=tk.X)

water_level_var  = tk.StringVar(value="🌊 Water Level: Normal (0.0 m) | Wave H: 0.3 m")
current_info_var = tk.StringVar(value="💨 Current: 0.35 m/s [Nominal Tide]")
drift_status_var = tk.StringVar(value="⚓ Sensors: 35/35 Moored Securely")
net_health_var   = tk.StringVar(value="📡 Network: STABLE (Health: 100%)")

tk.Label(hud_bar, textvariable=water_level_var,
         font=('Segoe UI', 8, 'bold'), bg='#0B1B30', fg='#48CAE4').pack(side=tk.LEFT, padx=10, pady=3)
tk.Label(hud_bar, textvariable=current_info_var,
         font=('Segoe UI', 8, 'bold'), bg='#0B1B30', fg='#FFD166').pack(side=tk.LEFT, padx=14, pady=3)
tk.Label(hud_bar, textvariable=drift_status_var,
         font=('Segoe UI', 8, 'bold'), bg='#0B1B30', fg='#FFAA00').pack(side=tk.LEFT, padx=14, pady=3)
tk.Label(hud_bar, textvariable=net_health_var,
         font=('Segoe UI', 8, 'bold'), bg='#0B1B30', fg='#52B788').pack(side=tk.RIGHT, padx=12, pady=3)

# ── Bottom status bar ─────────────────────────────────────────────────────
bot_bar = tk.Frame(root, bg='#E2EAF4', height=24)
bot_bar.pack(side=tk.BOTTOM, fill=tk.X)

status_var = tk.StringVar(value="Simulation running…")
tk.Label(bot_bar, textvariable=status_var,
         font=('Segoe UI', 8), bg='#E2EAF4', fg='#4A6FA5').pack(side=tk.LEFT, padx=10)
tk.Label(bot_bar, text="LOC @ X=5000 m  |  Outer Yard (X>5000)  |  Inner Patrol Zone (X<5000)",
         font=('Segoe UI', 8), bg='#E2EAF4', fg='#4A6FA5').pack(side=tk.RIGHT, padx=10)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Simulation Container: 3D Viewport (Left) + Network Dashboard (Right)
# ─────────────────────────────────────────────────────────────────────────────
main_container = tk.Frame(root, bg=C_WIN)
main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

canvas_frame = tk.Frame(main_container, bg=C_PANEL)
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fig = plt.figure(figsize=(16, 10), facecolor=C_PANEL)
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

canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Dock dedicated desktop Network Dashboard on the right
net_dashboard = NetworkDashboard(main_container)
net_dashboard.pack(side=tk.RIGHT, fill=tk.Y)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Seabed & ocean surface
# ─────────────────────────────────────────────────────────────────────────────
Xg = np.linspace(0, MAP, 35)
Yg = np.linspace(0, MAP, 35)
Xg, Yg = np.meshgrid(Xg, Yg)
Zg = sim.get_seabed_depth(Xg, Yg)

ax.plot_surface(Xg, Yg, Zg, cmap='GnBu', alpha=0.65, edgecolor='none')

# Dynamic water surface mesh initialized
X_surf_pts = np.linspace(0, MAP, 18)
Y_surf_pts = np.linspace(0, MAP, 18)
X_surf, Y_surf = np.meshgrid(X_surf_pts, Y_surf_pts)
Z_surf_init = sim.get_water_surface_height(X_surf, Y_surf, 0.0)
ocean_surf = ax.plot_surface(X_surf, Y_surf, Z_surf_init,
                             color=C_OCEAN_TOP, alpha=0.20, edgecolor='#90E0EF', linewidth=0.15)

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
# 9. Sensor nodes & Mooring Tethers
# ─────────────────────────────────────────────────────────────────────────────
SENSING_RANGE = sim.SENSING_RANGE
sensor_glow   = []
sensor_core   = []
tether_lines  = []

for s in sensors:
    sx, sy, sz = s.pos
    ax_x, ax_y, bed_z = s.anchor
    sg = ax.scatter([sx], [sy], [sz], color=C_SENSOR_IDLE, marker='o', s=320,
                    edgecolors='#90E0EF', linewidths=1.6, alpha=0.30, zorder=4)
    sc = ax.scatter([sx], [sy], [sz], color=C_SENSOR_IDLE, marker='o', s=110,
                    edgecolors='#023E8A', linewidths=1.0, zorder=5)
    tline, = ax.plot([ax_x, sx], [ax_y, sy], [bed_z, sz],
                     color='#ADE8F4', linestyle=':', linewidth=0.9, alpha=0.55)
    sensor_glow.append(sg)
    sensor_core.append(sc)
    tether_lines.append(tline)

# ─────────────────────────────────────────────────────────────────────────────
# 9e. Sensor Drift Trails (ghost breadcrumb lines for drifting sensors)
# ─────────────────────────────────────────────────────────────────────────────
# Each sensor gets up to 12 trail segments (11 line objects in the pool)
MAX_TRAIL = 11
drift_trail_lines = []
for _ in sensors:
    seg_pool = [ax.plot([], [], [], color='#FF6B6B', linewidth=0.7,
                         alpha=0.0, linestyle='-')[0]
                for _ in range(MAX_TRAIL)]
    drift_trail_lines.append(seg_pool)

# ─────────────────────────────────────────────────────────────────────────────
# 9f. Turbulent Vortex Eye Markers
# ─────────────────────────────────────────────────────────────────────────────
vortex_markers = ax.scatter([], [], [], color='#FF1744', marker='x',
                             s=120, linewidths=2.5, alpha=0.0, zorder=8)
vortex_rings = [ax.plot([], [], [], color='#FF6B35', linewidth=1.2,
                         linestyle='--', alpha=0.0)[0]
                for _ in range(3)]

# ─────────────────────────────────────────────────────────────────────────────
# 9g. Rain Particle Effect (3D scatter of raindrops in viewport)
# ─────────────────────────────────────────────────────────────────────────────
_N_RAIN = 180
_rain_x = np.random.uniform(0, MAP, _N_RAIN)
_rain_y = np.random.uniform(0, MAP, _N_RAIN)
_rain_z = np.random.uniform(2, 65, _N_RAIN)
rain_scatter = ax.scatter(_rain_x, _rain_y, _rain_z,
                           color='#90E0EF', marker='|', s=14, alpha=0.0, zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# 9h. Lightning Flash Overlay (Tkinter canvas label, shown on squall)
# ─────────────────────────────────────────────────────────────────────────────
lightning_label = tk.Label(canvas_frame, text="⚡ VIOLENT SQUALL SURGE ⚡",
                            font=('Segoe UI', 16, 'bold'), bg='#1A0020',
                            fg='#FFD166', bd=3, relief='ridge',
                            padx=14, pady=8)
# Hidden initially — placed over canvas when squall fires
_lightning_visible = False
_lightning_frame_count = 0

# ─────────────────────────────────────────────────────────────────────────────
# 9i. Network-Broken Alert Banner (Tkinter label overlay)
# ─────────────────────────────────────────────────────────────────────────────
net_broken_banner = tk.Label(canvas_frame,
    text="🚨  SENSOR NETWORK COLLAPSED  🚨\nHarsh currents drove sensors away — communication links severed!",
    font=('Segoe UI', 12, 'bold'), bg='#7D0000', fg='#FFD166',
    bd=3, relief='ridge', padx=16, pady=10, justify='center')
_net_banner_visible = False

# ─────────────────────────────────────────────────────────────────────────────
# 9a. Buoy Gateways Visuals
# ─────────────────────────────────────────────────────────────────────────────
import network
buoy_plots = []
for name, (bx, by, bz) in network.BUOYS.items():
    bp = ax.scatter([bx], [by], [bz], color='#FFD700', marker='^', s=180,
                    edgecolors='#D4AF37', linewidths=2.0, zorder=7)
    ax.text(bx, by, bz + 4.0, name.replace("buoy_", "Buoy ").upper(),
            fontsize=7, fontweight='bold', color='#B8860B', ha='center')
    buoy_plots.append(bp)

# ─────────────────────────────────────────────────────────────────────────────
# 9b. Mobile Robot Nodes (AUVs) Visuals
# ─────────────────────────────────────────────────────────────────────────────
auv_states = auv.get_auv_states()
auv_cores = []
auv_rings = []
auv_labels = []

theta_pts = np.linspace(0, 2 * np.pi, 25)
for a in auv_states:
    ax_pos = a["pos"]
    # Core diamond (orange)
    ac = ax.scatter([ax_pos[0]], [ax_pos[1]], [ax_pos[2]], color='#FF7043', marker='D', s=130,
                    edgecolors='#FFD700', linewidths=1.2, zorder=6)
    auv_cores.append(ac)
    
    # Ring in XY plane at depth z representing sensing coverage
    rx = ax_pos[0] + SENSING_RANGE * np.cos(theta_pts)
    ry = ax_pos[1] + SENSING_RANGE * np.sin(theta_pts)
    rz = np.full_like(theta_pts, ax_pos[2])
    ring_line, = ax.plot(rx, ry, rz, color='#FF7043', linestyle='--', linewidth=0.9, alpha=0.45)
    auv_rings.append(ring_line)
    
    # Label text
    lbl = ax.text(ax_pos[0], ax_pos[1], ax_pos[2] + 4.0, f"AUV-{a['id']}", 
                  fontsize=6.5, fontweight='bold', color='#E65100', ha='center')
    auv_labels.append(lbl)

# ─────────────────────────────────────────────────────────────────────────────
# 9c. P2P Communication Link Visual Pool
# ─────────────────────────────────────────────────────────────────────────────
comm_lines = [ax.plot([], [], [], linestyle=':', color='#5A7A8A', linewidth=0.8, alpha=0.0)[0]
              for _ in range(250)]

# ─────────────────────────────────────────────────────────────────────────────
# 9d. Dynamic Water Current Flow Indicators (3D Quivers)
# ─────────────────────────────────────────────────────────────────────────────
current_grid_coords = [
    (1800, 2500), (5000, 2500), (8200, 2500),
    (1800, 5000), (5000, 5000), (8200, 5000),
    (1800, 7500), (5000, 7500), (8200, 7500),
]
current_quivers = []
for cx, cy in current_grid_coords:
    q_line, = ax.plot([], [], [], color='#00B4D8', linewidth=2.2, alpha=0.75)
    q_tip   = ax.scatter([], [], [], color='#00B4D8', marker='^', s=55, alpha=0.85)
    current_quivers.append((cx, cy, q_line, q_tip))

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
    Line2D([0],[0], marker='o', color='none', markerfacecolor='#FF1744',
           markersize=9,  label='Sensor — Snapped/Drifting', linestyle='None', markeredgecolor='#FFD700'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor='#FFB703',
           markersize=9,  label='Sensor — Mooring Strained', linestyle='None', markeredgecolor='#FB8500'),
    Line2D([0],[0], color='#00B4D8', linewidth=2.0, label='Water Current Flow Vector'),
    Line2D([0],[0], marker='D', color='none', markerfacecolor='#FF7043',
           markersize=9,  label='Mobile AUV Node',       linestyle='None', markeredgecolor='#FFD700'),
    Line2D([0],[0], color='#00E5FF', linewidth=1.5, linestyle='--', label='Active Multi-Hop Link'),
    Line2D([0],[0], color='#5A7A8A', linewidth=0.8, linestyle=':', label='Topology Link'),
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
    global fleet, ocean_surf, _lightning_visible, _lightning_frame_count, _net_banner_visible
    global _rain_x, _rain_y, _rain_z

    if is_paused:
        quiver_artists = [q[2] for q in current_quivers] + [q[3] for q in current_quivers]
        trail_artists = [seg for pool in drift_trail_lines for seg in pool]
        vortex_ring_artists = vortex_rings
        return (v_plots + v_texts + sensor_glow + sensor_core + tether_lines +
                auv_cores + auv_rings + auv_labels + comm_lines + quiver_artists +
                trail_artists + [vortex_markers] + vortex_ring_artists + [rain_scatter])

    frame_var.set(f"Frame: {frame:04d}")
    dt = 1.0 / 60.0
    t_sec = frame * dt

    # 1. Step ocean environment & non-deterministic stochastic current intensity
    ocean_env = sim.step_ocean_environment(frame, dt)
    sim.step_sensors(frame, dt)

    # 2. Dynamic 3D ocean surface update (every 2 frames for smooth animation)
    if frame % 2 == 0:
        try:
            ocean_surf.remove()
        except Exception:
            pass
        Z_surf = sim.get_water_surface_height(X_surf, Y_surf, t_sec)
        sc_key = sim.ACTIVE_SCENARIO_KEY
        if sc_key == 'MONSOON':
            surf_color = '#1B3A4B'
            edge_color = '#E0FAFF' if (frame % 6 < 3) else '#468FAF'
            surf_alpha = 0.38
            surf_lw    = 0.30
        else:
            surf_color = C_OCEAN_TOP
            edge_color = '#90E0EF'
            surf_alpha = 0.18
            surf_lw    = 0.15
        ocean_surf = ax.plot_surface(X_surf, Y_surf, Z_surf,
                                     color=surf_color, alpha=surf_alpha,
                                     edgecolor=edge_color, linewidth=surf_lw)

    # 3. Dynamic Water Current Flow Indicators (quivers)
    sc_key = sim.ACTIVE_SCENARIO_KEY
    is_monsoon = (sc_key == 'MONSOON')
    squall_active = ocean_env.get('squall_timer', 0) > 0
    quiver_color = '#FF1744' if squall_active else ('#FF9E00' if is_monsoon else '#00B4D8')
    arrow_mult = 300.0 if not is_monsoon else 150.0

    for cx, cy, q_line, q_tip in current_quivers:
        cvx, cvy, cvz = sim.get_current_vector(cx, cy, -4.0, t_sec)
        tx = cx + cvx * arrow_mult
        ty = cy + cvy * arrow_mult
        tz = -4.0 + cvz * 18.0
        q_line.set_data([cx, tx], [cy, ty])
        q_line.set_3d_properties([-4.0, tz])
        q_line.set_color(quiver_color)
        q_tip._offsets3d = ([tx], [ty], [tz])
        q_tip.set_color(quiver_color)

    # 3b. Turbulent Vortex Eye Markers — visible during Monsoon
    if is_monsoon:
        vx_c = [v[0] for v in ocean_env['vortices']]
        vy_c = [v[1] for v in ocean_env['vortices']]
        vz_c = [0.0, 0.0, 0.0]
        vortex_markers._offsets3d = (vx_c, vy_c, vz_c)
        vortex_markers.set_alpha(0.70)
        theta_v = np.linspace(0, 2 * np.pi, 40)
        vortex_radius = 800.0
        for vi, (vxc, vyc, vspin) in enumerate(ocean_env['vortices']):
            rx_v = vxc + vortex_radius * np.cos(theta_v)
            ry_v = vyc + vortex_radius * np.sin(theta_v)
            rz_v = np.zeros(40)
            vortex_rings[vi].set_data(rx_v, ry_v)
            vortex_rings[vi].set_3d_properties(rz_v)
            spin_color = '#FF6B35' if vspin > 0 else '#9B59B6'
            vortex_rings[vi].set_color(spin_color)
            vortex_rings[vi].set_alpha(0.50 + 0.15 * np.sin(frame * 0.12))
    else:
        vortex_markers.set_alpha(0.0)
        for vr in vortex_rings:
            vr.set_alpha(0.0)

    # 3c. Rain Particle Effect — animated rainfall streaks during Monsoon
    if is_monsoon:
        wave_h = ocean_env.get('wave_height', 0.5)
        rain_intensity = min(1.0, wave_h / 8.5)
        # Animate rain: shift particles downward, wrap at bottom
        _rain_z -= (2.5 + rain_intensity * 4.5)  # fall speed
        wrap_mask = _rain_z < 0
        _rain_z[wrap_mask] = np.random.uniform(55, 70, wrap_mask.sum())
        _rain_x[wrap_mask] = np.random.uniform(0, MAP, wrap_mask.sum())
        _rain_y[wrap_mask] = np.random.uniform(0, MAP, wrap_mask.sum())
        # Horizontal drift from storm current
        _rain_x -= 12.0 * rain_intensity
        _rain_y += 6.0 * rain_intensity
        _rain_x = np.clip(_rain_x, 0, MAP)
        _rain_y = np.clip(_rain_y, 0, MAP)
        rain_scatter._offsets3d = (_rain_x, _rain_y, _rain_z)
        rain_color = '#A8DADC' if not squall_active else '#E0FBFC'
        rain_scatter.set_color(rain_color)
        rain_scatter.set_alpha(0.35 + 0.25 * rain_intensity)
        rain_scatter.set_sizes(np.full(_N_RAIN, 10.0 + 8.0 * rain_intensity))
    else:
        rain_scatter.set_alpha(0.0)

    # 3d. Lightning Flash Effect on Squall Events
    if squall_active and is_monsoon:
        _lightning_frame_count += 1
        flash_on = (_lightning_frame_count % 8) < 4  # flicker every 4 frames
        if flash_on:
            if not _lightning_visible:
                lightning_label.place(relx=0.5, rely=0.08, anchor='center')
                _lightning_visible = True
        else:
            if _lightning_visible:
                lightning_label.place_forget()
                _lightning_visible = False
        # Optionally change facecolor to flicker
        if squall_active:
            fig.patch.set_facecolor('#0D1117' if flash_on else C_PANEL)
        else:
            fig.patch.set_facecolor(C_PANEL)
    else:
        _lightning_frame_count = 0
        if _lightning_visible:
            lightning_label.place_forget()
            _lightning_visible = False
        fig.patch.set_facecolor(C_PANEL)

    # ── DYNAMIC UWSN PIPELINE ──────────────────────────────────────────────
    sensor_active, total_detections = sim.collect_sensor_detections(
        fleet, sensors, frame
    )
    ml_status, ml_is_live = ml.update_model_from_collected_data()

    # Traffic manager now returns (fleet, violations)
    result = tms.manage_traffic_from_csv(fleet)
    fleet, violations = result if isinstance(result, tuple) else (result, [])

    # Step AUVs physics & state machine
    auv_states = auv.step_auvs(frame, dt=dt, active_violations=violations, fleet=fleet)

    # Build communication graph and route telemetry packets
    graph = network.build_network_graph(sensors, auv_states)
    delivered_paths, active_packets, network_stats = network.route_detections(fleet, sensors, auv_states, graph, frame)
    # ──────────────────────────────────────────────────────────────────────

    # Update dedicated Desktop Network Dashboard
    net_dashboard.update_dashboard(network_stats)

    # 3e. Network Collapsed Alert — ONLY shown when network actually reaches COLLAPSED state
    net_state = network_stats.get('network_state', 'STABLE')
    show_net_banner = (net_state == 'COLLAPSED')
    if show_net_banner:
        banner_fg = '#FFD166' if (frame % 10) < 5 else '#FF6B6B'
        net_broken_banner.config(fg=banner_fg)
        if not _net_banner_visible:
            net_broken_banner.place(relx=0.5, rely=0.5, anchor='center')
            _net_banner_visible = True
    else:
        if _net_banner_visible:
            net_broken_banner.place_forget()
            _net_banner_visible = False

    # Sensor blink visuals + per-vessel hit counting + mooring tether dynamics
    ship_xyz           = np.array([[v['pos'][0], v['pos'][1], v['depth']] for v in fleet])
    active_count       = 0
    drifting_count     = 0
    strained_count     = 0
    vessel_sensor_hits = np.zeros(len(fleet), dtype=int)

    for i, s in enumerate(sensors):
        sx, sy, sz = s.pos
        anchor = s.anchor
        status = s.status

        # Dynamic 3D position update as sensors drift away
        sensor_glow[i]._offsets3d = ([sx], [sy], [sz])
        sensor_core[i]._offsets3d = ([sx], [sy], [sz])

        # Mooring Tether lines
        if status == "MOORED":
            tether_lines[i].set_data([anchor[0], sx], [anchor[1], sy])
            tether_lines[i].set_3d_properties([anchor[2], sz])
            tether_lines[i].set_color('#ADE8F4')
            tether_lines[i].set_linestyle(':')
            tether_lines[i].set_linewidth(0.9)
            tether_lines[i].set_alpha(0.55)
        elif status == "STRAINED":
            strained_count += 1
            tether_lines[i].set_data([anchor[0], sx], [anchor[1], sy])
            tether_lines[i].set_3d_properties([anchor[2], sz])
            tether_lines[i].set_color('#FFAA00')
            tether_lines[i].set_linestyle('-')
            tether_lines[i].set_linewidth(1.8)
            tether_lines[i].set_alpha(0.85)
        else: # SNAPPED / DRIFTING
            drifting_count += 1
            tether_lines[i].set_data([anchor[0], anchor[0] + 50], [anchor[1], anchor[1] + 50])
            tether_lines[i].set_3d_properties([anchor[2], anchor[2] + 8])
            tether_lines[i].set_color('#FF3333')
            tether_lines[i].set_linestyle('--')
            tether_lines[i].set_linewidth(1.2)
            tether_lines[i].set_alpha(0.35)

        # Update drift trail for SNAPPED sensors
        trail = sensors[i].trail
        seg_pool = drift_trail_lines[i]
        if status == 'SNAPPED' and len(trail) >= 2:
            # Color fades from bright red at head to dark for older points
            for ti in range(MAX_TRAIL):
                if ti < len(trail) - 1:
                    p0 = trail[-(ti + 2)]  # older point
                    p1 = trail[-(ti + 1)]  # newer point
                    seg_pool[ti].set_data([p0[0], p1[0]], [p0[1], p1[1]])
                    seg_pool[ti].set_3d_properties([p0[2], p1[2]])
                    fade = max(0.08, 0.75 - ti * 0.065)
                    seg_pool[ti].set_alpha(fade)
                    seg_pool[ti].set_color('#FF6B6B' if ti < 3 else '#FF9E00')
                    seg_pool[ti].set_linewidth(max(0.5, 1.5 - ti * 0.1))
                else:
                    seg_pool[ti].set_alpha(0.0)
        else:
            for seg in seg_pool:
                seg.set_alpha(0.0)

        dists = np.linalg.norm(ship_xyz - np.array([sx, sy, sz]), axis=1)
        hits  = dists < SENSING_RANGE
        vessel_sensor_hits[hits] += 1

        active = sensor_active[i]
        if status == "SNAPPED":
            blink = (frame % 4) < 2
            fg = '#FF1744' if blink else '#9D0208'
            edge = '#FFD700' if blink else '#D00000'
            glow_alpha = 0.95
        elif status == "STRAINED":
            fg = '#FF9E00'
            edge = '#FB8500'
            glow_alpha = 0.80
        elif active:
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

    # Update HUD status telemetry
    curr_info = sim.get_monsoon_current_info()
    if is_monsoon:
        surge_txt = f"+{curr_info['surge_level']:.1f} m" if curr_info['surge_level'] >= 0 else f"{curr_info['surge_level']:.1f} m"
        water_level_var.set(f"🌊 Water Level: {surge_txt} [Storm Surge] | Peak Wave: {curr_info['wave_height']:.1f} m")
        squall_tag = " [⚡ GALE SQUALL SURGE!]" if curr_info['squall_active'] else ""
        current_info_var.set(f"💨 Current: {curr_info['intensity_mps']:.2f} m/s{squall_tag}")
    else:
        water_level_var.set("🌊 Water Level: Normal (0.0 m) | Wave H: 0.3 m [Calm]")
        current_info_var.set(f"💨 Current: {curr_info['intensity_mps']:.2f} m/s [Nominal Tide]")

    if drifting_count > 0:
        drift_status_var.set(f"⚠️ Swept Away: {drifting_count}/{len(sensors)} Drifting ({strained_count} strained)")
    else:
        drift_status_var.set(f"⚓ Sensors: {len(sensors)}/{len(sensors)} Moored Securely")

    pdr_val = network_stats.get('pdr', 100.0)
    health_score = network_stats.get('health_score', 100.0)
    if net_state == 'COLLAPSED':
        net_health_var.set(f"🚨 NETWORK COLLAPSED (PDR: {pdr_val:.1f}% | Links Severed)")
    elif net_state == 'CRITICAL':
        net_health_var.set(f"⚠️ Network: CRITICAL (PDR: {pdr_val:.1f}% | Health: {health_score:.0f}%)")
    elif net_state == 'DEGRADED':
        net_health_var.set(f"⚡ Network: DEGRADED (PDR: {pdr_val:.1f}% | Health: {health_score:.0f}%)")
    else:
        net_health_var.set(f"📡 Network: STABLE (PDR: {pdr_val:.1f}% | Health: {health_score:.0f}%)")

    ml_prefix = "ML LIVE" if ml_is_live else "Collecting"
    status_var.set(
        f"Frame {frame:04d}  |  {ml_prefix}: {total_detections} detections  |  "
        f"State: {net_state} ({health_score:.0f}%)  |  PDR: {pdr_val:.1f}%"
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

    # Update AUV visuals
    for idx, a in enumerate(auv_states):
        ax_pos = a["pos"]
        auv_cores[idx]._offsets3d = ([ax_pos[0]], [ax_pos[1]], [ax_pos[2]])
        
        if a["status"] == "DISPATCHED":
            auv_cores[idx].set_facecolor('#FF3D3D')  # Bright red when responding
            auv_cores[idx].set_edgecolor('#FFD700')
        elif a["status"] == "CHARGING":
            auv_cores[idx].set_facecolor('#00E5FF')  # Cyan/blue when charging
            auv_cores[idx].set_edgecolor('#FFFFFF')
        else:
            auv_cores[idx].set_facecolor('#FF7043')  # Orange when patrolling
            auv_cores[idx].set_edgecolor('#FFD700')
            
        rx = ax_pos[0] + SENSING_RANGE * np.cos(theta_pts)
        ry = ax_pos[1] + SENSING_RANGE * np.sin(theta_pts)
        rz = np.full_like(theta_pts, ax_pos[2])
        auv_rings[idx].set_data(rx, ry)
        auv_rings[idx].set_3d_properties(rz)
        
        auv_labels[idx].set_position((ax_pos[0], ax_pos[1]))
        auv_labels[idx].set_3d_properties(ax_pos[2] + 4.0)
        status_abbrev = {"PATROLLING": "PATR", "DISPATCHED": "DISP", "CHARGING": "CHRG"}.get(a["status"], "AUV")
        auv_labels[idx].set_text(f"AUV-{a['id']} [{status_abbrev}] ({int(a['battery_pct'])}%)")

    # Update active and topology communication lines
    pos_map = {}
    for name, pos in network.BUOYS.items():
        pos_map[name] = np.array(pos)
    for i, spos in enumerate(sensors):
        pos_map[f"sensor_{i}"] = np.array(spos)
    for a in auv_states:
        pos_map[f"auv_{a['id']}"] = np.array(a["pos"])
        
    line_idx = 0
    drawn_active_edges = set()
    
    # 1. Draw active routing paths
    for paths in delivered_paths.values():
        for path in paths:
            for hop_idx in range(len(path) - 1):
                u, w = path[hop_idx], path[hop_idx + 1]
                edge_key = tuple(sorted([u, w]))
                if edge_key not in drawn_active_edges:
                    drawn_active_edges.add(edge_key)
                    if line_idx < len(comm_lines):
                        p_u = pos_map[u]
                        p_v = pos_map[w]
                        comm_lines[line_idx].set_data([p_u[0], p_v[0]], [p_u[1], p_v[1]])
                        comm_lines[line_idx].set_3d_properties([p_u[2], p_v[2]])
                        
                        snr = network._link_snr(np.linalg.norm(p_u - p_v))
                        color = '#00FF99' if snr >= 20.0 else ('#00E5FF' if snr >= 12.0 else '#FF3D3D')
                        
                        comm_lines[line_idx].set_color(color)
                        comm_lines[line_idx].set_linewidth(2.0)
                        comm_lines[line_idx].set_alpha(0.85)
                        comm_lines[line_idx].set_linestyle('--')
                        line_idx += 1
                        
    # 2. Draw normal topology edges
    for u, w in graph.edges:
        edge_key = tuple(sorted([u, w]))
        if edge_key not in drawn_active_edges:
            if line_idx < len(comm_lines):
                p_u = pos_map[u]
                p_v = pos_map[w]
                comm_lines[line_idx].set_data([p_u[0], p_v[0]], [p_u[1], p_v[1]])
                comm_lines[line_idx].set_3d_properties([p_u[2], p_v[2]])
                
                comm_lines[line_idx].set_color('#5A7A8A')
                comm_lines[line_idx].set_linewidth(0.8)
                comm_lines[line_idx].set_alpha(0.20)
                comm_lines[line_idx].set_linestyle(':')
                line_idx += 1
                
    # 3. Disable remaining visual pool lines
    for k in range(line_idx, len(comm_lines)):
        comm_lines[k].set_data([], [])
        comm_lines[k].set_3d_properties([])
        comm_lines[k].set_alpha(0.0)

    global LAST_DASHBOARD_WRITE_TIME
    current_time = time.time()
    
    if current_time - LAST_DASHBOARD_WRITE_TIME >= 0.5:
        # Build telemetry data for dashboard
        dashboard_robots = []
        for a in auv_states:
            dashboard_robots.append({
                "id":          a["id"],
                "x":           round(float(a["pos"][0]), 1),
                "y":           round(float(a["pos"][1]), 1),
                "z":           round(float(a["pos"][2]), 1),
                "vx":          round(float(a["vel"][0]), 2),
                "vy":          round(float(a["vel"][1]), 2),
                "vz":          round(float(a["vel"][2]), 2),
                "battery_pct": round(float(a["battery_pct"]), 1),
                "status":      a["status"],
                "waypoint":    [round(float(w), 1) for w in a["waypoint"]],
                "current_zone": a.get("patrol_zone", "Outer Yard")
            })

        dashboard_links = []
        for u, w, data in graph.edges(data=True):
            p_u = pos_map[u]
            p_w = pos_map[w]
            d_edge = np.linalg.norm(p_u - p_w)
            delay = d_edge / 1500.0
            
            is_active_hop = False
            for paths in delivered_paths.values():
                for path in paths:
                    for hop_idx in range(len(path) - 1):
                        if (path[hop_idx] == u and path[hop_idx+1] == w) or (path[hop_idx] == w and path[hop_idx+1] == u):
                            is_active_hop = True
                            break
                    if is_active_hop:
                        break
                if is_active_hop:
                    break
            
            payload_type = "TELEMETRY" if is_active_hop else "BEACON"
            dashboard_links.append({
                "source_id": u,
                "target_id": w,
                "snr":       round(float(data.get("snr", 0.0)), 1),
                "delay":     round(delay, 4),
                "payload_type": payload_type
            })

        dashboard_routed = {
            "delivered": list(delivered_paths.keys()),
            "paths": {str(k): v for k, v in delivered_paths.items()}
        }

        active_scen = sim.get_active_scenario()
        try:
            data_dict = {
                "frame":             frame,
                "active_sensors":    active_count,
                "total_sensors":     len(sensors),
                "ml_status":         ml_status,
                "violations":        violations,
                "vessels":           dashboard_vessels,
                "robots":            dashboard_robots,
                "robot_nodes":       dashboard_robots,
                "network_links":     dashboard_links,
                "active_comm_links": dashboard_links,
                "routed_detections": dashboard_routed,
                "environment": {
                    "scenario":          active_scen["name"],
                    "scenario_key":      sim.ACTIVE_SCENARIO_KEY,
                    "badge":             active_scen["badge"],
                    "description":       active_scen["description"],
                    "sea_state":         f"Douglas {active_scen['sea_state'][0]}–{active_scen['sea_state'][1]}",
                    "wind_speed_knots":  f"{active_scen['wind_speed_knots'][0]}–{active_scen['wind_speed_knots'][1]} kn",
                    "ambient_noise_db":  round(network.get_current_ambient_noise_db(), 1),
                    "current_mult":      active_scen.get("current_mult", 1.0)
                },
                "network_stats": {
                    "scenario":          active_scen["name"],
                    "scenario_key":      sim.ACTIVE_SCENARIO_KEY,
                    "ambient_noise_db":  round(network.get_current_ambient_noise_db(), 1),
                    "pdr":               network_stats["pdr"],
                    "avg_hop_count":     network_stats["avg_hop_count"],
                    "active_alerts":     network_stats["active_alerts"],
                    "total_energy_saved_j": network_stats["total_energy_saved_j"],
                    "avg_latency_default_s": network_stats["avg_latency_default_s"],
                    "avg_latency_compressed_s": network_stats["avg_latency_compressed_s"],
                    "node_states":       network_stats["node_states"]
                }
            }
            with open(DASHBOARD_JSON, 'w') as _f:
                json.dump(data_dict, _f)
            with open(DASHBOARD_JS, 'w') as _f:
                _f.write(f"window.DASHBOARD_DATA = {json.dumps(data_dict)};")
            LAST_DASHBOARD_WRITE_TIME = current_time
        except Exception:
            pass

    quiver_artists = [q[2] for q in current_quivers] + [q[3] for q in current_quivers]
    trail_artists = [seg for pool in drift_trail_lines for seg in pool]
    vortex_ring_artists = vortex_rings
    return (v_plots + v_texts + sensor_glow + sensor_core + tether_lines +
            auv_cores + auv_rings + auv_labels + comm_lines + quiver_artists +
            trail_artists + [vortex_markers] + vortex_ring_artists + [rain_scatter])

# ─────────────────────────────────────────────────────────────────────────────
# 13. Run & Season Selection
# ─────────────────────────────────────────────────────────────────────────────
ani = None

def on_closing():
    global ani
    try:
        if ani and ani.event_source:
            ani.event_source.stop()
    except Exception:
        pass
    try:
        root.quit()
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass
    os._exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)

# Set initial default scenario: Normal Season
sim.set_active_scenario('NORMAL')
sc = sim.get_active_scenario()
season_var.set(f"{sc['badge']}")
status_var.set(f"Simulation active — {sc['name']} ({sc['badge']})")

def start_animation():
    global ani
    if ani is None:
        ani = FuncAnimation(fig, master_loop, interval=16, blit=False, cache_frame_data=False)
        canvas.draw_idle()

if __name__ == '__main__':
    # Launch FuncAnimation directly on app startup (Normal season default, no intro overlay)
    start_animation()
    # Open live web dashboard after initial boot telemetry settles
    root.after(1200, open_dashboard)
    root.mainloop()