# Implementation Plan — Harbor Traffic Control System Upgrades

## Phase 1: Repository Audit ✅ COMPLETE
- Audited all source files: main.py, simulator.py, mlmodel.py, network.py, network_manager.py, auv.py, traffic_algo.py
- Audited web dashboards: dashboard.html, networking_dashboard.html
- Audited documentation: README.md, PROJECT CONTEXT.md
- Understood architecture: Tkinter + Matplotlib 3D simulation, HTTP server, JSON telemetry, ML inference pipeline

---

## Phase 2: Fix Simulation Crash Notifications
**Goal**: Move error notifications out of the simulation viewport into a dedicated sidebar.

### Current State:
- Errors likely use `messagebox.showerror()` or overlay labels on canvas_frame
- Network broken banner (`net_broken_banner`) overlays the 3D viewport
- Lightning flash overlay (`lightning_label`) overlays the 3D viewport

### Implementation:
1. Create a `NotificationSidebar` class (tk.Frame) on the right side
2. Add notification types: INFO, WARNING, RECOVERABLE_ERROR, FATAL_ERROR
3. Thread-safe queue for background thread notifications
4. Periodic UI polling via `root.after()`
5. Expandable traceback for errors
6. Deduplication of repeated notifications
7. Clean shutdown handling

### Files to Modify:
- `main.py` - Add NotificationSidebar, integrate with main layout

---

## Phase 3: Redesign Tkinter Simulation Window (Grid Layout)
**Goal**: Professional maritime control station UI with 75%+ viewport.

### Current Layout Issues:
- Top toolbar + HUD bar + main_container (pack-based)
- Network dashboard fixed at 380px width on right
- No responsive resizing
- Inconsistent fonts/sizes

### New Layout (Grid-based):
```
┌─────────────────────────────────────────────────────────────────┐
│ Header: Title | Season | Status | Controls                      │
├───────────────────────────────────────┬─────────────────────────┤
│                                       │ System Status            │
│                                       ├─────────────────────────┤
│         LARGE 3D SIMULATION (75%+)    │ Notifications            │
│                                       ├─────────────────────────┤
│                                       │ Network / ML Metrics     │
├───────────────────────────────────────┴─────────────────────────┤
│ Footer: FPS | Active Nodes | Sim Time | Health                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation:
1. Restructure main window to use `grid()` with proper row/column weights
2. Create typography system (title, heading, body, telemetry, button, warning, metadata)
3. Move NetworkDashboard into right sidebar (collapsible)
4. Add NotificationSidebar (from Phase 2) into right sidebar
5. Responsive matplotlib canvas that fills available space
6. Professional color scheme, consistent spacing
7. Maximized window on startup

### Files to Modify:
- `main.py` - Complete UI restructuring

---

## Phase 4: Dark/Light Themes for Web Dashboards
**Goal**: Theme toggle with CSS custom properties, persistence, accessibility.

### Implementation:
1. Define CSS custom properties for both themes in `:root` and `[data-theme="light"]`
2. Add theme toggle button in header of both dashboards
3. Use `localStorage` for persistence
4. Update all colors, charts, network topology visualization
5. Consolas/JetBrains Mono for telemetry, sans-serif for headings
6. Semantic HTML, keyboard navigation, focus states, reduced motion

### Files to Modify:
- `dashboard.html`
- `networking_dashboard.html`

---

## Phase 5: Prediction Logging & Ground-Truth Validation
**Goal**: Systematic prediction logging with validation against simulation ground truth.

### Current ML Pipeline:
- `mlmodel.py`: VesselMLP predicts `Pred_Type` (6 classes) and `Pred_Weight_kg`
- `traffic_algo.py`: Reads predictions from `stage3_live_predictions.csv`
- Ground truth available in `simulator.py`: `Actual_Type`, `Actual_Weight_kg`, `Actual_Speed_knots`, `Actual_Depth_m`

### Implementation:
1. Create `prediction_logger.py` module with CSV storage
2. Log structure: prediction_id, run_id, timestamp, vessel_id, model_name, task, predicted, actual, error, validation_status, scenario, confidence
3. Hook into `mlmodel.py` inference to log predictions
4. Hook into `simulator.py`/`traffic_algo.py` to match ground truth
5. Validation engine: classification accuracy, MAE/RMSE for weight, circular error for direction
6. Aggregate metrics by run, model, scenario

### Files to Create:
- `prediction_logger.py` - Core logging & validation

### Files to Modify:
- `mlmodel.py` - Integrate prediction logging
- `traffic_algo.py` - Provide ground truth for validation

---

## Phase 6: Model Evaluation Pipeline
**Goal**: Reproducible evaluation against simulation ground truth.

### Implementation:
1. Evaluation baseline using simulation authoritative state
2. Audit existing VesselMLP (architecture, training, features, scaler, encoder)
3. Separate normal/monsoon evaluation
4. Document limitations (no speed/direction prediction models)
5. Prevent data leakage (time-separated observations)

### Files to Modify:
- `mlmodel.py` - Add evaluation functions
- `prediction_logger.py` - Add aggregation/export

---

## Phase 7: Rewrite README Documentation
**Goal**: Technically accurate, academically suitable documentation.

### Implementation:
- Follow suggested structure in Future_Updates.txt
- Document actual ML models, networking, scenarios, UI
- Include real evaluation results
- Mark unverified results

### Files to Modify:
- `README.md` - Complete rewrite

---

## Phase 8: Testing & Regression Prevention
**Goal**: Verify all functionality works.

### Test Categories:
- UI: resize, notifications, shutdown
- Web: theme toggle, persistence, live updates
- Prediction: classification match, regression errors, direction wraparound, persistence
- Integration: ML inference, networking, monsoon, traffic, AUV
- Error handling: recoverable/fatal, background threads, clean shutdown

---

## Phase 9: Performance, Code Quality, Architecture
**Goal**: Modular, efficient, maintainable code.

### Principles:
- Clear module responsibilities
- No duplicated logic
- Non-blocking main loop
- Thread-safe background processing
- Explicit units
- Efficient prediction logging

---

## Phase 10: Final Report
**Goal**: Engineering report with changes, validation, ML docs, test results, execution instructions, remaining issues, next steps.

---

## Execution Order:
1. Phase 2 → Phase 3 (UI foundation - tightly coupled)
2. Phase 4 (Web dashboards - independent)
3. Phase 5 → Phase 6 (ML logging & evaluation - coupled)
4. Phase 7 (Documentation)
5. Phase 8 (Testing)
6. Phase 9 (Quality)
7. Phase 10 (Report)