import sys
import os
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("================================================================================")
print("HARBOR TRAFFIC CONTROL SYSTEM -- ENVIRONMENT & SYSTEM VERIFICATION SUITE")
print("================================================================================")

# Step 1: Verify Python Dependencies
print("\n[Step 1/4] Checking Python version & required libraries...")
print(f"Python Version: {sys.version}")

required_modules = [
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "joblib",
    "torch"
]

missing_modules = []
for mod in required_modules:
    try:
        __import__(mod)
        print(f"  [OK] {mod:15s}: Available")
    except ImportError as e:
        print(f"  [FAIL] {mod:15s}: MISSING ({e})")
        missing_modules.append(mod)

if missing_modules:
    print(f"\n[ERROR] Missing required modules: {missing_modules}")
    sys.exit(1)
else:
    print("SUCCESS: All required Python packages are installed successfully!")

# Step 2: Verify Network Manager Classes
print("\n[Step 2/4] Testing modular network architecture (`network_manager.py`)...")
try:
    from network_manager import RobotNode, CommLink, NetworkManager
    print("  [OK] Imported RobotNode, CommLink, NetworkManager successfully")

    # Instantiate RobotNode
    r = RobotNode(101, "AUV-Alpha", [6500, 3000, -15])
    print(f"  [OK] Created RobotNode #{r.id} ({r.name}) at pos={r.pos.tolist()}, zone={r.get_zone()}")

    # Instantiate CommLink
    link = CommLink("link_test", 101, "robot", [6500, 3000, -15], 1, "sensor", [6400, 3100, -10])
    print(f"  [OK] Created CommLink: SNR={link.snr_db} dB, status={link.status}, throughput={link.throughput_kbps} kbps")

    # Instantiate NetworkManager
    nm = NetworkManager()
    print(f"  [OK] Initialized NetworkManager with {len(nm.robots)} default AUVs")

    # Test motion & link evaluation update
    mock_sensors = [(6400, 3100, -10), (5000, 2500, -10)]
    mock_fleet = [{
        'id': 1,
        'type': 'Cargo Ship',
        'pos': [6550.0, 3050.0],
        'depth': -5.0
    }]
    nm.update_network_state(mock_sensors, mock_fleet, frame=1, speed_mult=1.0)
    print(f"  [OK] Updated network state: {len(nm.active_links)} active links detected")
    print("SUCCESS: Network architecture test passed!")
except Exception as e:
    print(f"\n[ERROR] Error testing network manager classes: {e}")
    sys.exit(1)

# Step 3: Verify Schema Extension Compatibility
print("\n[Step 3/4] Testing JSON Schema extension compatibility...")
try:
    telemetry = nm.get_network_json_data()
    print("  [OK] Generated network JSON telemetry structure:")
    print(f"    - Robots count: {len(telemetry['robots'])}")
    print(f"    - Links count : {len(telemetry['network_links'])}")
    print(f"    - Network Stats: {telemetry['network_stats']}")

    # Merge mock dashboard data
    existing_schema = {
        "frame": 1,
        "active_sensors": 2,
        "total_sensors": 35,
        "ml_status": "ML LIVE (CPU)",
        "violations": [],
        "vessels": []
    }
    existing_schema.update(telemetry)
    json_output = json.dumps(existing_schema, indent=2)
    print("  [OK] Schema JSON output successfully formatted!")
    print("SUCCESS: Schema extension compatibility verified!")
except Exception as e:
    print(f"\n[ERROR] Error verifying JSON schema: {e}")
    sys.exit(1)

# Step 4: Verify Simulator & TMS imports
print("\n[Step 4/4] Verifying core simulator & traffic algorithm modules...")
try:
    import simulator as sim
    import traffic_algo as tms
    import mlmodel as ml

    sensors = sim.generate_sensors(num_sensors=35)
    fleet = sim.generate_fleet()
    print(f"  [OK] Simulator initialized: {len(sensors)} sensors, {len(fleet)} fleet vessels")
    print("SUCCESS: Core modules verified successfully!")
except Exception as e:
    print(f"\n[ERROR] Error importing core simulation modules: {e}")
    sys.exit(1)

print("\n================================================================================")
print("SYSTEM ENVIRONMENT & MODULE DEPENDENCIES PASSED ALL VERIFICATIONS!")
print("================================================================================")
