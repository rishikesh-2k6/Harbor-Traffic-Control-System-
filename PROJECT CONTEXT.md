PROJECT CONTEXT

Project Title: Adaptive Multi-Robot Networking in Underwater Wireless
Sensor Networks for Harbor Monitoring

Background: The existing project simulates a harbor-monitoring system
using an Underwater Wireless Sensor Network (UWSN). A grid of underwater
sensors detects vessels and estimates properties such as vessel type,
speed, direction, and weight using acoustic physics and machine-learning
models.

Current System: - Simulates a 10 km × 10 km harbor environment. -
Contains multiple underwater sensors placed at different depths. -
Tracks vessels such as cargo ships, tankers, ferries, cruisers, fishing
vessels, and speedboats. - Uses acoustic features such as sound
propagation, Doppler shift, signal-to-noise ratio, and
time-of-arrival. - Uses a machine-learning model to classify vessel
types and estimate weight. - Displays the simulation in a real-time 3D
dashboard.

Project Extension: The new objective is to extend the system from a
passive sensor network into a collaborative robotic network.

The updated system should introduce underwater robots (AUVs) that can:

1.  Receive information from nearby sensors.
2.  Communicate with neighboring robots.
3.  Establish and maintain underwater communication links.
4.  Forward information to a surface ship or buoy.
5.  Adapt to underwater constraints such as:
    -   Water currents
    -   Limited communication range
    -   Signal attenuation
    -   Dynamic topology changes
    -   Energy constraints

Core Research Problem: How can multiple underwater robots dynamically
establish and maintain communication in a challenging underwater
environment while efficiently sharing vessel information?

Expected Outcome: Design and simulate a networking algorithm inspired by
swarm behavior, where robots cooperate to exchange information and
maintain network connectivity.
