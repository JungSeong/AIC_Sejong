# AIS Stiffness/Damping Calibration

Headless calibration sweep for AIC Cartesian impedance control.

It sends the same no-contact Cartesian waypoint repeatedly while sweeping
`target_stiffness` and `target_damping`, then records `commanded_delta` vs
`reported_tcp_delta` from `/aic_controller/controller_state`.

## Run

After adding or updating this package:

```bash
pixi install
```

Terminal 1:

```bash
pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py \
  gazebo_gui:=false launch_rviz:=false start_aic_engine:=false
```

Terminal 2:

```bash
pixi run ros2 run ais_stiffness_damping_calibration ais_stiffness_damping_calibration \
  --delta-m 0.02 0.0 0.0 \
  --stiffness-xyz 75,75,50 100,100,50 150,150,75 200,200,100 \
  --damping-xyz 30,30,20 60,60,30 90,90,45 \
  --duration-s 2.5 --return-s 1.5 \
  --ros-args -p use_sim_time:=true
```

Distance sweep along one axis:

```bash
pixi run ros2 run ais_stiffness_damping_calibration ais_stiffness_damping_calibration \
  --axis x --distance-m 0.02 0.05 0.10 0.20 \
  --stiffness-xyz 100,100,50 150,150,75 200,200,100 \
  --damping-xyz 60,60,30 90,90,45 120,120,60 \
  --duration-s 6.0 --return-s 4.0 \
  --ros-args -p use_sim_time:=true
```

Arbitrary Cartesian deltas:

```bash
pixi run ros2 run ais_stiffness_damping_calibration ais_stiffness_damping_calibration \
  --delta-m-list 0.02,0,0 0.10,0,0 0.20,0,0 0,0.10,0 0,0,-0.05 \
  --duration-s 6.0 --return-s 4.0 \
  --ros-args -p use_sim_time:=true
```

Results are written under:

```text
ais/ais_stiffness_damping_calibration/outputs/<timestamp>/
```

## Visualize

Use the latest sweep output:

```bash
pixi run ros2 run ais_stiffness_damping_calibration ais_stiffness_damping_visualize
```

Or pass a specific CSV:

```bash
pixi run ros2 run ais_stiffness_damping_calibration ais_stiffness_damping_visualize \
  ais/ais_stiffness_damping_calibration/outputs/<timestamp>/stiffness_damping_sweep.csv
```

This writes:

```text
sweep_overview.png
ranked_options.csv
score_by_delta.csv
summary.md
```

The score ranks each option within each requested delta and combines:

- `final_error_norm_mm`: 30%
- overshoot norm from `max_overshoot_*_mm`: 35%
- `tail_xy_peak_to_peak_mm`: 20%
- `peak_abs_velocity_z_mps`: 15%

## Random Walk Check

After choosing an impedance option, test whether tracking error changes with
different source/destination poses. The command below samples random source
positions near the current TCP, then runs random-walk segments no longer than
10 cm.

```bash
pixi run ros2 run ais_stiffness_damping_calibration ais_stiffness_damping_random_walk \
  --source-count 5 \
  --steps-per-source 12 \
  --workspace-half-range-m 0.12,0.12,0.04 \
  --min-step-m 0.02 \
  --max-step-m 0.10 \
  --stiffness-xyz 200,200,100 \
  --damping-xyz 120,120,60 \
  --duration-s 4.0 \
  --ros-args -p use_sim_time:=true
```

This keeps orientation fixed to the initial TCP orientation, so the first check
isolates Cartesian position/source/destination effects. It writes:

```text
random_walk.csv
random_walk.json
random_walk_summary.csv
random_walk_summary.md
```

## Hold Check

To separate movement error from no-op hold drift, command the current TCP pose
back to itself repeatedly. The default is 5 hold commands. Each hold command
uses the latest reported TCP position and orientation.

```bash
pixi run ros2 run ais_stiffness_damping_calibration ais_stiffness_damping_hold_check \
  --count 5 \
  --stiffness-xyz 200,200,100 \
  --damping-xyz 120,120,60 \
  --duration-s 4.0 \
  --ros-args -p use_sim_time:=true
```

This writes:

```text
hold_check.csv
hold_check.json
hold_check_summary.md
```

If `hold_drift_z_mm` is still around `-0.65`, the z bias is present even when
`commanded_delta_z` is zero. If it stays near zero, the random-walk z error came
from movement/settling rather than a no-op hold bias.

Key columns:

- `commanded_delta_*_m`
- `reported_tcp_delta_*_m`
- `requested_delta_*_m`
- `requested_delta_norm_m`
- `tracking_ratio_*`
- `final_error_*_mm`
- `max_overshoot_*_mm`
- `tail_peak_to_peak_*_mm`
- `peak_abs_velocity_z_mps`
- `delta_error_norm_mm`
- `overshoot_norm_mm`
- `hold_drift_*_mm`
- `hold_drift_norm_mm`
