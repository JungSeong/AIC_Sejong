# AIS Ground Truth Guided Residual Visual Servoing

SFP-only training package for jointly improving:

- approach YOLO pose prediction, by saving YOLO failure scenes in the existing
  SFP YOLO pose dataset format;
- distance prediction, by saving image/backbone/ground-truth residual samples
  compatible with the existing distance prediction dataset loader.

The training policy uses ground truth only for data labels, success checks, and
training-time fallback motion. Runtime policy inputs should still come from
images, force/torque, and learned model predictions.

Default artifact layout:

```text
ws_aic/data/ais_ground_truth_guided_residual_visual_servoing/v0/yolo
ws_aic/data/ais_ground_truth_guided_residual_visual_servoing/v0/distance_prediction
ws_aic/model/ais_ground_truth_guided_residual_visual_servoing/v0/yolo
ws_aic/model/ais_ground_truth_guided_residual_visual_servoing/v0/distance_prediction
```

Set `AIC_GRVS_VERSION=v1` to create a new experiment version.

Package layout:

```text
ais_ground_truth_guided_residual_visual_servoing/
  core/       shared config, geometry adapters, image conversion, task frames
  data/       YOLO and distance-prediction dataset writers
  policies/   rollout/data-collection policies
  batch/      SFP-only batch configs and collect/test runners
  training/   supervised training entry points
```

Batch loop:

```text
collect batch -> train on accumulated replay buffer -> test trained policy -> collect batch ...
```

The replay buffer is the versioned data root. New batches append to:

```text
ws_aic/data/ais_ground_truth_guided_residual_visual_servoing/v0/yolo
ws_aic/data/ais_ground_truth_guided_residual_visual_servoing/v0/distance_prediction
ws_aic/data/ais_ground_truth_guided_residual_visual_servoing/v0/replay_buffer/batches.jsonl
```

Training alignment is capped at 10 commands by default:

```bash
export AIC_GRVS_ALIGN_MAX_ATTEMPTS=10
```

The training action follows the original force-first retry pseudocode:

```text
low-pass force/torque
if |Fz| is high:
  lift 10 mm
else if |Fx| or |Fy| is high:
  lift 50 mm
else:
  run GT-guided distance-delta action
```

Run the training policy:

```bash
AIC_YOLO_MODEL_PATH=/home/whyz/aic_sejong/ws_aic/model/ais_yolo/approach/SFP/weights/best.pt \
AIC_DISTANCE_MODEL_PATH=/home/whyz/aic_sejong/ws_aic/model/ais_distance_prediction/sfp_distance_resnet50_left_center_right_concat/best.pt \
pixi run ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=ais_ground_truth_guided_residual_visual_servoing.SfpGrvsTrainingPolicy
```

Run the passive capture node alongside any policy/controller that is already
moving the robot. It only subscribes to `observations` and TF; it does not send
motion commands:

```bash
AIC_GRVS_VERSION=v1 pixi run grvs_capture_node --ros-args \
  -p use_sim_time:=true \
  -p target_module_name:=nic_card_mount_0 \
  -p port_name:=sfp_port_0 \
  -p cable_name:=cable_0 \
  -p plug_name:=sfp_tip \
  -p sample_interval_s:=0.1 \
  -p record_distance:=true \
  -p record_rotation:=true
```

Joint training entry point:

```bash
pixi run grvs_train_joint --dry-run
```

Run batch cycles after starting the simulator with
`ground_truth:=true start_aic_engine:=false`:

```bash
AIC_GRVS_VERSION=v0 pixi run grvs_batch_round \
  --cycles 100 \
  --max-runtime-hours 6 \
  --domain-id 20 \
  --collect-episodes 20 \
  --test-episodes 10 \
  --continue-on-collect-error \
  --continue-on-test-error
```

One cycle is:

```text
collect -> train -> test
```

Every cycle writes:

```text
data/<package>/<version>/metrics/episodes.jsonl
model/<package>/<version>/snapshots/cycle_###/
```

Or run the stages manually:

```bash
AIC_GRVS_VERSION=v0 pixi run grvs_collect_batch --domain-id 20 --episodes 20
AIC_GRVS_VERSION=v0 pixi run grvs_train_joint
AIC_GRVS_VERSION=v0 pixi run grvs_test_batch --domain-id 20 --episodes 10
```
