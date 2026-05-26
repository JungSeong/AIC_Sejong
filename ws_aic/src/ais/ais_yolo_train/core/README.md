# AIS YOLO Approach Train

Utilities for training approach YOLO models.

- `SFP`: pose model. Predicts one `port_pair` bbox and 8 keypoints.
- `SC`: pose model. Predicts one `sc_port` bbox and 4 keypoints.
- `TASK_BOARD`: pose model. Predicts one `task_board` bbox and 4 fixed local keypoints.

`SFP` keypoints:

```text
0: port0_top_left
1: port0_top_right
2: port0_bottom_right
3: port0_bottom_left
4: port1_top_left
5: port1_top_right
6: port1_bottom_right
7: port1_bottom_left
```

`SC` keypoints:

```text
0: sc_top_left
1: sc_top_right
2: sc_bottom_right
3: sc_bottom_left
```

`TASK_BOARD` keypoints are fixed in `task_board_base_link`, not image-order corners:

```text
0: board_neg_x_pos_y
1: board_pos_x_pos_y
2: board_pos_x_neg_y
3: board_neg_x_neg_y
```

SC labels use the visible `sc_port_link` opening face corners. Older bbox-only
SC labels, and SC labels collected with the old side/top-face keypoints, are
not compatible with the current SC pose model and should be recollected.

Assumption: the insertion orientation is already corrected. With that canonical
orientation, `port0` is the left port in the image and `port1` is the right port.

## Collect labels

Run the simulator from a source-built AIC workspace with a task board and
ground-truth TF enabled. This launch command requires `aic_bringup` to be built
and sourced; the pixi environment used for the collector does not install
`aic_bringup` by itself.

For `SFP`:

```bash
cd ~/aic_sejong/ws_aic
source install/setup.bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  spawn_task_board:=true \
  nic_card_mount_0_present:=true \
  ground_truth:=true \
  start_aic_engine:=false
```

For `SC`, launch with the SC port present instead:

```bash
cd ~/aic_sejong/ws_aic
source install/setup.bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  spawn_task_board:=true \
  sc_port_0_present:=true \
  ground_truth:=true \
  start_aic_engine:=false
```

For `TASK_BOARD`, only the task board and ground-truth TF are required:

```bash
cd ~/aic_sejong/ws_aic
source install/setup.bash
ros2 launch aic_bringup aic_gz_bringup.launch.py \
  spawn_task_board:=true \
  ground_truth:=true \
  start_aic_engine:=false
```

Then:

```bash
cd ws_aic/src
pixi run python ais/ais_yolo_train/collect_dataset.py --target SFP --episodes 500
pixi run python ais/ais_yolo_train/collect_dataset.py --target SC --episodes 500
pixi run python ais/ais_yolo_train/collect_dataset.py --target TASK_BOARD --episodes 500
```

For `MAGENTA`, the collector stores full-frame images only when the magenta
feature is visible. By default it does not use the high absolute viewpoint
generator. It samples near the current TCP pose, preserves the current TCP
orientation, and applies the same z lift convention as DebugSFP
`initial_lift` (`AIC_DISTANCE_INITIAL_LIFT_M`, default 50 mm). Before capture,
it also performs a bounded xy acquisition scan so the blob is sufficiently
visible before jittered data collection starts:

```bash
pixi run python ais/ais_yolo_train/collect_dataset.py \
  --target MAGENTA \
  --episodes 500 \
  --n-viewpoints 500 \
  --board-pose-id trial12_pose_001 \
  --debug-dir /tmp/magenta_debug \
  --debug-every 20
```

Useful MAGENTA viewpoint options:

```text
--magenta-lift-m 0.050
--magenta-acquire-xy-step-m 0.015
--magenta-acquire-xy-radius-m 0.060
--magenta-acquire-min-area 300
--magenta-acquire-edge-margin-px 20
--magenta-xy-jitter-m 0.015
--magenta-z-jitter-m 0.005
--magenta-viewpoint-mode current-lift
```

Use `--magenta-viewpoint-mode absolute` only if you explicitly want the legacy
absolute viewpoint generator.

For `SFP`, `SC`, and `TASK_BOARD`, the collector publishes the same viewpoint
commands used by `ais_motion_planning/collect_dataset_v2.py`. Use
`--n-viewpoints 0` if you only want to sample the current camera pose.

The default dataset path is:

```text
ws_aic/data/yolo/approach/SFP
ws_aic/data/yolo/approach/SC
ws_aic/data/yolo/approach/TASK_BOARD
ws_aic/data/magenta_marker_cv
```

`SFP` label format:

```text
0 x_center y_center width height kpt0_x kpt0_y ... kpt7_x kpt7_y
```

`SC` label format:

```text
0 x_center y_center width height kpt0_x kpt0_y ... kpt3_x kpt3_y
```

`TASK_BOARD` uses the same 4-keypoint pose label format as `SC`, but the
keypoint identities are the fixed board-local points listed above.

All values are normalized to `0..1`. Use `--debug-dir /tmp/approach_yolo_debug`
to save images with generated labels overlaid.

## Train

Open `notebook/train_SFP.ipynb`, `notebook/train_SC.ipynb`, or
`notebook/train_TASK_BOARD.ipynb` and run the cells.
Weights are written under:

```text
ws_aic/model/ais_yolo/approach/SFP
ws_aic/model/ais_yolo/approach/SC
```

## View model output

Open `notebook/visualize_yolo_pred_SFP.ipynb` or
`notebook/visualize_yolo_pred_SC.ipynb` and run the cells. They can visualize
predictions from saved dataset images or from the live ROS camera topics inline
in the notebook.

Open `notebook/visualize_dataset_gt_SFP.ipynb` or
`notebook/visualize_dataset_gt_SC.ipynb` to visualize saved dataset ground truth
labels.

Open `notebook/eda_TASK_BOARD.ipynb` to inspect TaskBoard label statistics,
keypoint distributions, and sample overlays.

To evaluate the trained TaskBoard keypoints as PnP poses:

```bash
cd ws_aic/src
pixi run python ais/ais_yolo_train/evaluate_task_board_pose.py \
  --split val \
  --conf 0.8 \
  --records-csv /tmp/task_board_pose_records.csv \
  --consistency-csv /tmp/task_board_pose_consistency.csv
```

This validates `T_camera_task_board` quality against saved labels and checks
left/center/right pose consistency in the camera rig frame. Saved datasets do
not currently include per-frame `T_base_camera`, so `T_base_task_board` must be
validated in runtime with live camera extrinsics.

Open `notebook/visualize_triangulation_error_SFP.ipynb` or
`notebook/visualize_triangulation_error_SC.ipynb` to compare multi-camera
triangulated predictions against validation labels. These notebooks plot the
Euclidean error distribution and the 90%, 95%, and 99% empirical coverage
thresholds.
