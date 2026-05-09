# AIS YOLO Approach Train

Utilities for training approach YOLO models.

- `SFP`: pose model. Predicts one `port_pair` bbox and 8 keypoints.
- `SC`: pose model. Predicts one `sc_port` bbox and 4 keypoints.

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

Then:

```bash
cd ws_aic/src
pixi run python ais/ais_yolo_train/collect_dataset.py --target SFP --episodes 500
pixi run python ais/ais_yolo_train/collect_dataset.py --target SC --episodes 500
```

By default the collector publishes the same viewpoint commands used by
`ais_motion_planning/collect_dataset_v2.py`. Use `--n-viewpoints 0` if you only
want to sample the current camera pose.

The default dataset path is:

```text
ws_aic/data/yolo/approach/SFP
ws_aic/data/yolo/approach/SC
```

`SFP` label format:

```text
0 x_center y_center width height kpt0_x kpt0_y ... kpt7_x kpt7_y
```

`SC` label format:

```text
0 x_center y_center width height kpt0_x kpt0_y ... kpt3_x kpt3_y
```

All values are normalized to `0..1`. Use `--debug-dir /tmp/approach_yolo_debug`
to save images with generated labels overlaid.

## Train

Open `notebook/train_SFP.ipynb` or `notebook/train_SC.ipynb` and run the cells.
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

Open `notebook/visualize_triangulation_error_SFP.ipynb` or
`notebook/visualize_triangulation_error_SC.ipynb` to compare multi-camera
triangulated predictions against validation labels. These notebooks plot the
Euclidean error distribution and the 90%, 95%, and 99% empirical coverage
thresholds.
