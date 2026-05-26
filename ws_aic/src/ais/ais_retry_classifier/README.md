# ais_retry_classifier

SFP insertion retry/success classifier dataset tools.

This package now captures the deployable feature set we can also use during
evaluation: a final `center_image_path` plus a small tabular vector:

```text
pred_xy_offset_mm
fz
delta_fz
fxy_norm
cmd_insert_depth_mm
```

It does not use GT `xy_offset_mm`, GT `depth_mm`, TF-derived Euclidean distance,
or `/scoring/insertion` as model inputs. `/scoring/insertion_event` is saved only
to create labels:

- `complete_insert` -> `binary_success=1`
- `partial_insert`, `side_wall_contact`, `top_surface_contact`, `timeout_or_unknown` -> `binary_success=0`

The failure class is still saved for analysis, even when training a binary
classifier.

## Capture One Episode

Run this while the policy is running and the `observations` topic plus
`/scoring/insertion_event` are active. Start the capture window just before the
insert attempt so `cmd_insert_depth_mm` means "how far the policy commanded TCP
down from insertion start".

In practice, launching Gazebo/task-board with a helper such as `sfp()` is only
the environment setup step. Do not start `retry_capture_features` immediately
after `sfp()` if the robot will sit idle or spend a long time in approach before
insertion. The capture node records a fixed time window starting at node launch.
Start it in another terminal right before the insertion policy/action reaches
the insert attempt, or run it with a long enough window and treat
`cmd_insert_depth_mm` carefully.

```bash
ros2 run ais_retry_classifier retry_capture_features --ros-args \
  -p episode_id:=sfp_complete_0001 \
  -p intended_class:=complete_insert \
  -p target_module_name:=nic_card_mount_0 \
  -p port_name:=sfp_port_0 \
  -p cable_name:=cable_0 \
  -p plug_name:=sfp_tip \
  -p observation_topic:=observations \
  -p baseline_window_s:=0.5 \
  -p duration_s:=8.0 \
  -p output_dir:=/tmp/retry_classifier_dataset
```

Output:

```text
/tmp/retry_classifier_dataset/features.csv
/tmp/retry_classifier_dataset/episodes.jsonl
/tmp/retry_classifier_dataset/images/center/<episode_id>_center.png
```

Important parameters:

- `observation_topic`: `aic_model_interfaces/msg/Observation` topic, default `observations`
- `baseline_window_s`: first window used to estimate baseline Fz and insertion-start TCP z, default `0.5`
- `distance_model_path`: optional distance prediction checkpoint override
- `distance_device`: `auto`, `cpu`, or CUDA device string, default `auto`
- `centered_xy_mm`: threshold for "near port center", default `2.0`
- `wall_xy_mm`: threshold for "far from port center", default `4.0`
- `fxy_contact_n`: side-wall lateral force threshold, default `4.0`
- `fz_contact_n`: top-surface vertical force threshold, default `4.0`
- `fz_stuck_n`: strong axial stuck-force threshold above baseline, default `8.0`
- `min_cmd_insert_depth_mm`: enough commanded insertion progress for partial-insert rules, default `5.0`
- `force_class`: override the auto label for controlled experiments

## Scenario Plan

Create a balanced plan for the four initial cases:

```bash
ros2 run ais_retry_classifier retry_make_scenarios \
  --episodes-per-class 100 \
  --output-dir /tmp/retry_classifier_plan
```

This writes:

```text
scenario_plan.csv
scenario_plan.yaml
```

The plan stores intended offsets and force hints. A separate motion runner or
manual policy should use these values to actually command the robot.

## Train Baseline

```bash
ros2 run ais_retry_classifier retry_train_baseline \
  --csv /tmp/retry_classifier_dataset/features.csv \
  --output /tmp/retry_classifier_dataset/baseline_model.json
```

The baseline is a small standardized logistic regression implemented with
NumPy over only the five tabular features. The `center_image_path` is stored for
the later multimodal model and is intentionally ignored by this baseline.

## EDA Notebook

Open:

```text
notebooks/retry_classifier_eda.ipynb
```

By default it reads:

```text
/tmp/retry_classifier_dataset/features.csv
```

Override the dataset path before launching Jupyter:

```bash
export AIC_RETRY_DATASET_DIR=/tmp/retry_classifier_dataset
# or
export AIC_RETRY_FEATURE_CSV=/tmp/retry_classifier_dataset/features.csv
```

The notebook checks class balance, missing columns, feature distributions,
scatter plots for the success/failure rules, feature correlations, and center
image samples by class.

## Recommended First Dataset

Start balanced and small:

```text
complete_insert:      100 episodes
partial_insert:      100 episodes
side_wall_contact:   100 episodes
top_surface_contact: 100 episodes
```

Then inspect the model confusion matrix and add episodes where the classes are
confused.
