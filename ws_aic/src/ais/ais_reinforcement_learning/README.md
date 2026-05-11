# AIS Reinforcement Learning

SFP insertion RL module.

Current scope:

1. Run the existing YOLO approach.
2. Finish align/insert with a TF cheatcode policy and record state-action pairs.
   The state uses non-cheat signals; TF is used only to create the action label
   and debug metadata.
3. Train a small supervised action model from recorded rollouts.
4. Provide a headless RL entry point to wire into the simulator loop next.

Run the semi-cheatcode policy:

```bash
AIC_YOLO_MODEL_PATH=/home/whyz/aic_sejong/ws_aic/model/ais_yolo/approach/SFP/weights/best.pt \
AIC_DISTANCE_MODEL_PATH=/home/whyz/aic_sejong/ws_aic/model/ais_distance_prediction/sfp_distance_resnet50_left_center_right_concat/best.pt \
pixi run ros2 run aic_model aic_model \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=ais_reinforcement_learning.SfpSemiCheatcodePolicy
```

Train the supervised action model from recorded rollouts:

```bash
pixi run python -m ais_reinforcement_learning.train_supervised
```
