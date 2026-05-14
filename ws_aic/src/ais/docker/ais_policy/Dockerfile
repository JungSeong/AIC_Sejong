FROM docker.io/library/ros:kilted-ros-core AS build

# Install pixi
RUN apt update && apt install -y git curl && curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

# Add base dependencies
COPY ws_aic/src/aic/aic_model /ws_aic/src/aic/aic_model
COPY ws_aic/src/aic/aic_interfaces /ws_aic/src/aic/aic_interfaces
COPY ws_aic/src/aic/aic_utils /ws_aic/src/aic/aic_utils
COPY ws_aic/src/aic/robostack_mapping.json /ws_aic/src/aic/
COPY ws_aic/src/aic/pixi_env_setup.sh /ws_aic/src/aic/
COPY ws_aic/src/pixi.toml ws_aic/src/pixi.lock /ws_aic/src/

# Add other dependencies
COPY ws_aic/src/ais/ais_policy/motion_planning_node /ws_aic/src/ais/ais_policy/motion_planning_node
COPY ws_aic/src/ais/ais_policy/distance_prediction /ws_aic/src/ais/ais_policy/distance_prediction

SHELL ["/bin/bash", "-c"]
RUN cd /ws_aic/src && \
    sed -i '/ros-kilted-aic-example-policies/d; /data-gen-node/d; /ais-reinforcement-learning/d' pixi.toml && \
    pixi install
RUN rm -f \
    /ws_aic/src/ais/ais_policy/distance_prediction/distance_prediction_policy/DistancePredictionPolicy.py \
    /ws_aic/src/ais/ais_policy/distance_prediction/distance_prediction_policy/yolo_approach.py
RUN sed -i \
    -e '/^from rclpy.time import Time/d' \
    -e '/^from tf2_ros import TransformException/d' \
    -e '/^from geometry_msgs.msg import Point, Pose, Quaternion/d' \
    -e '/^from motion_planning_node.core.geometry import /d' \
    -e '/^class ModelFeedbackController:/,$d' \
    /ws_aic/src/ais/ais_policy/distance_prediction/distance_prediction_policy/model_feedback.py

COPY ws_aic/src/ais/ais_distance_prediction/model /ws_aic/src/ais/ais_distance_prediction/model
COPY ws_aic/src/model/yolo-port-keypoint-detection/weights/best.pt /models/port_detector.pt
COPY ws_aic/src/model/sfp_distance_resnet50_left_center_right_concat/best.pt /models/distance_prediction.pt
ENV AIC_YOLO_MODEL_PATH=/models/port_detector.pt
ENV AIC_DISTANCE_MODEL_PATH=/models/distance_prediction.pt
ENV AIC_DISTANCE_LABEL_PORT_FRAME_MODE=entrance
ENV AIC_DISTANCE_ALIGN_USE_PORT_FRAME_ROTATION=false
ENV AIC_DISTANCE_ALIGN_CORRECTION_X_SIGN=-1.0
ENV AIC_DISTANCE_ALIGN_CORRECTION_Y_SIGN=1.0
ENV AIC_DISTANCE_ALIGN_RETRY_ENABLED=true
ENV AIC_DISTANCE_ALIGN_RETRY_FORCE_XY_THRESHOLD_N=2.0
ENV AIC_DISTANCE_ALIGN_RETRY_FORCE_Z_THRESHOLD_N=4.0
ENV AIC_DISTANCE_ALIGN_RETRY_LATERAL_STEP_M=0.002
ENV AIC_DISTANCE_ALIGN_RETRY_LIFT_M=0.005
ENV AIC_DISTANCE_ALIGN_RETRY_FORCE_X_SIGN=-1.0
ENV AIC_DISTANCE_ALIGN_RETRY_FORCE_Y_SIGN=-1.0
ENV AIC_DISTANCE_ALIGN_RETRY_USE_TCP_FRAME=true

WORKDIR /ws_aic/src
COPY --chmod=755 <<"EOF" /entrypoint.sh
#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if [[ -z "$AIC_ROUTER_ADDR" ]]; then
  echo "AIC_ROUTER_ADDR must be provided"
  exit 1
fi

should_enable_acl() {
  [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]
}

# prepare the credentials file
if should_enable_acl; then
  if [[ ! (-n "$AIC_MODEL_PASSWD" ) ]]; then
    echo "AIC_MODEL_PASSWD must be provided"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
fi

# start the model
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'
if should_enable_acl; then
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"
exec pixi run --as-is ros2 run aic_model aic_model "$@"
EOF

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--ros-args", "-p", "policy:=distance_prediction_policy.DebugDistancePolicy", "-p", "use_sim_time:=true"]
