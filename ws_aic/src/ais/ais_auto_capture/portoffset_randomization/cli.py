"""PortOffset randomization runner의 역할별 CLI 정의."""

from __future__ import annotations

import argparse

from .constants import ENGINE_SETUP, POLICY_MODULE


def _add_trial_args(parser: argparse.ArgumentParser) -> None:
    """trial 개수, 순서, simulator 시작과 관련된 기본 인자를 추가한다."""
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--port-types", default="sfp,sc")
    parser.add_argument(
        "--port-order",
        choices=("round_robin", "random"),
        default="random",
    )
    parser.add_argument("--color-log", dest="color_log", action="store_true", default=True)
    parser.add_argument("--no-color-log", dest="color_log", action="store_false")
    parser.add_argument("--samples-per-trial", type=int, default=24)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument(
        "--trial-timeout-s",
        type=float,
        default=None,
        help="Defaults to time-limit-s + 180s.",
    )
    parser.add_argument("--distrobox", default="aic_eval")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rootless-distrobox", action="store_true")
    parser.add_argument("--engine-setup", default=ENGINE_SETUP)
    parser.add_argument("--policy", default=POLICY_MODULE)
    parser.add_argument("--policy-start-wait-s", type=float, default=5.0)
    parser.add_argument("--robot-joint-noise-deg", type=float, default=4.0)
    parser.add_argument("--cable-rpy-noise-deg", type=float, default=20.0)


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """dataset 경로와 Hugging Face 업로드 관련 인자를 추가한다."""
    parser.add_argument("--dataset-version", default="")
    parser.add_argument(
        "--push-to-hub",
        dest="push_to_hub",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-push-to-hub",
        dest="push_to_hub",
        action="store_false",
    )
    parser.add_argument(
        "--vision-offset-repo-id",
        default="aic-sejong-team/aic-vision-offset-dataset",
    )
    parser.add_argument("--vision-offset-hf-revision", default="main")
    parser.add_argument("--vision-offset-hf-path-in-repo", default="")
    parser.add_argument(
        "--upload-on-port-type",
        choices=("", "sfp", "sc"),
        default="",
    )
    parser.add_argument("--hf-private", action="store_true", default=False)


def _add_pose_args(parser: argparse.ArgumentParser) -> None:
    """port-local XYZ/RPY sampling 범위 인자를 추가한다."""
    parser.add_argument("--port-xy-limit-mm", type=float, default=50.0)
    parser.add_argument("--port-z-limit-mm", type=float, default=100.0)
    parser.add_argument("--dx-min-mm", type=float, default=-50.0)
    parser.add_argument("--dx-max-mm", type=float, default=50.0)
    parser.add_argument("--dy-min-mm", type=float, default=-50.0)
    parser.add_argument("--dy-max-mm", type=float, default=50.0)
    parser.add_argument("--dz-min-mm", type=float, default=0.0)
    parser.add_argument("--dz-max-mm", type=float, default=100.0)
    parser.add_argument("--port-roll-limit-deg", type=float, default=25.0)
    parser.add_argument("--port-pitch-limit-deg", type=float, default=25.0)
    parser.add_argument("--port-yaw-limit-deg", type=float, default=35.0)
    parser.add_argument("--roll-min-deg", type=float, default=None)
    parser.add_argument("--roll-max-deg", type=float, default=None)
    parser.add_argument("--pitch-min-deg", type=float, default=None)
    parser.add_argument("--pitch-max-deg", type=float, default=None)
    parser.add_argument("--yaw-min-deg", type=float, default=None)
    parser.add_argument("--yaw-max-deg", type=float, default=None)
    parser.add_argument("--rpy-norm-max-rad", type=float, default=None)
    parser.add_argument("--actual-rpy-norm-max-rad", type=float, default=None)
    parser.add_argument("--base-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--min-visible-cameras", type=int, default=1)
    parser.add_argument("--visibility-margin-px", type=float, default=8.0)


def _add_stability_args(parser: argparse.ArgumentParser) -> None:
    """로봇 정지 판정과 촬영 안정화 인자를 추가한다."""
    parser.add_argument("--capture-settle-s", type=float, default=1.0)
    parser.add_argument("--stability-timeout-s", type=float, default=5.0)
    parser.add_argument("--stable-samples", type=int, default=5)
    parser.add_argument("--stability-poll-s", type=float, default=0.1)
    parser.add_argument("--linear-speed-tol-mm-s", type=float, default=2.0)
    parser.add_argument("--angular-speed-tol-deg-s", type=float, default=2.0)


def _add_world_args(parser: argparse.ArgumentParser) -> None:
    """Gazebo 조명과 배경 Gaussian randomization 인자를 추가한다."""
    parser.add_argument(
        "--randomize-lighting",
        dest="randomize_lighting",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-randomize-lighting",
        dest="randomize_lighting",
        action="store_false",
    )
    parser.add_argument("--light-intensity-scale-min", type=float, default=0.65)
    parser.add_argument("--light-intensity-scale-max", type=float, default=1.35)
    parser.add_argument("--light-color-jitter", type=float, default=0.12)
    parser.add_argument("--light-pose-xy-jitter-m", type=float, default=0.25)
    parser.add_argument("--light-pose-z-jitter-m", type=float, default=0.20)
    parser.add_argument("--ambient-min", type=float, default=0.0)
    parser.add_argument("--ambient-max", type=float, default=0.08)
    parser.add_argument("--background-min", type=float, default=0.08)
    parser.add_argument("--background-max", type=float, default=0.20)


def _add_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    """policy와 simulator PGID 종료 단계 및 cleanup 인자를 추가한다."""
    parser.add_argument("--policy-stop-grace-s", type=float, default=10.0)
    parser.add_argument("--post-summary-wait-s", type=float, default=3.0)
    parser.add_argument("--sim-sigint-grace-s", type=float, default=5.0)
    parser.add_argument("--sim-cleanup-grace-s", type=float, default=2.0)
    parser.add_argument("--sim-sigkill-grace-s", type=float, default=1.0)
    parser.add_argument("--between-trial-wait-s", type=float, default=3.0)
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """수집 runner와 보조 도구가 공유할 CLI parser를 구성한다."""
    parser = argparse.ArgumentParser(
        description="Collect PortOffsetCollect samples from randomized trials."
    )
    _add_trial_args(parser)
    _add_dataset_args(parser)
    _add_pose_args(parser)
    _add_stability_args(parser)
    _add_world_args(parser)
    _add_lifecycle_args(parser)
    return parser


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    return build_parser().parse_args()
