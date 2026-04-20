#!/usr/bin/env python3
"""
현재 시뮬레이션에 존재하는 모든 TF 프레임 출력
"""
import time
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def main():
    rclpy.init()
    node = rclpy.create_node("check_frames")
    buf = Buffer()
    TransformListener(buf, node)

    # TF 수집 대기
    print("TF 프레임 수집 중 (3초)...")
    start = time.time()
    while (time.time() - start) < 3.0:
        rclpy.spin_once(node, timeout_sec=0.1)

    # 전체 프레임 출력
    all_frames = buf.all_frames_as_string()
    print("\n" + "=" * 70)
    print("  현재 시뮬레이션에 존재하는 모든 TF 프레임:")
    print("=" * 70)
    print(all_frames)

    # port 관련 프레임만 필터링
    print("\n" + "=" * 70)
    print("  포트 관련 프레임 (검색 키워드: port, nic, sc, sfp):")
    print("=" * 70)
    lines = all_frames.split("\n")
    for line in lines:
        lower = line.lower()
        if any(k in lower for k in ["port", "nic", "sc_", "sfp"]):
            print(f"  {line}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
