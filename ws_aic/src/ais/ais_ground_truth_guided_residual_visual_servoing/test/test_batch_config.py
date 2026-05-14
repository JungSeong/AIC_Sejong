from __future__ import annotations

from ais_ground_truth_guided_residual_visual_servoing.batch.engine_config import (
    make_sfp_batch_config,
)


def test_sfp_batch_config_contains_only_sfp_tasks():
    config = make_sfp_batch_config(episodes=4, seed=7, diversify=True)

    assert len(config["trials"]) == 4
    for trial in config["trials"].values():
        assert set(trial["tasks"]) 
        for task in trial["tasks"].values():
            assert task["plug_type"] == "sfp"
            assert task["port_type"] == "sfp"
            assert task["cable_type"] == "sfp_sc"
            assert task["port_name"] in {"sfp_port_0", "sfp_port_1"}
