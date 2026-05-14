from __future__ import annotations


def sfp_port_frame_candidates(task) -> tuple[str, ...]:
    module = str(getattr(task, "target_module_name", "") or "")
    port = str(getattr(task, "port_name", "") or "")
    if port.endswith("_link"):
        base_names = (f"{port}_entrance", port)
    else:
        base_names = (f"{port}_link_entrance", f"{port}_link", port)
    return tuple(f"task_board/{module}/{name}" for name in base_names)


def sfp_port_pair_frame_candidates(task) -> tuple[tuple[str, ...], tuple[str, ...]]:
    module = str(getattr(task, "target_module_name", "") or "")
    return (
        (
            f"task_board/{module}/sfp_port_0_link_entrance",
            f"task_board/{module}/sfp_port_0_link",
        ),
        (
            f"task_board/{module}/sfp_port_1_link_entrance",
            f"task_board/{module}/sfp_port_1_link",
        ),
    )


def sfp_plug_tip_frame_candidates(task) -> tuple[str, ...]:
    cable = str(getattr(task, "cable_name", "") or "")
    plug = str(getattr(task, "plug_name", "") or "")
    names: list[str] = []
    if plug:
        names.extend(
            [
                f"{cable}/{plug}_tip_link",
                f"{cable}/{plug}_link",
                f"{cable}/{plug}",
                f"{plug}_tip_link",
                f"{plug}_link",
                plug,
            ]
        )
    names.extend(
        [
            f"{cable}/sfp_tip_link",
            f"{cable}/sfp_tip_tip_link",
            f"{cable}/sfp_link",
        ]
    )
    deduped = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return tuple(deduped)
