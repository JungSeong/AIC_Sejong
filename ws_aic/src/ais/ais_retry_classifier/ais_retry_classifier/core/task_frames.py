from __future__ import annotations


def sfp_port_frame_candidates(
    target_module_name: str,
    port_name: str = "sfp_port_0",
) -> tuple[str, ...]:
    if port_name.endswith("_link"):
        names = (f"{port_name}_entrance", port_name)
    else:
        names = (f"{port_name}_link_entrance", f"{port_name}_link", port_name)
    return tuple(f"task_board/{target_module_name}/{name}" for name in names)


def sfp_plug_tip_frame_candidates(
    cable_name: str = "cable_0",
    plug_name: str = "sfp_tip",
) -> tuple[str, ...]:
    names = [
        f"{cable_name}/{plug_name}_tip_link",
        f"{cable_name}/{plug_name}_link",
        f"{cable_name}/{plug_name}",
        f"{plug_name}_tip_link",
        f"{plug_name}_link",
        plug_name,
        f"{cable_name}/sfp_tip_link",
        f"{cable_name}/sfp_tip_tip_link",
        f"{cable_name}/sfp_link",
    ]
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return tuple(deduped)
