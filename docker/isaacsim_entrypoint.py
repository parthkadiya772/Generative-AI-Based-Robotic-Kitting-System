"""Headless Isaac Sim bootstrap for the kitting bridge."""

import asyncio
import os
import runpy

from isaacsim import SimulationApp


public_ip = os.environ.get("LIVESTREAM_PUBLIC_IP", "").strip()
livestream_port = os.environ.get("LIVESTREAM_PORT", "49100")

stream_args = [
    "--/isaac/startup/ros_bridge_extension=",
    f"--/app/livestream/port={livestream_port}",
]

if public_ip:
    stream_args.append(
        f"--/app/livestream/publicEndpointAddress={public_ip}"
    )
    print(
        f"[KITTING] WebRTC endpoint: {public_ip}:{livestream_port} "
        "(TCP) and 47998 (UDP)",
        flush=True,
    )
else:
    print(
        "[KITTING] WebRTC public endpoint is unset; remote clients must "
        "route directly to this host",
        flush=True,
    )

simulation_app = SimulationApp({
    "headless": True,
    "hide_ui": False,
    "extra_args": stream_args,
}, experience="/isaac-sim/apps/isaacsim.exp.full.streaming.kit")
print("[KITTING] SimulationApp created", flush=True)

try:
    import omni.usd
    print("[KITTING] omni.usd imported", flush=True)

    project_root = os.environ.get("KITTING_PROJECT_ROOT", "/workspace/robot_in_air")
    scene_path = os.path.join(project_root, "AIKIDO.usd")
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"USD scene not found: {scene_path}")

    print(f"[KITTING] Opening stage: {scene_path}", flush=True)
    omni.usd.get_context().open_stage(scene_path)
    simulation_app.update()
    print("[KITTING] Stage update complete", flush=True)

    bridge_path = os.path.join(project_root, "generative_kitting", "isaac_sim_bridge.py")
    os.environ["KITTING_BRIDGE_AUTOSTART"] = "0"
    print(f"[KITTING] Loading bridge: {bridge_path}", flush=True)
    bridge_namespace = runpy.run_path(bridge_path, run_name="__main__")
    print("[KITTING] Scheduling bridge coroutine", flush=True)
    simulation_app.run_coroutine(
        bridge_namespace["start_bridge"](),
        run_until_complete=False,
    )

    while simulation_app.is_running() and not simulation_app.is_exiting():
        simulation_app.update()
finally:
    simulation_app.close()
