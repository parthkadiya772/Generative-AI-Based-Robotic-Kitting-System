# Archive (not used by the running system)

Kept for reference and reproducibility of the UR10-era results. Nothing in
`generative_kitting/` imports or reads these files.

- `src_ur10_legacy/` — the original monolithic UR10 control script
  (`robot_control.py`, Lula IK) with its URDFs and Lula config. Replaced by the
  Isaac Sim bridge (`generative_kitting/isaac_sim_bridge.py`) with cuMotion.
  Needs `ISAACSIM_PATH` (see `.env.example`).
- `ur10_lula/AIKIDO.yaml` — the UR10 Lula robot description that sat in
  `generative_kitting/robot_config/ur10/`; unreferenced since the bridge moved to
  cuMotion (`robot_config/ur10/AIKIDO.xrdf`). The frozen UR10 config itself
  (`robot_config/ur10/robot.yaml`, URDF, XRDF, meshes) stays in place.

Archived in UR10e migration Step 9 (2026-10-14).
