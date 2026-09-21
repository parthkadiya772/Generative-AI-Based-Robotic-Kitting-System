# SPEC: Robot Self-Filtering → cuMotion Obstacles

**Target:** Isaac Sim 6.0.1, Python API (Script Editor / VS Code Edition), no ROS 2.
**Robot:** UR10 + Robotiq 2F-140 + wrist camera, on a *vagn* riding a gantry.
**Sensors:** 4 corner cameras + 2 overhead (bin, kitting tray) + 1 wrist camera.

---

## 0. INSTRUCTIONS FOR THE IMPLEMENTING AGENT

Read this section fully before writing any code.

**This document is a specification, not an implementation.** It defines contracts,
invariants and failure modes. It deliberately does not give you working API calls for
most stages, because the author could not verify them against this specific Isaac Sim
build.

Rules:

1. **Verify every API before you use it.** Isaac Sim moved namespaces between 4.5,
   5.x and 6.0. Semantics, sensors, replicator annotators and motion generation all
   changed. Use §11 to introspect the installed build. Do not write an import from
   memory.

2. **Code blocks are marked. Respect the marks.**
   - `PURE MATH — safe to use as written`: version-independent, verifiable by
     inspection. Use these directly.
   - `CONTRACT ONLY — do not copy`: describes intent and signature shape. You must
     find the real API yourself.

3. **Stop and report rather than guess.** If an API in §11 does not exist, or
   behaves differently from what a stage assumes, halt and say so. Do not substitute
   something that looks similar. A wrong obstacle API produces a robot that silently
   fails to avoid obstacles — the worst possible failure here.

4. **Build in the order given in §10.** It is ordered by which check fails fastest,
   not by data flow. Do not implement the whole pipeline and then test.

5. **One assumption in §6 is load-bearing and must be verified first.** The entire
   obstacle architecture rests on it. It is called out explicitly there.

6. **Every stage has an exit check.** Do not proceed to the next stage until the
   current one passes. These are in §9, cross-referenced from each stage.

---

## 1. Problem statement

Six cameras observe the workspace. All of them see the robot. Unless robot points are
removed from the cloud, the robot's own collision spheres end up inside obstacles at
the start state, and the planner has no legal configuration to plan from.

Observed symptom when this happens: **planner returns success, robot does not move.**

The wrist camera is the dominant case — it is centimetres from the gripper, so a
large fraction of its frame is robot.

---

## 2. Pipeline contract

```
PER CAMERA (7x)
  in:  depth image, semantic segmentation, camera intrinsics, camera world pose
  out: Nx3 float32 points, world frame, robot pixels removed

MERGE
  in:  7 point arrays
  out: single Nx3 array, world frame

FILTER CHAIN  (see §5)
  A. geometric robot rejection
  B. voxel downsample
  C. cap to N_MAX

  NOTE: no scene cropping, no ground/plane removal. Everything that is
  not the robot assembly is a real obstacle and must survive to cuMotion.

OBSTACLE SYNC
  in:  <= N_MAX voxel centres
  out: cuMotion world populated, planner-ready
```

**Global invariant:** at every stage boundary, points are `float32`, shape `(N, 3)`,
in the **world frame**, in metres. Any stage that changes frame or units is a bug.

---

## 3. Stage 1 — Semantic labelling

### Contract

Every prim that is part of the robot assembly carries one shared semantic class, so
that segmentation yields a single ID set to reject.

### What to label

| Subtree | Why |
|---|---|
| UR10 links | seen by all cameras |
| Robotiq 2F-140 links | dominates wrist camera frame |
| wrist camera body / mount | is robot, and occludes |
| `vagn` | moves with robot, not an obstacle |
| `gantry` carriage | same |
| cabling / dress pack meshes | flap, generate noise |

### What NOT to label

Bin, kitting tray, table, parts being picked, static frame. These are what you want
to keep.

### API to find

```
CONTRACT ONLY — do not copy

apply_semantic_label(prim, class_name) -> None
  Applies a "class"-type semantic label to a single Gprim.
```

The helper lives in the semantics utils module. Its name changed across versions
(older builds used an "add/update semantics" style name; newer ones use an
"add labels" style). Introspect the module — do not assume.

Walk the subtree yourself and label every `Gprim`/`Mesh` descendant. Labelling only
the root Xform is not sufficient.

### Known hazard: instanceable assets

The robot asset path contains `ur10_instanceable`, which indicates USD instancing.
**Semantics applied to an instance proxy may not resolve in segmentation output.**

If Check 1 (§9) shows the robot unlabelled, you have two options: label the prototype
prim in the referenced layer, or un-instance the asset. Report which you chose.

### Exit check
→ Check 1, then Check 2.

---

## 4. Stage 2 — Capture and deproject

### Annotator contract

Two annotators per camera, **from the same render product** so they are pixel-aligned:

| Purpose | Requirement |
|---|---|
| depth | Z-depth along the optical axis, **not** radial range |
| segmentation | per-pixel integer class IDs + an ID→label mapping, **not** colorized |

Isaac Sim offers both a planar-depth and a radial-range annotator. The maths in this
document assumes **planar** depth. If you use the radial one, the deprojection is
different and your cloud will be subtly wrong — wrong in a way that looks almost
right, which is worse than obviously broken.

### Intrinsics

Obtain the intrinsics matrix from the camera object's own accessor. **Do not compute
fx/fy from focal length and aperture by hand** — USD aperture defaults are easy to get
wrong, and a silently wrong fx uniformly scales your entire cloud, which passes visual
inspection but wrecks collision geometry.

Intrinsics are static unless resolution or focal length changes. Cache them.

### Camera pose

```
CONTRACT ONLY — do not copy

get_camera_world_pose(camera) -> (position_xyz, quaternion_wxyz)
```

Confirm the quaternion ordering of whatever accessor you use. Isaac Sim core APIs
generally return **wxyz**; some other APIs return xyzw. The maths below assumes wxyz.
Getting this wrong rotates your entire cloud.

**The wrist camera pose changes every frame.** Re-read it on every capture. Caching it
smears its cloud across the workspace as the arm moves. Corner and overhead cameras
may be cached.

### Class ID resolution

Segmentation IDs are **not stable across runs**. Re-resolve the robot ID set from the
ID→label mapping every frame. It is cheap. Hardcoding an ID will work until it
silently doesn't.

The mapping's value type varies by version — it may be a dict like `{"class": "..."}`
or a plain string. Handle both.

### Deprojection

```
PURE MATH — safe to use as written

Isaac Sim / USD camera convention: +X right, +Y up, -Z forward.
Image row index v increases DOWNWARD.

Given pixel (u, v) with planar depth d, and intrinsics fx, fy, cx, cy:

    x_cam =  (u - cx) * d / fx
    y_cam = -(v - cy) * d / fy      # image Y down  -> camera Y up
    z_cam = -d                      # camera looks down -Z

Both sign flips matter. Getting one wrong mirrors the cloud, and a mirrored
cloud frequently lands on top of the robot — reproducing the exact symptom
this whole pipeline exists to prevent.

Then: p_world = R(q) @ p_cam + t
```

```python
# PURE MATH — safe to use as written
import numpy as np

def quat_wxyz_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def deproject(depth, K, valid_mask, cam_pos, cam_quat_wxyz):
    """depth: (H,W) planar depth. valid_mask: (H,W) bool, robot already removed.
    Returns (N,3) float32 in world frame."""
    vs, us = np.nonzero(valid_mask)
    if vs.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    d = depth[vs, us].astype(np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.stack([
         (us.astype(np.float32) - cx) * d / fx,
        -(vs.astype(np.float32) - cy) * d / fy,
        -d,
    ], axis=1)
    R = quat_wxyz_to_matrix(cam_quat_wxyz)
    return (p_cam @ R.T + np.asarray(cam_pos, dtype=np.float32)).astype(np.float32)
```

Validity mask should combine: finite depth, depth within `[z_min, z_max]`, and pixel
class not in the robot ID set.

### Warm-up

Annotators return `None` for the first few frames until they initialise. Guard every
read. You will need to step the simulation (or the replicator orchestrator) before
data is valid.

### Exit check
→ Check 3. **Do not proceed past Check 3 under any circumstances.**

---

## 5. Stage 3 — Filter chain

### Design decision: keep the whole scene

**Only the robot assembly is removed.** Floor, table, walls, bin, kitting tray and
fixtures are all real obstacles the robot must avoid, and they stay in the cloud.
There is **no workspace crop and no ground-plane removal**. Do not add them.

The only spatial rejection permitted is the per-camera `[z_min, z_max]` depth range
in Stage 2, which exists to reject near-field sensor noise and far-field background
beyond the cell — not to sculpt the scene.

Two consequences follow. Both are budget/accuracy issues, not correctness issues, but
both must be measured rather than assumed:

**1. Voxel count is large.** Rough estimate for 7 cameras at 640×480 with the full
scene retained: order **12,000–25,000 voxels at 4 cm**. This is 25–50× above a
typical `N_MAX`. Expect to need a coarser voxel (6–8 cm) and a much larger pool.

Constraint on voxel size: it must stay **smaller than the thinnest object that
matters**. A bin wall of 8 mm will disappear at an 8 cm voxel if the surface samples
fall unluckily. Measure the thinnest real obstacle before choosing.

**2. Occlusion holes are a real collision risk.** A perceived surface has gaps
wherever no camera had line of sight. The planner treats a gap as free space and will
route through it. Analytic geometry cannot fail this way; perceived geometry always
can.

Likely blind spots in this cell: the bin's inner far wall, surfaces behind fixtures,
and the underside of the tray. **Check 4b** below exists specifically to find these.
If a critical surface has holes that camera placement cannot close, report it — that
surface may need an analytic primitive as a backstop regardless of this decision.

### A. Geometric robot rejection — **do not skip**

Segmentation is not pixel-perfect. At silhouette edges you get mixed pixels: depth
belongs to the robot, class ID belongs to background. A few hundred such points,
hugging the arm outline, is enough to put the start state back in collision.

```python
# PURE MATH — safe to use as written
def reject_near_robot(points, sphere_centers_world, sphere_radii, margin=0.05):
    """Drop points inside any robot collision sphere inflated by `margin`."""
    if points.shape[0] == 0 or len(sphere_centers_world) == 0:
        return points
    C = np.asarray(sphere_centers_world, dtype=np.float32)
    r = np.asarray(sphere_radii, dtype=np.float32) + margin
    keep = np.ones(points.shape[0], dtype=bool)
    CHUNK = 64                      # bounds the distance matrix
    for i in range(0, C.shape[0], CHUNK):
        c, rr = C[i:i+CHUNK], r[i:i+CHUNK]
        d2 = ((points[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        keep &= (d2 > (rr ** 2)[None, :]).all(axis=1)
    return points[keep]
```

```
CONTRACT ONLY — do not copy

get_robot_sphere_world_poses(robot, joint_state) -> (centers Nx3, radii N)
  Forward-kinematics the XRDF collision spheres to world frame at the
  current configuration.
```

Find this on the cuMotion robot/kinematics object. It has a useful side effect: if
the spheres are misplaced, Check 4 shows it as a robot-shaped hole in the wrong
location.

**Known gap:** `vagn` and `gantry` currently have no collision spheres in the XRDF, so
this pass does not cover them. Until spheres are added, use an explicit AABB around
the vagn computed from its live world pose. Flag this in your implementation as a
temporary measure.

### B. Voxel downsample

```python
# PURE MATH — safe to use as written
def voxel_downsample(points, voxel=0.06):
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return ((keys[idx].astype(np.float32) + 0.5) * voxel).astype(np.float32)
```

### C. Cap

Truncate to `N_MAX`.

**Truncation here is dangerous and must be treated as an error, not a warning.**
`np.unique` returns voxels in sorted key order, so truncating keeps one spatial
region and silently discards another — the planner then sees a scene with a whole
section missing. This is worse than a coarse voxel everywhere.

If truncation occurs: log loudly, and increase the voxel size or `N_MAX` until it
stops. Do not ship a configuration that truncates in normal operation.

### Starting parameters

| Parameter | Value | Rationale |
|---|---|---|
| voxel size | 0.06 m | full-scene retention; **must stay below thinnest real obstacle** |
| `N_MAX` | **measure — expect thousands, not hundreds** | see §6 |
| rejection margin | 0.05 m | covers mixed-pixel fringe |
| `z_min`, `z_max` | 0.15, 4.0 | sensor noise / beyond-cell only, not scene sculpting |

The `N_MAX` figure is the main open question in this design. Measure it first (§6),
because if the achievable pool size is far below the voxel count, the full-scene
approach needs revisiting before anything else is built.

### Exit check
→ Check 4, then Check 4b.

---

## 6. Stage 4 — Obstacle sync

### ⚠ Load-bearing assumption — verify this first

> **cuMotion cannot update the properties (size, shape) of an existing obstacle.
> Only the transform and the collision-enabled state can change at runtime.**

This is stated in the Isaac Sim 6.0 cuMotion world interface documentation. The entire
architecture below follows from it.

**Verify before implementing.** If property updates turn out to be supported, or the
constraint has changed, the fixed-pool design is unnecessary — stop and report rather
than building on an assumption that no longer holds.

### Architecture: fixed obstacle pool

Given the constraint, you cannot create and destroy obstacles per frame. Instead:

1. **At init:** allocate a fixed pool of `N_MAX` identical cubes, edge length = voxel
   size, parked far outside the workspace, all collision-disabled.
2. **Per update, for `k = min(len(voxels), N_MAX)`:** set transform to voxel centre,
   enable collision.
3. **For indices `k .. N_MAX-1`:** disable collision. Leave parked.
4. **Sync:** transforms every update; properties only when `k` changed.

Nothing is allocated or freed at runtime.

### APIs to find

```
CONTRACT ONLY — do not copy

create_fixed_cube(prim_path, position, size) -> handle
set_obstacle_transform(handle, position) -> None
set_obstacle_collision_enabled(handle, bool) -> None

world_binding.synchronize_transforms()   # fast path, every update
world_binding.synchronize_properties()   # only when enabled-set changed
```

Two integration routes exist. Determine which your build supports and report your
choice:

- **Route A — USD prims + `SceneQuery` / `WorldBinding`.** Create the pool as USD
  cubes with physics collision API under one parent prim, include that parent in the
  tracked prims, let `WorldBinding` push them into cuMotion. Better documented.
- **Route B — direct `CumotionWorldInterface` population.** The 6.0 docs note the
  world interface can be populated directly from core API objects. Fewer moving parts
  if the API supports it.

### Pool sizing — measure, do not guess

```
CONTRACT ONLY — do not copy

for n in (50, 100, 200, 500, 1000, 2000):
    t0 = now()
    populate n obstacles; synchronize; run one plan
    report elapsed
```

Pick the largest `n` whose total fits your perception period. If 500 cuboids costs
300 ms, you cannot run at 30 Hz — that is fine, see §7.

### Obstacle cache

cuMotion preallocates capacity per obstacle type at load time. **Exceeding it causes
updates to fail or obstacles to be silently dropped** — which presents as collision
avoidance mysteriously not working, with no error. Set capacity to at least `N_MAX`
plus headroom for static scene obstacles. Find the parameter name via §11.

### Robot base transform

cuMotion plans in the **robot base frame**. The vagn and gantry move the base. The
world interface exposes a method to update the world→robot-root transform; it must be
called whenever the base moves, or every obstacle lands in the wrong place.

This is a strong candidate cause of "obstacles appear on top of the robot".

### Exit check
→ Check 5, then Check 6.

---

## 7. Stage 5 — Timing

**Never run perception inside the physics step callback.** Depth capture,
segmentation readback, deprojection and world sync are far too slow for a 60 Hz tick.
Blocking there freezes the entire simulation — which is indistinguishable from "the
robot is not moving".

| Loop | Rate | Does |
|---|---|---|
| physics | 60 Hz | execute current trajectory point only |
| perception | 2–5 Hz | capture, filter, voxelise, push to cuMotion |
| plan | on demand | new goal, or material world change |

Replan on a new goal or a meaningful change in the voxel set — not every frame.
Compare successive voxel key sets with a cheap hash and skip the replan when nothing
moved.

### Exit check
→ Check 7.

---

## 8. Failure mode reference

| Symptom | Likely cause |
|---|---|
| Planner reports success, robot never moves | start state in collision — self-points survived |
| Cloud mirrored / upside down | sign error in Y or Z deprojection term |
| Cloud rotated wrongly | quaternion ordering (wxyz vs xyzw) |
| Cloud uniformly scaled wrong | intrinsics computed by hand instead of read from camera |
| Cloud smears when arm moves | wrist camera pose cached, not re-read |
| Whole sim freezes, sim time stalls | perception inside physics callback |
| Obstacles look right, robot still collides | vagn / gantry missing collision spheres in XRDF |
| Obstacles wrong only when gantry moves | robot base transform not updated |
| Works at 200 voxels, fails at 2000 | obstacle cache overflow |
| Thin fringe of points hugging the arm | mixed pixels — increase rejection margin |
| Robot drives through a bin wall | occlusion hole in that surface — see Check 4b |
| Whole region of the scene missing from obstacles | `N_MAX` truncation — sorted-order cut |
| Thin obstacle vanishes from voxel set | voxel size larger than the obstacle |
| Robot unlabelled in segmentation | instanceable asset — semantics on instance proxy |

---

## 9. Checks

Each stage's exit check. Do not skip ahead.

**Check 1 — labels resolve.** Dump the ID→label mapping for one camera. The robot
class must be present. If absent, Stage 1 failed — most likely instancing.

**Check 2 — mask is doing something.** Print masked-pixel fraction per camera. The
wrist camera should mask a large share of its frame. If it masks ~0%, labels are not
reaching the gripper.

**Check 3 — cloud lands correctly.** Render surviving points as debug spheres and
inspect in the viewport. They must lie on bin and tray surfaces. Mirrored, rotated,
scaled or centred-on-robot means the deprojection, quaternion ordering, intrinsics or
camera pose is wrong. **Hard gate — do not proceed.**

**Check 4 — robot-shaped hole.** A clean void where the arm is. A partial hole with a
fringe along the silhouette means the rejection margin is too small.

**Check 4b — no occlusion holes in critical surfaces.** With the robot parked clear of
the workspace, inspect the voxel set for gaps in surfaces the robot must not hit:
bin walls (inner and outer), tray edges, table surface, fixtures.

A gap is free space to the planner. Walk the viewport camera around the bin and look
through the wall voxels — if you can see through, so can the planner.

Fixes, in order of preference: reposition or add a camera to cover the blind spot;
coarsen the voxel so neighbouring samples merge across the gap; or, if neither works,
add an analytic primitive for that surface as a backstop and report it.

**Check 5 — obstacles land correctly.** Enable debug prim visualisation on the world
interface. Obstacle boxes must coincide with the debug points from Check 3. Points
right but boxes wrong ⇒ robot base transform problem.

**Check 6 — start state collision-free.** Query the current joint configuration
against the populated world. Must report clean. If not, nothing downstream works and
no planner tuning will fix it.

**Check 7 — plan is non-degenerate.**

```python
# PURE MATH — safe to use as written
q = traj.positions                      # (T, dof)
print("T =", q.shape[0],
      "max joint delta =", float(np.abs(q[-1] - q[0]).max()),
      "max step delta  =", float(np.abs(np.diff(q, axis=0)).max()))
```

A success flag with `max joint delta ≈ 0` is a failure wearing a success flag.

---

## 10. Build order

Ordered by fastest failure, not data flow.

1. **One-box test — before anything else.** Push a single 20 cm cuboid in free space
   through the world pipeline. Confirm the robot still plans and moves. **If it does
   not, the problem is the world mechanism, not the cloud, and this entire spec is
   premature. Stop and report.**
2. Label the robot → Check 1.
3. **One camera only.** Get Checks 2–4 passing on a single corner camera.
4. Add the remaining six → re-run Check 3. A bad pose on one camera is far easier to
   spot before all clouds are merged.
5. Obstacle pool → Checks 5–6.
6. Connect the planner → Check 7.

---

## 11. API verification

Run these before writing any API-touching code. Report what you find.

```python
# semantics helper
import isaacsim.core.utils.semantics as sem
print([m for m in dir(sem) if not m.startswith("_")])

# available annotators
import omni.replicator.core as rep
print(rep.AnnotatorRegistry.get_registered_annotators())

# cuMotion surface
import isaacsim.robot_motion.cumotion as cm
print([m for m in dir(cm) if not m.startswith("_")])

from isaacsim.robot_motion.cumotion import CumotionWorldInterface
print([m for m in dir(CumotionWorldInterface) if not m.startswith("_")])
help(CumotionWorldInterface.__init__)      # look for obstacle cache capacity

# camera accessors — confirm intrinsics and pose method names + quaternion order
from isaacsim.sensors.camera import Camera
print([m for m in dir(Camera) if not m.startswith("_")])
```

**Authoritative reference for this build:** the `isaacsim.robot_motion.cumotion.examples`
extension. Enable via **Window > Extensions** (clear `@feature` from the search bar if
it does not appear) and read `world_interface/scenario.py`. Where that file and this
spec disagree, **the example file wins** — report the discrepancy.

---

## 12. Standing caveat

NVIDIA's position is that collision-free planning from camera-perceived worlds is an
open research problem: it works well in sparse environments, but as obstacle density
rises, occlusions cause many failures. A cell with a gantry, a vagn, a bin and a
kitting tray is not sparse.

**In simulation you have ground truth.** Adding physics colliders to the bin, tray and
fixtures and letting `SceneQuery` + `WorldBinding` discover them gives correct, cheap,
occlusion-free obstacles with none of this complexity.

Build this perception pipeline only to prototype what will run on real hardware — and
even then, get the ground-truth path working first, so there is a known-good baseline
to compare against.

---

## 13. Open items for the implementer to resolve

Report on each:

1. Does the build support obstacle property updates? (§6 load-bearing assumption)
2. Route A or Route B for obstacle integration? (§6)
3. Did instancing block semantics, and how was it resolved? (§3)
4. Measured pool size and per-update cost. (§6)
5. Obstacle cache capacity parameter name and value set. (§6)
6. Are `vagn` and `gantry` collision spheres still missing from the XRDF? (§5C)
7. Is the second gantry axis a planning DOF, or externally driven? This changes
   whether the robot base transform must be updated per frame. (§6)
8. **Measured voxel count for the full retained scene**, and the voxel size needed to
   fit it inside the achievable `N_MAX`. If these cannot be reconciled, stop and
   report — the full-scene-in-cloud approach may not be viable at this camera count
   and resolution. (§5)
9. Thinnest real obstacle in the cell, in metres. Sets the voxel size ceiling. (§5)
10. Any critical surface with occlusion holes that camera placement cannot close.
    (Check 4b)
