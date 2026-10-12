# Pipeline updates pending — verified findings

Findings verified empirically against the live Isaac Sim 6.0.1 scene during the
cuMotion / self-filtering debugging sessions. **These are not yet applied to the
pipeline.** Apply them after the robot-movement debugging is finished.

Every entry here was confirmed by direct test against the running sim, not inferred
from documentation or API signatures. Where something is still unverified, it says so
explicitly.

---

## 1. `/World/rsd555/Depth` is a depth-only sensor

**The finding.** The RealSense D555 depth prim is a *depth sensor*, not a camera. It
has no colour imager, so it has no RGB output of any kind. The only annotators valid
on it are depth annotators:

| Annotator | Use |
|---|---|
| `distance_to_image_plane` | **Correct.** Planar Z-depth along the optical axis — what pinhole deprojection expects. |
| `distance_to_camera` | Radial range, **not** planar. Deprojecting this with a pinhole model bends the scene. Do not use for point clouds. |

**Two false alarms this caused.** Both were chased at length before the cause was
understood. Neither was ever a real bug:

1. **"The depth prim renders solid black in the viewport."** Expected. Looking through
   a camera prim renders the ordinary RGB beauty pass; a depth-only sensor has no such
   pass to render. Ruled out along the way, and *not* the cause: the
   `OmniLensDistortionOpenCvPinholeAPI` schema on the prim, its `clippingRange`, its
   resolution, and a missing `cameraProjectionType` attribute. Do not re-investigate
   these.

2. **`[depth] WARNING: frame still dark (mean=0.0)`** in the pick-and-place logs. Also
   expected, for the same reason — see §2 for exactly why it fires.

> **Consequence for earlier debugging:** this warning was previously treated as a
> possible cause of a failed grasp (gripper closed on empty air, no contact detected).
> That link is now doubtful — the warning is a false positive and says nothing about
> whether depth data was valid. **The grasp miss needs re-diagnosing on its own terms.**

---

## 2. `_render_frame` / `_get_camera` apply RGB logic to the depth-only sensor

`generative_kitting/isaac_sim_bridge.py`:

```python
CAMERA_ANNOTATORS = ["rgb", "distance_to_image_plane"]   # applied to ALL aliases
```

This single list is attached to every alias in `CAMERA_ALIASES`, including
`"depth"` → `/World/rsd555/Depth`. Three concrete consequences:

**a. The warm-up loop never exits early.** `_get_camera` breaks out only when *both*
`rgb` and `distance_to_image_plane` read non-`None`. On the depth-only sensor `rgb`
never becomes valid, so the loop always runs the full `_WARMUP_MAX_TICKS = 400` ticks
before giving up. Every depth-camera initialisation pays 400 ticks it does not need.

**b. The dark-frame warning fires every time.** `_render_frame` requests `rgb`, and the
depth-only sensor returns an all-zeros array rather than `None` — so the `rgb is None`
guard does not catch it, and the `mean <= 3` branch prints the warning instead.

**c. `ValueError: The annotator 'rgb' was not configured`** appeared in earlier logs
from this same mismatch.

**Fix to apply:** give the annotator list a per-alias override so the depth alias
requests only depth, and make both the warm-up predicate and the dark-frame check
conditional on whether the alias actually has colour. Roughly:

```python
CAMERA_ANNOTATORS = ["rgb", "distance_to_image_plane"]
CAMERA_ANNOTATORS_BY_ALIAS = {"depth": ["distance_to_image_plane"]}   # depth-only sensor
```

...then have `_get_camera` select via `CAMERA_ANNOTATORS_BY_ALIAS.get(alias, CAMERA_ANNOTATORS)`,
gate the warm-up break on only the annotators actually requested, and skip the
dark-frame check entirely for aliases with no `"rgb"` in their list.

---

## 3. cuMotion obstacle API — what can and cannot change at runtime

Verified directly against the live `CumotionWorldInterface` via `/api/probe_obstacle_update`:

| Operation | Result |
|---|---|
| `add_spheres` (create) | works |
| `update_obstacle_transforms` (move) | works |
| `update_obstacle_enables` (toggle on/off) | works |
| `update_sphere_properties` (resize) | **`NotImplementedError`** — "not implemented by the planning world child class" |
| Re-adding an existing `prim_path` | **`ValueError`** — already exists; obstacles cannot be replaced or removed |

**Important trap:** every `update_*_properties` method *appears* in
`dir(CumotionWorldInterface)` with a full signature and docstring. They are inherited
abstract stubs. The class listing tells you nothing about whether the concrete object
implements them — only calling it does.

**Consequence:** obstacles can be created once, then only moved or enabled/disabled.
This confirms the fixed-obstacle-pool architecture described in
`robot_self_filtering_spec.md` §6 is required, not optional.

---

## 4. Semantic labelling API (Isaac Sim 6.0.1)

Real signature, confirmed by introspection:

```python
isaacsim.core.utils.semantics.add_labels(
    prim: Usd.Prim, labels: list[str], instance_name: str = "class", overwrite: bool = True)
```

Companion helpers that exist and are useful: `get_labels`, `remove_labels`,
`check_missing_labels(prim_path)`, `check_incorrect_labels`, `count_labels_in_scene`.

Labels must be applied to the `Gprim`/`Mesh` descendants, not the link `Xform`s.
Labelling the subtree root alone does nothing.

**No USD instancing in this scene.** Every prim under `/World/gantry` reports
`instance=False, instanceable=False, instanceProxy=False`, so the instancing hazard
described in the spec (§3) does not apply here.

---

## 5. Which prims move with the robot (measured, not assumed)

Measured by jogging `gantry_vagn_joint` by −0.3 m and comparing world transforms
before/after. This determines what must be treated as robot vs. as a real obstacle:

| Prim | Result | Treat as |
|---|---|---|
| `/World/gantry` | static | fixed structure |
| `/World/gantry/gantry_home` | static | fixed mounting bracket |
| `/World/gantry/node/gantry` (rail mesh) | static | **real obstacle — must stay in collision world** |
| `/World/gantry/gantry_home/vagn` | moved −0.3 | robot |
| `.../ur10_instanceable/*` (whole arm) | moved −0.3 | robot |
| `/World/rsd555` (wrist camera) | moved −0.3 | robot |

The UR10 is a *sibling* of `vagn` in the USD hierarchy, not a child — they are joined
by the `vagn_robot_joint` fixed physics joint, not by prim nesting. USD hierarchy does
not reflect the kinematic chain here; do not infer motion from parenting.

---

## 6. Replicator: unresolved semantic-label issue on the six point-cloud cameras

**Status: OPEN — not yet resolved.**

The `pointcloud` annotator already carries per-point semantic data of its own, which
would avoid needing a second annotator entirely:

```
keys: data (N,3) float32 | pointInstance (N,) uint32 | pointNormals (N,4)
      pointRgb (N,4) uint8 | pointSemantic (N,) uint32 | info
```

But on all six `POINTCLOUD_CAMERAS`, **every point returns the same semantic ID**
(`{1: 134982}` — one ID, zero differentiation), even with the robot confirmed visible
in cam1/cam2/cam4 by eye, and even after `add_labels` succeeded on 54 robot meshes.
A separately attached `semantic_segmentation` annotator on those same cameras also
reports only `BACKGROUND`/`UNLABELLED`, never `robot`.

The wrist camera — which no other code path has ever attached a Replicator render
product to — resolved `robot` correctly on the first try, at 31.9% of frame.

**Leading hypothesis (untested):** Kit caches per-prim-path semantic render-graph state
from the first render product created against a camera. Every "bridge restart" this
session only re-executed Python inside the *same* Kit process, so those six cameras may
be stuck with semantic state captured before `add_labels` ever ran. **The test is a full
cold restart of Isaac Sim itself, then labelling before anything renders those cameras.**

### Two Replicator gotchas confirmed along the way

- **`rep.orchestrator.step()` fails inside Kit** — `OrchestratorError: Synchronous call
  to step can only be performed in a standalone workflow`. Use `await
  rep.orchestrator.step_async()`. In a Script Editor snippet, wrap in `async def` and
  schedule with `asyncio.ensure_future(...)`; do not rely on top-level `await`.

- **Two render products on the same camera prim break segmentation.** Creating a second,
  independent render product for a camera that already has one yields an empty
  segmentation buffer (`shape (0,)`) every time, on every camera tested, regardless of
  tick count. `_pc_render_products` was added to `isaac_sim_bridge.py` to keep the
  render product handle so additional annotators can share it — which is also what
  `robot_self_filtering_spec.md` §2 requires anyway, for pixel alignment.

---

## 7. Script Editor namespace isolation

Separate `exec(open(...).read())` runs in Isaac Sim's Script Editor **do not reliably
share a Python namespace**. A script cannot count on reaching `STATE` or any other
global defined by a previously executed script, even though the bridge is alive and
serving.

**Practice to follow:** anything that needs the bridge's live in-process state should be
added as an HTTP endpoint on the bridge and called over `localhost:8600`, the way
`/api/joint_report`, `/api/probe_obstacle_update` and friends already work. Do not write
Script Editor scripts that reach into bridge internals.
