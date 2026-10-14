"""nvblox collision-world service (Step 7).

Runs in its own venv (nvblox_torch links libtorch 2.9.1, Isaac Sim ships
2.11), so the Isaac Sim bridge talks to it over local HTTP:

    POST /integrate   .npz body: per camera depth_<i> (H,W float32 m, 0 = invalid),
                      mask_<i> (H,W uint8, 0 = ignore), K_<i> (3,3), T_w_c_<i> (4,4,
                      optical frame x right / y down / z forward -> world).
                      Integrates into the existing map (clear=0, default; decayed
                      first when tsdf_decay_factor is set) or rebuilds it (clear=1),
                      then updates the ESDF.
    GET  /sdf         .npz: dense signed distance over the workspace box for cuMotion
    POST /sdf         same; JSON {"carve": [[x, y, z_min, z_max, radius], ...]} marks those
                      vertical columns free first (grasp targets):
                      values (nx,ny,nz float32 m, z fastest, value i at voxel centre
                      workspace_min + (i + 0.5) * voxel_size), origin, voxel_size.
    POST /query       {"points": [[x, y, z], ...]} -> ESDF distance [m] per point
    POST /clear
    GET  /status      voxel size, frames integrated, GPU memory of this process

Settings come from robot.yaml (nvblox section), same file the bridge reads.
usage: generative_kitting/nvblox_service/run.sh generative_kitting/nvblox_service/server.py
"""

import io
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
import torch
from scipy import ndimage
from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.mapper_params import (EsdfIntegratorParams, MapperParams,
                                        ProjectiveIntegratorParams, TsdfDecayIntegratorParams,
                                        ViewCalculatorParams)
from nvblox_torch.sensor import Sensor

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
from generative_kitting.utils.robot_config import load_robot_config  # noqa: E402

CFG = load_robot_config().nvblox
if CFG is None:
    sys.exit("robot.yaml has no nvblox section")


def _make_mapper():
    params = MapperParams()
    proj = ProjectiveIntegratorParams()
    proj.projective_integrator_max_integration_distance_m = float(CFG["max_integration_distance"])
    params.set_projective_integrator_params(proj)
    # Allocate blocks only inside the workspace box. Unbounded, every ray from
    # the 3.7 m corner cameras allocates free space above the cell too.
    lo, hi = CFG["workspace_min"], CFG["workspace_max"]
    view = ViewCalculatorParams()
    view.workspace_bounds_type = "kBoundingBox"
    view.workspace_bounds_min_corner_x_m, view.workspace_bounds_min_corner_y_m = lo[0], lo[1]
    view.workspace_bounds_max_corner_x_m, view.workspace_bounds_max_corner_y_m = hi[0], hi[1]
    view.workspace_bounds_min_height_m, view.workspace_bounds_max_height_m = lo[2], hi[2]
    params.set_view_calculator_params(view)
    esdf = EsdfIntegratorParams()
    esdf.esdf_integrator_max_site_distance_vox = float(CFG["esdf_max_site_distance_vox"])
    params.set_esdf_integrator_params(esdf)
    if CFG.get("tsdf_decay_factor") is not None:
        decay = TsdfDecayIntegratorParams()
        decay.tsdf_decay_factor = float(CFG["tsdf_decay_factor"])
        params.set_tsdf_decay_integrator_params(decay)
    return Mapper(voxel_sizes_m=float(CFG["voxel_size"]), mapper_parameters=params)


MAPPER = _make_mapper()
STATS = {"frames": 0, "updates": 0, "last_update": None, "last_sdf": None}


def _gpu_mib():
    """GPU memory used by this process [MiB], as nvidia-smi reports it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            pid, mib = (v.strip() for v in line.split(","))
            if int(pid) == os.getpid():
                return int(mib)
    except Exception:
        pass
    return None


def integrate(frames, clear):
    t0 = time.perf_counter()
    # clear() does not hand blocks back to the pool, which then doubles
    # (16k -> 131k blocks over 8 rebuilds, 2026-10-14); incremental is the default.
    if clear:
        MAPPER.clear()
    elif CFG.get("tsdf_decay_factor") is not None:
        MAPPER.decay()
    n_valid = []
    for f in frames:
        depth = torch.from_numpy(f["depth"].astype(np.float32)).cuda()
        mask = torch.from_numpy(f["mask"].astype(np.uint8)).cuda()
        K = f["K"]
        h, w = f["depth"].shape
        sensor = Sensor.from_camera(fu=float(K[0, 0]), fv=float(K[1, 1]),
                                    cu=float(K[0, 2]), cv=float(K[1, 2]), width=w, height=h)
        pose = torch.from_numpy(f["T_w_c"].astype(np.float32))
        MAPPER.add_depth_frame(depth, pose, sensor, mask_frame=mask)
        n_valid.append(int(((f["depth"] > 0) & (f["mask"] > 0)).sum()))
    t1 = time.perf_counter()
    MAPPER.update_esdf()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    STATS["frames"] += len(frames)
    STATS["updates"] += 1
    STATS["last_update"] = {
        "cameras": len(frames), "valid_pixels": n_valid, "cleared": clear,
        "integrate_ms": round(1e3 * (t1 - t0), 1), "esdf_ms": round(1e3 * (t2 - t1), 1),
        "tsdf_blocks": MAPPER.tsdf_layer_view().num_blocks(),
        "pool_blocks": MAPPER.tsdf_layer_view().num_allocated_blocks(),
        "gpu_mib": _gpu_mib(),
    }
    return STATS["last_update"]


def query(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    # ESDF queries are spheres [x, y, z, radius]; radius 0 gives the plain distance.
    spheres = torch.from_numpy(np.hstack([pts, np.zeros((len(pts), 1), np.float32)])).cuda()
    d = MAPPER.query_layer(QueryType.ESDF, spheres).squeeze(-1)
    return d.cpu().numpy().tolist()


def dense_sdf(carve=()):
    """Signed distance over the workspace box: exact EDT of the TSDF occupancy.

    nvblox's own ESDF jumps by up to 0.27 m between neighbouring voxels
    (2-3 % of pairs, 2026-10-14), which cuMotion's inspect_sdf flags. The
    TSDF is reliable, so occupied = observed (weight > 0) and on or behind a
    surface (distance <= 0); the EDT of that is 1-Lipschitz by construction.
    Unobserved voxels count as free, like the ESDF query. Measured against
    the integrated points: median -0.4 to -0.7 cm (reads closer), p90 +0.6 cm.
    ``carve`` columns [x, y, z_min, z_max, radius] are set free before the
    EDT, so the field stays exact around them.
    """
    t0 = time.perf_counter()
    vs = float(CFG["voxel_size"])
    lo, hi = np.asarray(CFG["workspace_min"], float), np.asarray(CFG["workspace_max"], float)
    n = np.round((hi - lo) / vs).astype(int)
    axes = [torch.arange(n[k], device="cuda", dtype=torch.float32) * vs + float(lo[k]) + vs / 2
            for k in range(3)]
    X, Y, Z = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
    tsdf = MAPPER.query_layer(QueryType.TSDF, pts).reshape(*n, 2).cpu().numpy()
    del X, Y, Z, pts
    occ = (tsdf[..., 1] > 0) & (tsdf[..., 0] <= 0)
    n_carved = 0
    centres = [lo[k] + (np.arange(n[k]) + 0.5) * vs for k in range(3)]
    for x, y, z0, z1, r in carve:
        ix = np.flatnonzero(np.abs(centres[0] - x) <= r)
        iy = np.flatnonzero(np.abs(centres[1] - y) <= r)
        iz = np.flatnonzero((centres[2] >= z0) & (centres[2] <= z1))
        if not (len(ix) and len(iy) and len(iz)):
            continue
        disc = np.hypot(*np.meshgrid(centres[0][ix] - x, centres[1][iy] - y, indexing="ij")) <= r
        block = occ[ix[0]:ix[-1] + 1, iy[0]:iy[-1] + 1, iz[0]:iz[-1] + 1]
        n_carved += int(block[disc].sum())
        block[disc] = False
    # voxel-centre distances, surface half a voxel between occupied and free
    outside = ndimage.distance_transform_edt(~occ, sampling=vs)
    inside = ndimage.distance_transform_edt(occ, sampling=vs)
    values = np.where(occ, vs / 2 - inside, outside - vs / 2).astype(np.float32)
    STATS["last_sdf"] = {"shape": n.tolist(), "occupied": int(occ.sum()), "carved": n_carved,
                         "unobserved": int((tsdf[..., 1] <= 0).sum()),
                         "ms": round(1e3 * (time.perf_counter() - t0), 1)}
    return values, lo, vs


def _unpack_npz(body):
    data = np.load(io.BytesIO(body))
    n = int(data["num_cameras"])
    frames = [{k: data[f"{k}_{i}"] for k in ("depth", "mask", "K", "T_w_c")} for i in range(n)]
    return frames, bool(data["clear"]) if "clear" in data else False


class Handler(BaseHTTPRequestHandler):
    def _send(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        if urlparse(self.path).path == "/status":
            self._send({"status": "ok", "voxel_size": CFG["voxel_size"],
                        "max_integration_distance": CFG["max_integration_distance"],
                        "gpu_mib": _gpu_mib(), **STATS})
        elif urlparse(self.path).path == "/sdf":
            self._send_sdf(())
        else:
            self._send({"error": f"unknown endpoint {self.path}"}, 404)

    def _send_sdf(self, carve):
        try:
            values, origin, vs = dense_sdf(carve)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)
            return
        buf = io.BytesIO()
        np.savez(buf, values=values, origin=origin, voxel_size=vs)
        body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Sdf-Stats", json.dumps(STATS["last_sdf"]))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/sdf":
            self._send_sdf(json.loads(self._body() or b"{}").get("carve", []))
            return
        try:
            if path == "/integrate":
                frames, clear = _unpack_npz(self._body())
                self._send({"status": "ok", **integrate(frames, clear)})
            elif path == "/query":
                pts = json.loads(self._body())["points"]
                self._send({"status": "ok", "distances": query(pts)})
            elif path == "/clear":
                MAPPER.clear()
                self._send({"status": "ok"})
            else:
                self._send({"error": f"unknown endpoint {path}"}, 404)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, fmt, *args):
        pass


def main():
    url = urlparse(CFG["url"])
    server = HTTPServer((url.hostname, url.port), Handler)
    print(f"[nvblox] voxel {CFG['voxel_size']} m, listening on {CFG['url']}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
