import asyncio
import os
import numpy as np
import omni.usd
import omni.kit.app
import omni.replicator.core as rep

# Updated to match your log's exact camera paths
CORNER_CAMERAS = {
    "/World/pointcloud_view/cam1": (640, 480),
    "/World/pointcloud_view/cam2": (640, 480),
    "/World/pointcloud_view/cam3": (640, 480),
    "/World/pointcloud_view/cam4": (640, 480),
}

OUT_PATH = "c:/KP/AI_and_Automation/Sem_4/Thesis/robot_in_air/generative_kitting/logs/native_merged_pointcloud.ply"
VOXEL_SIZE = 0.01  # 1 cm grid

def save_ply(filename, points):
    """Saves Nx3 NumPy array as a standard ASCII .ply file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.5f %.5f %.5f")

def voxel_downsample(points, voxel_size):
    """Pure NumPy voxel grid downsampling."""
    if len(points) == 0:
        return points
    grid_coords = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
    return points[unique_indices]

async def main():
    stage = omni.usd.get_context().get_stage()
    annotators = {}
    
    print("Initializing Direct Replicator Pipeline...")

    for cam_path, res in CORNER_CAMERAS.items():
        cam_prim = stage.GetPrimAtPath(cam_path)
        if not cam_prim.IsValid():
            print(f"!! Prim path not found: {cam_path}")
            continue
            
        # 1. Create a Replicator Render Product directly
        rp = rep.create.render_product(cam_path, res)
        
        # 2. Create a custom annotator that forces unlabelled geometry to render
        annotator = rep.AnnotatorRegistry.get_annotator(
            "pointcloud", 
            init_params={"includeUnlabelled": True}
        )
        
        # 3. Attach it manually
        annotator.attach(rp)
        annotators[cam_path] = annotator

    print("Stepping Replicator Orchestrator to generate data...")
    
    # Force the Replicator graph to evaluate (this guarantees the data is pushed)
    for _ in range(5):
        await rep.orchestrator.step_async()

    all_world_points = []

    for cam_path, annotator in annotators.items():
        # Fetch the data dict from the Replicator annotator
        pc_data = annotator.get_data()
        
        if pc_data is None or "data" not in pc_data or len(pc_data["data"]) == 0:
            print(f"[{cam_path}] Point cloud buffer empty.")
            continue
            
        # Extract the array and reshape to Nx3
        pts = pc_data["data"].reshape(-1, 3)
        
        # Filter out invalid / skybox points (origin [0,0,0])
        valid_mask = np.linalg.norm(pts, axis=1) > 0.01
        world_pts = pts[valid_mask]
        
        if len(world_pts) > 0:
            print(f"[{cam_path}] Captured {len(world_pts)} valid points.")
            all_world_points.append(world_pts)
        else:
            print(f"[{cam_path}] Points captured but all were invalid (skybox).")

    if not all_world_points:
        print("\n!! No valid point cloud data gathered.")
        return

    # Stack all points from the 4 cameras together
    merged_pts = np.vstack(all_world_points)

    # Downsample points into a lightweight 1cm voxel grid
    downsampled = voxel_downsample(merged_pts, VOXEL_SIZE)
    
    save_ply(OUT_PATH, downsampled)
    print(f"\nSuccessfully saved {len(downsampled)} native Replicator points to: {OUT_PATH}")

asyncio.ensure_future(main())