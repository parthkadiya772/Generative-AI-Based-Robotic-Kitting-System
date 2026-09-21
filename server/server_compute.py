import zmq
import json
import numpy as np
import torch
import curobo
import nvblox_torch

# Bind ZeroMQ REP socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

print("[Compute Server] Listening on port 5555 (cuRobo & nvblox ready)...")

while True:
    # Multipart receive: Header (JSON) + Payload (raw float32 depth buffer)
    frames = socket.recv_multipart()
    metadata = json.loads(frames[0].decode("utf-8"))
    depth_raw = frames[1]

    # Ingest directly into CUDA tensor
    depth_np = np.frombuffer(depth_raw, dtype=np.float32).reshape(metadata["shape"])
    depth_tensor = torch.from_numpy(depth_np).cuda()

    # Compute GPU test statistics
    valid_mask = torch.isfinite(depth_tensor) & (depth_tensor > 0.0)
    mean_depth = float(depth_tensor[valid_mask].mean().item()) if valid_mask.any() else 0.0

    # Send processing confirmation back to Windows
    response = {
        "status": "success",
        "frame_id": metadata["frame_id"],
        "device": torch.cuda.get_device_name(0),
        "mean_depth_m": round(mean_depth, 3)
    }
    socket.send_json(response)