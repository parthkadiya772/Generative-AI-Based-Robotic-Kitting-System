# Hybrid Workflow Architecture: Windows Isaac Sim & Docker Compute Backend

## 1. Architectural Overview
This project uses a hybrid split-runtime architecture. It separates the heavy graphics rendering (handled natively on Windows) from the complex CUDA-based robotic AI libraries (handled in a Linux Docker container).

* **Simulation & Graphics (Windows Host):** Runs Isaac Sim 6.0.1 natively. This allows direct access to the RTX GPU's bare-metal Vulkan drivers for hardware ray-tracing and physics simulation without the WSL2 virtualization bottlenecks.
* **Compute & Planning (Docker Container via WSL2):** Runs a custom Linux-based environment containing PyTorch, `cuRobo` (for trajectory optimization), and `nvblox` (for 3D voxel reconstruction). 

## 2. Component Structure & Volume Mounting
The project code is developed on Windows but mapped into the Docker container, allowing live edits without rebuilding the image.

* **Project Root (`robot_in_air/`):**
  * `docker-compose.yml`: Configures the container setup, port mappings, and volume mounts.
  * `server/server_compute.py`: The Python compute backend. It is stored on the Windows host but executed inside the Docker container.
* **Volume Mount:** The `docker-compose.yml` mounts the current Windows directory (`./`) directly to `/workspace` inside the container. This means the `server` folder and its scripts are automatically accessible to the Docker environment.

## 3. Network & Port Configuration
Because the simulation and the compute backend are running in separate processes, they communicate over a local network bridge.

* **Port Mapping:** The `docker-compose.yml` explicitly binds port `5555:5555/tcp`. This exposes the Docker container's internal ZeroMQ server to the Windows localhost.
* **ZeroMQ (TCP Sockets):** Communication is handled via synchronous TCP sockets using the `pyzmq` library. 
  * **Docker acts as the Server (REP):** Listens on `tcp://0.0.0.0:5555`.
  * **Windows acts as the Client (REQ):** Connects to `tcp://127.0.0.1:5555`.

## 4. Execution Workflow
The pipeline operates in a continuous loop exchanging data between the two environments:

1. **Initialization:** The Docker container is launched (`docker compose up -d`), and `server_compute.py` is started inside it.
2. **Data Capture (Windows):** The Isaac Sim script extracts data from the existing scene (e.g., depth arrays from the camera, joint states from the robot articulation).
3. **Transmission (Windows -> Docker):** Windows sends the sensor data over port 5555. To avoid heavy serialization overhead, this is sent as raw binary buffers alongside metadata.
4. **Processing (Docker):** `server_compute.py` receives the data, feeds the depth into `nvblox`, and uses `cuRobo` to generate collision-free motion plans.
5. **Feedback (Docker -> Windows):** The resulting trajectory data is sent back over the socket to the Windows script, which applies the movement to the robot articulation in the simulation.