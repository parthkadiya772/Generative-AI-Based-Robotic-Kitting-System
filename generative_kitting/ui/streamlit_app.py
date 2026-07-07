"""
Streamlit Web UI for the Generative Kitting System.

Provides an operator dashboard with:
  - Live video stream from Isaac Sim camera with perception overlays
  - Selectable camera perception modes (Raw, VLM, Confidence, Affordance, Grid)
  - Auto-refreshing video feed with configurable FPS
  - Chat-style command interface
  - Real-time execution log
  - VLM detected objects table
  - Task plan viewer
  - Emergency stop button
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Add parent directory to path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
load_dotenv(os.path.join(_PROJECT_ROOT, "..", ".env"))

import yaml
from PIL import Image

from utils.logger import setup_logger, get_session_id, log
from perception.vlm_perception import VLMPerception
from perception.camera_interface import MockCameraInterface, BridgeCameraInterface
from perception.visualiser import (
    render_perception_overlay,
    PERCEPTION_MODES,
)
from orchestration.llm_planner import LLMPlanner
from orchestration.workflow_engine import (
    KittingWorkflowEngine, WorkflowPhase, SCENE_ANALYSIS_PROMPT,
    _with_catalogue,
)
from knowledge.parts_database import PartsDatabase
from knowledge.seed_data import seed_parts, seed_kits
from evaluation import EvaluationRecorder, PlaceVerifier


# ═════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Generative Kitting System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 100%;
    }

    /* ── Header ─────────────────────────────────── */
    .header-gradient {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .header-gradient h1 {
        color: #e0e7ff;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
    }

    .header-gradient p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0.3rem 0 0 0;
    }

    /* ── Cards ──────────────────────────────────── */
    .glass-card {
        background: rgba(30, 30, 50, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(100, 100, 200, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .glass-card h3 {
        color: #c7d2fe;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
    }

    /* ── Video Feed ─────────────────────────────── */
    .video-container {
        position: relative;
        border: 2px solid rgba(100, 100, 200, 0.2);
        border-radius: 10px;
        overflow: hidden;
    }

    .perception-badge {
        position: absolute;
        top: 8px;
        right: 8px;
        background: rgba(0, 0, 0, 0.7);
        color: #a5b4fc;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        z-index: 10;
    }

    .stream-status {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.75rem;
        margin-top: 6px;
    }

    .stream-live {
        background: rgba(22, 163, 74, 0.15);
        color: #22c55e;
        border: 1px solid rgba(22, 163, 74, 0.25);
    }

    .stream-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulse-dot 1.5s ease-in-out infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    /* ── Execution Log Items ───────────────────── */
    .log-success {
        background: rgba(22, 163, 74, 0.15);
        border-left: 3px solid #22c55e;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #86efac;
    }

    .log-fail {
        background: rgba(220, 38, 38, 0.15);
        border-left: 3px solid #ef4444;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #fca5a5;
    }

    .log-retry {
        background: rgba(234, 179, 8, 0.15);
        border-left: 3px solid #eab308;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #fde68a;
    }

    .log-info {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #93c5fd;
    }

    /* ── Status Badge ──────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-ready   { background: rgba(22,163,74,0.2); color: #22c55e; border: 1px solid rgba(22,163,74,0.3); }
    .status-busy    { background: rgba(234,179,8,0.2); color: #eab308; border: 1px solid rgba(234,179,8,0.3); }
    .status-error   { background: rgba(220,38,38,0.2); color: #ef4444; border: 1px solid rgba(220,38,38,0.3); }
    .status-offline { background: rgba(100,116,139,0.2); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3); }

    /* ── Buttons ───────────────────────────────── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* ── Chat Messages ─────────────────────────── */
    .chat-operator {
        background: linear-gradient(135deg, #1e3a5f, #2d4a7a);
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #e0e7ff;
    }

    .chat-system {
        background: linear-gradient(135deg, #1e1e3e, #2d2d5e);
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        color: #c7d2fe;
    }

    /* ── Perception Selector ───────────────────── */
    .perception-selector-label {
        color: #a5b4fc;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }

    /* ── Scrollable containers ─────────────────── */
    .scroll-container {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }

    .scroll-container::-webkit-scrollbar { width: 6px; }
    .scroll-container::-webkit-scrollbar-track { background: rgba(30,30,50,0.5); border-radius: 3px; }
    .scroll-container::-webkit-scrollbar-thumb { background: rgba(100,100,200,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═════════════════════════════════════════════════════════════

_BRIDGE_URL = (
    f"http://{os.environ.get('BRIDGE_HOST', '127.0.0.1')}"
    f":{os.environ.get('BRIDGE_PORT', '8600')}"
)
_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def init_session_state():
    """Initialise Streamlit session state variables."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml",
    )
    if "config" not in st.session_state:
        with open(config_path, "r") as f:
            raw = f.read()
        expanded = re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            raw,
        )
        st.session_state.config = yaml.safe_load(expanded)

    if "session_id" not in st.session_state:
        st.session_state.session_id = get_session_id()

    if "vlm" not in st.session_state:
        st.session_state.vlm = VLMPerception(
            st.session_state.config.get("perception", {})
        )

    if "planner" not in st.session_state:
        st.session_state.planner = LLMPlanner(
            st.session_state.config.get("orchestration", {})
        )

    if "camera" not in st.session_state:
        # Try connecting to Isaac Sim bridge first
        bridge = BridgeCameraInterface(_BRIDGE_URL)
        if bridge.is_available:
            st.session_state.camera = bridge
            st.session_state.bridge_connected = True
        else:
            st.session_state.camera = MockCameraInterface()
            st.session_state.bridge_connected = False

    if "db" not in st.session_state:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "knowledge", "kitting.db",
        )
        st.session_state.db = PartsDatabase(db_path)
        if not st.session_state.db.get_all_parts():
            seed_parts(st.session_state.db)
            seed_kits(st.session_state.db)

    # Evaluation recorder — persists per-event metrics to JSONL +
    # SQLite under logs/evaluation. Constructed once per session so
    # task IDs stay monotonic for the lifetime of the Streamlit run.
    if "evaluator" not in st.session_state:
        eval_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "evaluation",
        )
        st.session_state.evaluator = EvaluationRecorder(
            log_dir=eval_dir,
            session_id=st.session_state.session_id,
        )

    # Place verifier — Camera_Kit + VLM. Only useful in bridge mode.
    if "place_verifier" not in st.session_state:
        if st.session_state.get("bridge_connected", False):
            st.session_state.place_verifier = PlaceVerifier(
                camera_interface=st.session_state.camera,
                vlm=st.session_state.vlm,
            )
        else:
            st.session_state.place_verifier = None

    # Set initial status based on connection
    _initial_status = "ready" if st.session_state.get("bridge_connected", False) else "offline"

    defaults = {
        "chat_history": [],
        "scene_description": None,
        "task_plan": None,
        "execution_log": [],
        "system_status": _initial_status,
        "perception_mode": "🎯 VLM Detections",
        "auto_stream": False,
        "stream_fps": 1.0,
        "last_annotated_image": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ═════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-gradient">
    <h1>🤖 Generative AI-Based Kitting System</h1>
    <p>Neuro-Symbolic AI-based robotic kitting — VLM perception · LLM planning · Isaac Sim execution</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# SIDEBAR — Configuration & Controls
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Session info
    st.markdown(f"**Session:** `{st.session_state.session_id}`")

    # Status
    status = st.session_state.system_status
    badge_class = {
        "ready": "status-ready",
        "busy": "status-busy",
        "error": "status-error",
        "offline": "status-offline",
    }.get(status, "status-offline")
    st.markdown(
        f'<span class="status-badge {badge_class}">{status}</span>',
        unsafe_allow_html=True,
    )

    # Isaac Sim bridge connection indicator — live check every render
    _cam = st.session_state.get("camera")
    if isinstance(_cam, BridgeCameraInterface):
        _cam._check_connection()
        if _cam.is_available:
            st.session_state.bridge_connected = True
            # Restore ready status if it was offline (but not if in error/busy)
            if st.session_state.system_status == "offline":
                st.session_state.system_status = "ready"
        else:
            # Bridge went down — fall back to mock camera
            st.session_state.bridge_connected = False
            st.session_state.camera = MockCameraInterface()
            # Set offline unless user triggered emergency stop
            if st.session_state.system_status not in ("error", "busy"):
                st.session_state.system_status = "offline"
    else:
        # Using mock camera — not connected
        if not st.session_state.get("bridge_connected", False):
            if st.session_state.system_status not in ("error", "busy"):
                st.session_state.system_status = "offline"

    if st.session_state.get("bridge_connected", False):
        st.markdown('🟢 **Isaac Sim Connected**')
    else:
        st.markdown('🔴 **Isaac Sim Offline** (mock camera)')
        if st.button("🔌 Connect to Isaac Sim", key="connect_bridge"):
            bridge = BridgeCameraInterface(_BRIDGE_URL)
            if bridge.is_available:
                st.session_state.camera = bridge
                st.session_state.bridge_connected = True
                st.session_state.system_status = "ready"
                st.success("Connected to Isaac Sim!")
                st.rerun()
            else:
                st.error("Cannot reach Isaac Sim bridge on port 8600")

    st.divider()

    # ── Camera Perception Mode Selector ──────────────────────
    st.markdown("### 🎥 Camera Perception")

    st.session_state.perception_mode = st.selectbox(
        "Perception Overlay",
        options=list(PERCEPTION_MODES.keys()),
        index=list(PERCEPTION_MODES.keys()).index(st.session_state.perception_mode)
            if st.session_state.perception_mode in PERCEPTION_MODES else 1,
        key="perception_mode_select",
        help="Select the visual overlay mode for the camera feed",
    )

    # Perception mode descriptions
    mode_descriptions = {
        "📷 Raw Feed":         "Unprocessed camera image — no annotations",
        "🎯 VLM Detections":   "Bounding boxes, crosshairs, labels & confidence scores",
        "🔥 Confidence Map":   "Colour-coded circles: Red → Low, Green → High confidence",
        "🏷️ Affordance View":  "Shapes colour-coded by affordance: graspable / fragile / stackable",
        "📐 Scene Grid":       "Workspace coordinate grid with object position markers",
    }
    st.caption(mode_descriptions.get(
        st.session_state.perception_mode, ""
    ))

    st.divider()

    # ── Video Stream Controls ────────────────────────────────
    st.markdown("### 📡 Stream Controls")

    st.session_state.auto_stream = st.toggle(
        "Auto-Refresh Stream",
        value=st.session_state.auto_stream,
        key="auto_stream_toggle",
        help="Automatically refresh the camera feed at the configured FPS",
    )

    st.session_state.stream_fps = st.slider(
        "Refresh Rate (FPS)",
        min_value=0.2,
        max_value=5.0,
        value=st.session_state.stream_fps,
        step=0.2,
        key="stream_fps_slider",
        help="Camera feed refresh rate (higher = more responsive, more CPU)",
    )

    st.divider()

    # ── AI Model Selection (from config list) ────────────────
    st.markdown("### 🧠 AI Models")

    # ── VLM Model Selector ───────────────────────────────────
    vlm_models_list = st.session_state.config.get("perception", {}).get("vlm_models", [])
    if not vlm_models_list:
        # Fallback: build a single entry from flat config
        vlm_models_list = [{
            "name": st.session_state.config.get("perception", {}).get("vlm_model", "qwen2.5-vl:7b"),
            "provider": st.session_state.config.get("perception", {}).get("vlm_provider", "ollama_qwen"),
            "model": st.session_state.config.get("perception", {}).get("vlm_model", "qwen2.5-vl:7b"),
            "base_url": st.session_state.config.get("perception", {}).get("vlm_base_url", _OLLAMA_URL),
        }]

    vlm_names = [m["name"] for m in vlm_models_list]
    vlm_selected_name = st.selectbox(
        "👁️ Vision Model (VLM)",
        options=vlm_names,
        index=0,
        key="vlm_model_select",
        help="Select the Vision-Language Model for object detection",
    )

    # Look up the selected VLM config
    vlm_selected = next(
        (m for m in vlm_models_list if m["name"] == vlm_selected_name),
        vlm_models_list[0],
    )
    st.caption(
        f"📡 `{vlm_selected['provider']}` · "
        f"`{vlm_selected['model']}` · "
        f"`{vlm_selected['base_url']}`"
    )

    st.markdown("---")

    # ── LLM Model Selector ───────────────────────────────────
    llm_models_list = st.session_state.config.get("orchestration", {}).get("llm_models", [])
    if not llm_models_list:
        llm_models_list = [{
            "name": st.session_state.config.get("orchestration", {}).get("llm_model", "llama3.1:8b"),
            "provider": st.session_state.config.get("orchestration", {}).get("llm_provider", "ollama_llama"),
            "model": st.session_state.config.get("orchestration", {}).get("llm_model", "llama3.1:8b"),
            "base_url": st.session_state.config.get("orchestration", {}).get("llm_base_url", _OLLAMA_URL),
        }]

    llm_names = [m["name"] for m in llm_models_list]
    llm_selected_name = st.selectbox(
        "🧠 Planner Model (LLM)",
        options=llm_names,
        index=0,
        key="llm_model_select",
        help="Select the Large Language Model for task planning",
    )

    # Look up the selected LLM config
    llm_selected = next(
        (m for m in llm_models_list if m["name"] == llm_selected_name),
        llm_models_list[0],
    )
    st.caption(
        f"📡 `{llm_selected['provider']}` · "
        f"`{llm_selected['model']}` · "
        f"`{llm_selected['base_url']}`"
    )

    # ── Apply Model Selection ────────────────────────────────
    if st.button("🔄 Apply Models", width='stretch', key="apply_settings"):
        # Update VLM config from selection
        st.session_state.config["perception"]["vlm_provider"] = vlm_selected["provider"]
        st.session_state.config["perception"]["vlm_model"] = vlm_selected["model"]
        st.session_state.config["perception"]["vlm_base_url"] = vlm_selected["base_url"]

        # Update LLM config from selection
        st.session_state.config["orchestration"]["llm_provider"] = llm_selected["provider"]
        st.session_state.config["orchestration"]["llm_model"] = llm_selected["model"]
        st.session_state.config["orchestration"]["llm_base_url"] = llm_selected["base_url"]

        # Reinitialise models with new config
        st.session_state.vlm = VLMPerception(st.session_state.config["perception"])
        st.session_state.planner = LLMPlanner(st.session_state.config["orchestration"])

        st.success(
            f"✅ VLM → {vlm_selected['name']}\n\n"
            f"✅ LLM → {llm_selected['name']}"
        )

        st.session_state.execution_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": "Model Switch",
            "status": f"VLM={vlm_selected['model']}, LLM={llm_selected['model']}",
            "type": "info",
        })

    st.divider()

    # ── Emergency Stop ───────────────────────────────────────
    st.markdown("### 🚨 Safety")
    if st.button(
        "🛑 EMERGENCY STOP",
        width='stretch',
        type="primary",
        key="estop",
    ):
        st.session_state.system_status = "error"
        st.session_state.auto_stream = False
        st.session_state.execution_log.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": "EMERGENCY_STOP",
            "status": "executed",
            "type": "fail",
        })
        st.warning("Emergency stop triggered!")

    # ── Reset (clears emergency state) ────────────────────────
    if st.session_state.system_status == "error":
        if st.button(
            "🔄 Reset System",
            width='stretch',
            key="reset_estop",
        ):
            # Send robot home if bridge is connected
            _cam_reset = st.session_state.get("camera")
            if isinstance(_cam_reset, BridgeCameraInterface) and _cam_reset.is_available:
                try:
                    _cam_reset.send_home()
                except Exception:
                    pass

            # Clear error state
            is_connected = st.session_state.get("bridge_connected", False)
            st.session_state.system_status = "ready" if is_connected else "offline"
            st.session_state.execution_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "action": "SYSTEM_RESET",
                "status": "emergency state cleared",
                "type": "success",
            })
            st.success("System reset — ready for commands")
            st.rerun()

    st.divider()

    # ── Parts Database ───────────────────────────────────────
    st.markdown("### 📦 Parts Database")
    parts = st.session_state.db.get_all_parts()
    st.metric("Total Parts", len(parts))
    kits = st.session_state.db.get_all_kits()
    st.metric("Kit Definitions", len(kits))

    st.divider()

    # ── Hot Reload ───────────────────────────────────────────
    st.markdown("### 🔧 Developer Tools")
    if st.button("♻️ Hot Reload Backend", width='stretch', key="hot_reload"):
        # Clear cached imports so Python reloads changed modules
        import importlib
        modules_to_reload = [
            m for m in sys.modules
            if m.startswith(("perception", "orchestration", "simulation", "knowledge", "utils"))
        ]
        for mod_name in modules_to_reload:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass

        # Re-init VLM and planner with current config
        from perception.vlm_perception import VLMPerception
        from orchestration.llm_planner import LLMPlanner
        st.session_state.vlm = VLMPerception(
            st.session_state.config.get("perception", {})
        )
        st.session_state.planner = LLMPlanner(
            st.session_state.config.get("orchestration", {})
        )
        st.success("Backend modules reloaded!")
        st.rerun()


# ═════════════════════════════════════════════════════════════
# HELPER: Capture + Annotate Image
# ═════════════════════════════════════════════════════════════

def capture_and_annotate():
    """
    Capture a camera image and apply the selected perception overlay.
    Returns the annotated PIL Image.
    """
    raw_image = st.session_state.camera.capture_workspace_image()
    mode_key = PERCEPTION_MODES.get(st.session_state.perception_mode, "raw")

    detected_objects = []
    if st.session_state.scene_description:
        detected_objects = st.session_state.scene_description.get("detected_objects", [])

    annotated = render_perception_overlay(raw_image, detected_objects, mode=mode_key)
    st.session_state.last_annotated_image = annotated
    return annotated


# ═════════════════════════════════════════════════════════════
# MAIN LAYOUT — Three Column
# ═════════════════════════════════════════════════════════════

col_left, col_center, col_right = st.columns([1.3, 1.7, 1])


# ─── Left Column: Video Stream + Perception ─────────────────
with col_left:
    # ── Dual Camera Display (Tabs) ───────────────────────────
    st.markdown(
        '<div class="glass-card"><h3>📹 Camera Vision — Dual Perception</h3>',
        unsafe_allow_html=True,
    )

    # Controls row
    stream_col1, stream_col2, stream_col3 = st.columns([1, 1, 1])
    with stream_col1:
        scan_btn = st.button(
            "🔍 Analyze Scene",
            width='stretch',
            key="scan_btn",
        )
    with stream_col2:
        depth_btn = st.button(
            "🔬 Depth Analysis",
            width='stretch',
            key="depth_btn",
        )
    with stream_col3:
        refresh_btn = st.button(
            "🔄 Refresh",
            width='stretch',
            key="refresh_btn",
        )

    # Run VLM scene analysis (Phase 1: overhead RGB)
    if scan_btn:
        st.session_state.system_status = "busy"
        try:
            with st.spinner("🧠 Phase 1: Overhead VLM analysis..."):
                image = st.session_state.camera.capture_workspace_image()
                scene = st.session_state.vlm.analyze_scene(
                    image,
                    custom_prompt=_with_catalogue(SCENE_ANALYSIS_PROMPT))
                st.session_state.scene_description = scene
                st.session_state.system_status = "ready"

                st.session_state.execution_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "action": "VLM Scene Analysis (RGB)",
                    "status": f"{len(scene['detected_objects'])} objects detected",
                    "type": "success",
                })
        except Exception as e:
            st.session_state.system_status = "error"
            st.error(f"Perception failed: {e}")
            st.session_state.execution_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "action": "VLM Scene Analysis (RGB)",
                "status": str(e),
                "type": "fail",
            })

    # Run Depth analysis (Phase 2: arm-mounted RealSense)
    if depth_btn:
        st.session_state.system_status = "busy"
        try:
            with st.spinner("🔬 Phase 2: Depth spatial analysis..."):
                is_bridge = st.session_state.get("bridge_connected", False)
                if is_bridge and hasattr(st.session_state.camera, "capture_depth_image"):
                    depth_image = st.session_state.camera.capture_depth_image()
                    st.session_state.depth_image = depth_image

                    # Send depth image to VLM for spatial reasoning
                    depth_scene = st.session_state.vlm.analyze_scene(depth_image)
                    st.session_state.depth_description = depth_scene

                    st.session_state.execution_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "action": "VLM Depth Analysis (RealSense)",
                        "status": f"{len(depth_scene.get('detected_objects', []))} objects w/ depth",
                        "type": "success",
                    })
                else:
                    st.warning("Depth camera requires Isaac Sim bridge connection")

                st.session_state.system_status = "ready"
        except Exception as e:
            st.session_state.system_status = "error"
            st.error(f"Depth analysis failed: {e}")
            st.session_state.execution_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "action": "VLM Depth Analysis",
                "status": str(e),
                "type": "fail",
            })

    # Camera + perception tabs
    (cam_tab_rgb, cam_tab_depth, cam_tab_wrist,
     cam_tab_kit, cam_tab_vlm) = st.tabs(
        ["📹 RGB Overhead", "🔬 Depth (RealSense)",
         "🤖 Wrist (Live)", "📦 Tray (Kit)", "🎯 VLM Detections"])

    with cam_tab_rgb:
        try:
            annotated_image = capture_and_annotate()
            st.image(
                annotated_image,
                width="stretch",
                caption=f"Overhead • {st.session_state.perception_mode}",
            )
        except Exception as e:
            st.info(f"No RGB feed: {e}")

    with cam_tab_depth:
        if st.session_state.get("bridge_connected", False):
            try:
                # Always fetch a live depth frame (just like RGB tab)
                depth_img = st.session_state.camera.capture_depth_image()
                st.session_state.depth_image = depth_img
                st.image(
                    depth_img,
                    width="stretch",
                    caption="RealSense RSD455 • Pseudo Depth",
                )
            except Exception as e:
                st.info(f"No depth feed: {e}")
        else:
            st.info("🔌 Depth camera requires Isaac Sim bridge connection")

    with cam_tab_wrist:
        # Show the latest wrist camera image captured during workflow
        wrist_live = st.session_state.get("wrist_live_image")
        if wrist_live is not None:
            st.image(
                wrist_live,
                width="stretch",
                caption="Wrist RealSense • Live Scan",
            )
        elif st.session_state.get("bridge_connected", False):
            # No workflow image yet — try to fetch a live frame
            try:
                wrist_img = st.session_state.camera.capture_wrist_image()
                st.session_state["wrist_live_image"] = wrist_img
                st.image(
                    wrist_img,
                    width="stretch",
                    caption="Wrist RealSense • Live Feed",
                )
            except Exception as e:
                st.info(f"No wrist feed: {e}")
        else:
            st.info(
                "🔌 Wrist camera requires Isaac Sim bridge connection"
            )

    with cam_tab_kit:
        # Tray-overlook camera (/World/Camera_Kit). Used after each
        # place to verify the part actually landed in the tray. This
        # tab shows the most recent verification frame the workflow
        # captured — so the operator can independently audit the
        # gripper-confirmed-grasp signal.
        kit_verify = st.session_state.get("kit_verify_image")
        if kit_verify is not None:
            st.image(
                kit_verify,
                width="stretch",
                caption="Camera_Kit • last place-verification frame",
            )
        elif st.session_state.get("bridge_connected", False):
            try:
                kit_img = (
                    st.session_state.camera.capture_kit_image())
                st.image(
                    kit_img,
                    width="stretch",
                    caption="Camera_Kit • Live Feed",
                )
            except Exception as exc:
                st.info(f"Kit camera not available: {exc}")
        else:
            st.info(
                "🔌 Tray camera requires Isaac Sim bridge connection")

    with cam_tab_vlm:
        # Overlay image written by the workflow after each VLM analysis.
        # Red bboxes = VLM-labelled parts (with confidence). Green
        # bboxes = raw OWL-ViT2 / Qwen-grounding detections. If the
        # robot moves to the wrong place, comparing the overlay to
        # the raw RGB tab pinpoints whether the VLM, the bbox, or
        # the depth projection was at fault.
        overlay_img = st.session_state.get("vlm_overlay_image")
        if overlay_img is not None:
            st.image(
                overlay_img,
                width="stretch",
                caption="VLM detections — red = labelled parts, "
                        "green = raw detector bboxes",
            )
        else:
            st.info(
                "Run a workflow to populate this tab with the VLM's "
                "labelled detections."
            )

    # Stream status indicator
    if st.session_state.auto_stream:
        st.markdown(
            '<div class="stream-status stream-live">'
            '<div class="stream-dot"></div>'
            f'LIVE — {st.session_state.stream_fps:.1f} FPS | '
            f'{st.session_state.perception_mode}'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="stream-status" style="background:rgba(100,100,200,0.1);color:#94a3b8;'
            'border:1px solid rgba(100,100,200,0.15);">'
            '⏸️ PAUSED — Toggle auto-refresh in sidebar'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Detected Objects Table ───────────────────────────────
    if st.session_state.scene_description:
        st.markdown(
            '<div class="glass-card"><h3>🔎 Detected Objects</h3>',
            unsafe_allow_html=True,
        )
        objects = st.session_state.scene_description.get("detected_objects", [])
        if objects:
            import pandas as pd

            df = pd.DataFrame([
                {
                    "ID": obj["object_id"],
                    "Label": obj["label"],
                    "Conf": f"{obj['confidence']:.0%}",
                    "X": f"{obj['approximate_position'].get('x', 0):.3f}",
                    "Y": f"{obj['approximate_position'].get('y', 0):.3f}",
                    "Z": f"{obj['approximate_position'].get('z', 0):.3f}",
                    "Type": obj.get("affordance", "—"),
                }
                for obj in objects
            ])
            st.dataframe(
                df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Conf": st.column_config.TextColumn("Confidence"),
                    "Type": st.column_config.TextColumn("Affordance"),
                },
            )

            # Quick stats
            stat_cols = st.columns(3)
            stat_cols[0].metric("Objects", len(objects))
            avg_conf = sum(o["confidence"] for o in objects) / len(objects) if objects else 0
            stat_cols[1].metric("Avg Conf", f"{avg_conf:.0%}")
            unique_labels = len(set(o["label"] for o in objects))
            stat_cols[2].metric("Unique", unique_labels)
        else:
            st.info("No objects detected — run Scene Analysis")

        st.markdown("</div>", unsafe_allow_html=True)


# ─── Center Column: Command Interface ────────────────────────
with col_center:
    st.markdown(
        '<div class="glass-card"><h3>💬 Operator Console</h3>',
        unsafe_allow_html=True,
    )

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history[-20:]:
            if msg["role"] == "operator":
                st.markdown(
                    f'<div class="chat-operator">'
                    f'🧑‍🔧 <strong>Operator:</strong> {msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-system">'
                    f'🤖 <strong>System:</strong> {msg["content"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # Command input
    command = st.text_input(
        "Enter kitting command",
        placeholder="e.g., 'Pick all motor valves and place them in the kit tray'",
        key="command_input",
        label_visibility="collapsed",
    )

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        execute_btn = st.button(
            "▶️ Execute", width='stretch', type="primary", key="execute_btn"
        )
    with btn_col2:
        plan_only_btn = st.button(
            "📋 Plan Only", width='stretch', key="plan_btn"
        )
    with btn_col3:
        clear_btn = st.button(
            "🗑️ Clear Chat", width='stretch', key="clear_btn"
        )

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    # Support Enter key: detect when command changes
    should_execute = execute_btn
    should_plan = plan_only_btn

    if (should_execute or should_plan) and command:
        st.session_state.chat_history.append({"role": "operator", "content": command})

        is_bridge = st.session_state.get("bridge_connected", False)

        if should_execute and is_bridge:
            # ═══════════════════════════════════════════
            # FULL NEURO-SYMBOLIC WORKFLOW (Connected)
            # ═══════════════════════════════════════════
            st.session_state.system_status = "busy"

            workflow = KittingWorkflowEngine(
                camera=st.session_state.camera,
                vlm=st.session_state.vlm,
                planner=st.session_state.planner,
                config=st.session_state.config,
                recorder=st.session_state.get("evaluator"),
                place_verifier=st.session_state.get("place_verifier"),
            )

            # Phase progress collector
            phases_log = []

            def phase_callback(phase, status, detail):
                phases_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "phase": phase,
                    "status": status,
                    "detail": detail,
                })
                # Log to execution log
                type_map = {
                    "success": "success",
                    "running": "info",
                    "failed": "fail",
                    "warn": "retry",
                }
                st.session_state.execution_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "action": f"[{phase}] {detail[:60]}",
                    "status": status,
                    "type": type_map.get(status, "info"),
                })

            workflow.set_phase_callback(phase_callback)

            with st.spinner("🤖 Running neuro-symbolic kitting workflow..."):
                try:
                    result = workflow.execute_kitting_task(command)

                    # Update session state with results
                    if result.get("scene_description"):
                        st.session_state.scene_description = result["scene_description"]
                    if result.get("task_plan"):
                        st.session_state.task_plan = result["task_plan"]

                    # Build workflow report
                    wf_status = result.get("status", "unknown")
                    if wf_status == "completed":
                        phase_count = len(result.get("phases", []))
                        exec_results = result.get("execution_results", [])
                        success_steps = sum(1 for r in exec_results if r.get("status") == "success")
                        failed_steps = sum(1 for r in exec_results if r.get("status") == "failed")

                        msg = (
                            f"✅ **Workflow Complete!**\n\n"
                            f"📊 Phases: {phase_count} | "
                            f"Steps: {len(exec_results)} | "
                            f"✅ {success_steps} | ❌ {failed_steps}\n\n"
                        )

                        # Show plan summary
                        plan = result.get("task_plan", {})
                        if plan:
                            msg += f"📋 **Plan:** {plan.get('task_summary', '')}\n\n"
                            for s in plan.get("plan", []):
                                params = ", ".join(
                                    f"{k}={v}" for k, v in s.get("params", {}).items())
                                msg += f"✅ Step {s['step']}: `{s['action']}({params})`\n\n"

                        # Verification
                        verify = result.get("verification", {})
                        if verify:
                            v_objs = len(verify.get("detected_objects", []))
                            msg += f"\n🔍 **Verification:** {v_objs} objects in workspace\n\n"

                    elif wf_status == "error":
                        msg = f"❌ **Workflow Failed:** {result.get('error', 'Unknown error')}"
                    else:
                        msg = f"⚠️ **Workflow {wf_status}**"

                    st.session_state.chat_history.append({
                        "role": "system", "content": msg,
                    })

                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "system",
                        "content": f"❌ Workflow failed: {e}",
                    })

            st.session_state.system_status = "ready"

        else:
            # ═══════════════════════════════════════════
            # PLAN ONLY (or no bridge)
            # ═══════════════════════════════════════════

            # Step 1: Perception (if no scene data)
            if not st.session_state.scene_description:
                with st.spinner("📸 Capturing scene..."):
                    try:
                        image = st.session_state.camera.capture_workspace_image()
                        scene = st.session_state.vlm.analyze_scene(image)
                        st.session_state.scene_description = scene
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "system",
                            "content": f"❌ Perception failed: {e}",
                        })
                        st.rerun()

            # Step 2: Planning
            with st.spinner("🧠 Generating task plan..."):
                try:
                    plan = st.session_state.planner.generate_plan(
                        command,
                        st.session_state.scene_description,
                        workspace_bounds=st.session_state.config.get(
                            "execution", {}
                        ).get("workspace_bounds"),
                    )
                    st.session_state.task_plan = plan

                    plan_summary = (
                        f"📋 **Plan Generated:** {plan['task_summary']}\n\n"
                        f"**Steps:** {plan['total_steps']}\n\n"
                    )
                    for step in plan["plan"]:
                        params = ", ".join(
                            f"{k}={v}" for k, v in step.get("params", {}).items()
                        )
                        plan_summary += (
                            f"⬜ Step {step['step']}: "
                            f"`{step['action']}({params})`\n\n"
                        )

                    st.session_state.chat_history.append({
                        "role": "system", "content": plan_summary,
                    })

                    if should_execute and not is_bridge:
                        st.session_state.chat_history.append({
                            "role": "system",
                            "content": (
                                "⚠️ Isaac Sim bridge not connected. "
                                "Plan generated but not executed. "
                                "Start `isaac_sim_bridge.py` in Isaac Sim."
                            ),
                        })
                        for step in plan["plan"]:
                            st.session_state.execution_log.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "action": f"{step['action']}({step.get('params', {})})",
                                "status": "planned (offline)",
                                "type": "info",
                            })

                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "system",
                        "content": f"❌ Planning failed: {e}",
                    })

        st.rerun()

    # Task Plan Viewer
    if st.session_state.task_plan:
        with st.expander("📄 Full Task Plan JSON", expanded=False):
            st.json(st.session_state.task_plan)


# ─── Right Column: Execution Log ─────────────────────────────
with col_right:
    st.markdown(
        '<div class="glass-card"><h3>📊 Execution Log</h3>',
        unsafe_allow_html=True,
    )

    if st.session_state.execution_log:
        for entry in reversed(st.session_state.execution_log[-30:]):
            log_class = {
                "success": "log-success",
                "fail": "log-fail",
                "retry": "log-retry",
                "info": "log-info",
            }.get(entry.get("type", "info"), "log-info")

            icon = {
                "success": "✅",
                "fail": "❌",
                "retry": "🔄",
                "info": "ℹ️",
            }.get(entry.get("type", "info"), "ℹ️")

            st.markdown(
                f'<div class="{log_class}">'
                f'{icon} <strong>{entry["time"]}</strong> — '
                f'{entry["action"]}: {entry["status"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="log-info">ℹ️ No actions logged yet. '
            'Start by scanning the workspace.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Session Metrics
    if st.session_state.execution_log:
        st.markdown(
            '<div class="glass-card"><h3>📈 Session Metrics</h3>',
            unsafe_allow_html=True,
        )

        total = len(st.session_state.execution_log)
        success = sum(
            1 for e in st.session_state.execution_log if e.get("type") == "success"
        )
        failed = sum(
            1 for e in st.session_state.execution_log if e.get("type") == "fail"
        )

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total", total)
        mcol2.metric("Success", success)
        mcol3.metric("Failed", failed)

        if total > 0:
            st.progress(success / total if total > 0 else 0)

        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# QUANTITATIVE EVALUATION
# Surfaces the JSONL/SQLite data the workflow has been recording
# so the operator can pull thesis-quality numbers + CSV exports
# without leaving the dashboard.
# ═════════════════════════════════════════════════════════════

st.divider()
with st.expander("📊 Quantitative Evaluation", expanded=False):
 st.caption(
    "Per-event records persist to `logs/evaluation/eval.db` (SQLite) "
    "and `logs/evaluation/session_<id>.jsonl`. Use the CSV exports "
    "below to pull the tables that go in the thesis Results section."
 )

 evaluator = st.session_state.get("evaluator")
 if evaluator is None:
    st.info("Evaluation recorder not initialised.")
 else:
    eval_tab_summary, eval_tab_perception, eval_tab_pickplace, eval_tab_export = st.tabs(
        ["📈 Headline Numbers", "🎯 Perception", "🦾 Pick / Place", "💾 Export"]
    )

    summary = {}
    try:
        summary = evaluator.summary()
    except Exception as exc:
        st.warning(f"Cumulative summary unavailable: {exc}")

    # ── Headline cumulative metrics ─────────────────────────
    with eval_tab_summary:
        if not summary or not any(v.get("n", v.get("n_scans", 0))
                                  for v in summary.values()):
            st.info(
                "No evaluation events recorded yet. Run a kitting "
                "task to populate the metrics."
            )
        else:
            perc = summary.get("perception", {})
            pick = summary.get("pick", {})
            place = summary.get("place", {})
            task = summary.get("task", {})

            st.markdown("#### Perception (vs USD ground truth)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Scans", perc.get("n_scans", 0))
            c2.metric("Precision", f"{perc.get('precision', 0):.2f}")
            c3.metric("Recall", f"{perc.get('recall', 0):.2f}")
            c4.metric("F1", f"{perc.get('f1', 0):.2f}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Label accuracy",
                      f"{perc.get('label_accuracy', 0):.2f}")
            c2.metric("Mean IoU",
                      f"{perc.get('mean_iou', 0):.2f}")
            c3.metric(
                "Mean grounding error",
                f"{perc.get('mean_grounding_error_mm', 0):.1f} mm")
            c4.metric("Avg VLM latency",
                      f"{perc.get('vlm_latency_s', 0):.2f} s")

            st.markdown("#### Pick + Place")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pick attempts", pick.get("n", 0))
            c2.metric(
                "Pick success rate",
                f"{pick.get('rate', 0)*100:.1f} %")
            c3.metric("Avg pick cycle",
                      f"{pick.get('avg_cycle_s', 0):.2f} s")
            c4.metric(
                "Avg drift |xy|",
                f"{pick.get('avg_drift_xy_mm', 0):.1f} mm")

            c1, c2, c3, c4 = st.columns(4)
            n_place = place.get("n", 0)
            n_verified = place.get("n_verified", 0)
            place_label = (f"{n_place}"
                           if n_verified == n_place
                           else f"{n_place}  (verified: {n_verified})")
            c1.metric("Place attempts", place_label)
            c2.metric(
                "Place success (Camera_Kit VLM)",
                f"{place.get('rate_vlm', 0)*100:.1f} %"
                if n_verified else "—")
            c3.metric(
                "Avg place VLM confidence",
                f"{place.get('avg_vlm_confidence', 0):.2f}"
                if n_verified else "—")
            c4.metric("Avg place cycle",
                      f"{place.get('avg_cycle_s', 0):.2f} s")

            st.markdown("#### End-to-End Tasks")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total tasks", task.get("n", 0))
            c2.metric(
                "End-to-end success rate",
                f"{task.get('rate', 0)*100:.1f} %")
            c3.metric(
                "Avg completion ratio",
                f"{task.get('avg_completion_ratio', 0)*100:.1f} %")
            c4.metric(
                "Avg total time",
                f"{task.get('avg_end_to_end_s', 0):.1f} s")

    # ── Per-perception-event detail ─────────────────────────
    with eval_tab_perception:
        rows = evaluator.fetch_all("perception_events")
        if not rows:
            st.info("No perception events recorded yet.")
        else:
            # Strip the heavy detail_json column for the table view
            view = [
                {k: v for k, v in r.items() if k != "detail_json"}
                for r in rows
            ]
            try:
                import pandas as pd
                df = pd.DataFrame(view)
                st.dataframe(df, width='stretch', height=320)
                # IoU + grounding error trend
                if {"mean_iou", "mean_grounding_error_mm"} <= set(df.columns):
                    chart_df = df[
                        ["task_id", "mean_iou",
                         "mean_grounding_error_mm",
                         "precision_", "recall", "f1"]
                    ].set_index("task_id")
                    st.line_chart(chart_df)
            except Exception:
                st.write(view)

    # ── Per-pick / per-place detail ─────────────────────────
    with eval_tab_pickplace:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Pick events")
            picks = evaluator.fetch_all("pick_events")
            if picks:
                try:
                    import pandas as pd
                    pdf = pd.DataFrame([
                        {k: v for k, v in r.items()
                         if k != "detail_json"} for r in picks])
                    st.dataframe(pdf, width='stretch', height=260)
                except Exception:
                    st.write(picks[-10:])
            else:
                st.info("No pick events yet.")
        with col_b:
            st.markdown("##### Place events")
            places = evaluator.fetch_all("place_events")
            if places:
                try:
                    import pandas as pd
                    plf = pd.DataFrame([
                        {k: v for k, v in r.items()
                         if k != "detail_json"} for r in places])
                    st.dataframe(plf, width='stretch', height=260)
                except Exception:
                    st.write(places[-10:])
            else:
                st.info("No place events yet.")

        st.markdown("##### Tasks")
        tasks = evaluator.fetch_all("task_events")
        if tasks:
            try:
                import pandas as pd
                tdf = pd.DataFrame([
                    {k: v for k, v in r.items()
                     if k != "detail_json"} for r in tasks])
                st.dataframe(tdf, width='stretch', height=200)
            except Exception:
                st.write(tasks[-10:])
        else:
            st.info("No task events yet.")

    # ── CSV export ──────────────────────────────────────────
    with eval_tab_export:
        st.markdown(
            "Download the full per-event tables as CSV. Each table "
            "below maps to a column-set you can paste into the thesis "
            "Results chapter."
        )
        export_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "evaluation",
        )
        for table_name, friendly in (
            ("perception_events", "Perception (per scan)"),
            ("pick_events", "Pick events (per attempt)"),
            ("place_events", "Place events (per attempt)"),
            ("task_events", "Task events (per command)"),
        ):
            rows = evaluator.fetch_all(table_name)
            if not rows:
                st.caption(f"_{friendly} — no rows yet_")
                continue
            out_path = os.path.join(
                export_dir, f"{table_name}.csv")
            evaluator.export_csv(table_name, out_path)
            try:
                with open(out_path, "rb") as f:
                    st.download_button(
                        label=f"⬇ {friendly}  ({len(rows)} rows)",
                        data=f.read(),
                        file_name=f"{table_name}.csv",
                        mime="text/csv",
                        key=f"dl_{table_name}",
                    )
            except Exception as exc:
                st.warning(
                    f"Could not prepare {table_name} CSV: {exc}")


# ═════════════════════════════════════════════════════════════
# BOTTOM — Scene Summary + Perception Legend
# ═════════════════════════════════════════════════════════════

if st.session_state.scene_description:
    st.divider()

    bottom_col1, bottom_col2 = st.columns([2, 1])

    with bottom_col1:
        st.markdown("### 🌐 Scene Summary")
        st.info(
            st.session_state.scene_description.get(
                "scene_summary", "No summary available"
            )
        )

    with bottom_col2:
        st.markdown("### 🎨 Perception Mode Legend")
        st.markdown("""
| Mode | Overlay |
|------|---------|
| 📷 Raw Feed | No overlay — clean camera image |
| 🎯 VLM Detections | Bounding boxes + labels + confidence |
| 🔥 Confidence Map | Colour-coded circles (red→green) |
| 🏷️ Affordance View | Shaped markers by physical property |
| 📐 Scene Grid | Coordinate grid with markers |
""")


# ═════════════════════════════════════════════════════════════
# AUTO-REFRESH STREAM (via st.rerun with sleep)
# ═════════════════════════════════════════════════════════════

if st.session_state.auto_stream:
    delay = 1.0 / max(0.2, st.session_state.stream_fps)
    time.sleep(delay)
    st.rerun()
