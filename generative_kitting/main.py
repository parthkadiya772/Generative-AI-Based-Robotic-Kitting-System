"""
Generative Kitting System — Main Application.

Orchestrates the full closed-loop pipeline:
    1. Capture workspace image from Isaac Sim camera
    2. Send image to VLM → get scene description
    3. Send (user_command + scene) to LLM → get task plan
    4. Validate task plan
    5. Execute task plan via RobotController
    6. (Optional) Post-execution VLM verification

Can run in:
    - Interactive CLI mode (terminal REPL)
    - Headless mode (single command)
    - Streamlit UI mode (web interface)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import yaml

# Add package root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger, get_session_id, log
from utils.image_utils import save_debug_image
from perception.vlm_perception import VLMPerception
from perception.camera_interface import CameraInterface, MockCameraInterface
from orchestration.llm_planner import LLMPlanner
from knowledge.parts_database import PartsDatabase
from knowledge.seed_data import seed_database


class KittingApplication:
    """
    Main application class for the Generative Kitting System.

    Ties together all three layers (Perception → Orchestration → Execution)
    into a closed-loop pipeline.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration and initialise all modules.

        Parameters
        ----------
        config_path : str
            Path to the config.yaml file.
        """
        # Load config
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config_path
        )
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        # Set up logging
        setup_logger(self.config)
        self.session_id = get_session_id()
        log.info(f"═══ Kitting Application Starting — Session: {self.session_id} ═══")

        # Initialise Knowledge Layer
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "knowledge", "kitting.db"
        )
        self.db = self._init_database(db_path)

        # Initialise Perception Layer (VLM)
        self.vlm = VLMPerception(self.config.get("perception", {}))

        # Initialise Orchestration Layer (LLM)
        self.planner = LLMPlanner(self.config.get("orchestration", {}))

        # Camera (Isaac Sim or mock)
        self.camera = None

        # Execution Layer (initialised async when in sim mode)
        self.controller = None

        # State
        self.last_scene_description = None
        self.last_task_plan = None
        self.execution_history = []

        log.info("KittingApplication modules initialised")

    def _init_database(self, db_path: str) -> PartsDatabase:
        """Initialise or open the parts database, seeding if empty."""
        db = PartsDatabase(db_path)
        if not db.get_all_parts():
            log.info("Database empty — seeding with sample data")
            from knowledge.seed_data import seed_parts, seed_kits
            seed_parts(db)
            seed_kits(db)
        return db

    async def init_sim(self):
        """
        Initialise Isaac Sim components (robot controller + camera).
        Must be called within an Isaac Sim async context.
        """
        from execution.robot_controller import RobotController

        self.controller = RobotController(self.config)
        await self.controller.init_sim()

        # Set up camera
        camera_path = self.config.get("simulation", {}).get(
            "camera_prim_path", "/World/Camera_Top"
        )
        self.camera = CameraInterface(camera_prim_path=camera_path)

        if not self.camera.is_available:
            log.warning("Isaac Sim camera not available — using mock")
            self.camera = MockCameraInterface()

        # Register perception callback for request_perception_update
        async def perception_callback():
            return await asyncio.coroutine(self._capture_and_analyze)()

        self.controller.set_perception_callback(perception_callback)

        log.info("Isaac Sim initialisation complete")

    def init_standalone(self, mock_image_path: str = None):
        """
        Initialise for standalone mode (no Isaac Sim).
        Useful for testing perception + orchestration layers independently.
        """
        self.camera = MockCameraInterface(image_path=mock_image_path)
        self.controller = None
        log.info("Standalone mode initialised (no robot controller)")

    # ═════════════════════════════════════════════════════════
    # CORE PIPELINE
    # ═════════════════════════════════════════════════════════

    def run_kitting_cycle(self, user_command: str) -> Dict[str, Any]:
        """
        Execute a full kitting cycle: perceive → plan → execute.

        Parameters
        ----------
        user_command : str
            Natural language command from the operator.

        Returns
        -------
        dict
            Full execution report.
        """
        report = {
            "session_id": self.session_id,
            "user_command": user_command,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": {},
            "success": False,
        }

        cycle_start = time.time()

        try:
            # ── Step 1: Capture workspace image ──────────────
            log.info("═══ Step 1: Capturing workspace image ═══")
            image = self.camera.capture_workspace_image()

            if self.config.get("logging", {}).get("save_vlm_images", True):
                img_path = save_debug_image(image, prefix="cycle")
                report["steps"]["image_path"] = img_path

            # ── Step 2: VLM Scene Analysis ───────────────────
            log.info("═══ Step 2: VLM scene analysis ═══")
            vlm_start = time.time()
            scene_description = self.vlm.analyze_scene(image)
            vlm_time = time.time() - vlm_start

            self.last_scene_description = scene_description
            report["steps"]["perception"] = {
                "detected_objects": len(scene_description.get("detected_objects", [])),
                "scene_summary": scene_description.get("scene_summary", ""),
                "time_seconds": round(vlm_time, 2),
            }

            if self.config.get("logging", {}).get("save_task_plans", True):
                plan_dir = os.path.join(
                    self.config.get("logging", {}).get("log_dir", "logs"),
                    "vlm_outputs",
                )
                os.makedirs(plan_dir, exist_ok=True)
                with open(
                    os.path.join(plan_dir, f"scene_{self.session_id}.json"), "w"
                ) as f:
                    json.dump(scene_description, f, indent=2)

            log.info(
                f"VLM detected {len(scene_description['detected_objects'])} objects "
                f"in {vlm_time:.1f}s"
            )

            # ── Step 3: LLM Task Planning ────────────────────
            log.info("═══ Step 3: LLM task planning ═══")
            llm_start = time.time()

            workspace_bounds = self.config.get("execution", {}).get("workspace_bounds")
            # Kit tray position is detected by the perception pipeline,
            # not hardcoded.  In CLI mode (no bridge), pass None so the
            # LLM knows the tray wasn't localized.
            task_plan = self.planner.generate_plan(
                user_command=user_command,
                scene_description=scene_description,
                workspace_bounds=workspace_bounds,
                kit_tray_position=None,
            )
            llm_time = time.time() - llm_start

            self.last_task_plan = task_plan
            report["steps"]["planning"] = {
                "task_summary": task_plan.get("task_summary", ""),
                "total_steps": task_plan.get("total_steps", 0),
                "time_seconds": round(llm_time, 2),
            }

            if self.config.get("logging", {}).get("save_task_plans", True):
                plan_dir = os.path.join(
                    self.config.get("logging", {}).get("log_dir", "logs"),
                    "task_plans",
                )
                os.makedirs(plan_dir, exist_ok=True)
                with open(
                    os.path.join(plan_dir, f"plan_{self.session_id}.json"), "w"
                ) as f:
                    json.dump(task_plan, f, indent=2)

            log.info(
                f"LLM generated {task_plan['total_steps']}-step plan in {llm_time:.1f}s"
            )

            # ── Step 4: Execute Task Plan ────────────────────
            if self.controller is not None:
                log.info("═══ Step 4: Executing task plan ═══")
                # Run the async executor
                exec_report = asyncio.get_event_loop().run_until_complete(
                    self.controller.execute_plan(task_plan, self.session_id)
                )
                report["steps"]["execution"] = exec_report

                # Log all actions to database
                for step_result in exec_report.get("step_results", []):
                    self.db.log_action(
                        session_id=self.session_id,
                        action=step_result["action"],
                        object_id=step_result.get("params", {}).get("object_id", ""),
                        params=step_result.get("params"),
                        success=step_result["success"],
                        error_message=step_result.get("error", ""),
                        duration_seconds=step_result.get("duration_seconds", 0),
                    )

                report["success"] = exec_report["failed_steps"] == 0
            else:
                log.warning("No robot controller — plan generated but not executed")
                report["steps"]["execution"] = {
                    "status": "skipped",
                    "reason": "No robot controller (standalone mode)",
                }
                report["success"] = True  # Plan generation succeeded

        except Exception as e:
            log.error(f"Kitting cycle FAILED: {e}")
            report["error"] = str(e)
            import traceback
            traceback.print_exc()

        report["total_time_seconds"] = round(time.time() - cycle_start, 2)
        self.execution_history.append(report)

        log.info(
            f"═══ Kitting cycle complete — "
            f"{'SUCCESS' if report['success'] else 'FAILED'} — "
            f"{report['total_time_seconds']}s ═══"
        )

        return report

    def _capture_and_analyze(self) -> Dict[str, Any]:
        """Capture image and run VLM analysis (synchronous wrapper)."""
        image = self.camera.capture_workspace_image()
        return self.vlm.analyze_scene(image)

    # ═════════════════════════════════════════════════════════
    # INTERACTIVE MODE (CLI)
    # ═════════════════════════════════════════════════════════

    def interactive_mode(self):
        """
        Terminal-based REPL for operator interaction.

        Commands:
            - Any text → treated as a kitting command
            - "scan" → trigger VLM perception only
            - "plan <cmd>" → generate plan without executing
            - "parts" → list all parts in the database
            - "kits" → list all kit definitions
            - "history" → show execution history
            - "quit" / "exit" → exit
        """
        print("\n" + "═" * 60)
        print("  🤖 GENERATIVE KITTING SYSTEM — Interactive Mode")
        print(f"  Session: {self.session_id}")
        print("═" * 60)
        print("  Type a kitting command, or 'help' for options.")
        print("  Type 'quit' to exit.\n")

        while True:
            try:
                command = input("Operator > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not command:
                continue

            cmd_lower = command.lower()

            if cmd_lower in ("quit", "exit", "q"):
                print("Shutting down...")
                break

            elif cmd_lower == "help":
                self._print_help()

            elif cmd_lower == "scan":
                self._cmd_scan()

            elif cmd_lower.startswith("plan "):
                self._cmd_plan(command[5:].strip())

            elif cmd_lower == "parts":
                self._cmd_parts()

            elif cmd_lower == "kits":
                self._cmd_kits()

            elif cmd_lower == "history":
                self._cmd_history()

            elif cmd_lower == "scene":
                self._cmd_scene()

            else:
                # Full kitting cycle
                report = self.run_kitting_cycle(command)
                self._print_report(report)

    def _print_help(self):
        print("""
Available commands:
  <any text>     Run a full kitting cycle (perceive → plan → execute)
  scan           Trigger VLM perception and show detected objects
  plan <cmd>     Generate a task plan without executing
  scene          Show last VLM scene description
  parts          List all parts in the database
  kits           List all kit definitions
  history        Show execution history for this session
  help           Show this help message
  quit           Exit interactive mode
""")

    def _cmd_scan(self):
        print("Scanning workspace...")
        try:
            image = self.camera.capture_workspace_image()
            scene = self.vlm.analyze_scene(image)
            self.last_scene_description = scene
            print(f"\n📸 Scene Summary: {scene['scene_summary']}")
            print(f"   Detected {len(scene['detected_objects'])} objects:")
            for obj in scene["detected_objects"]:
                pos = obj.get("approximate_position", {})
                print(
                    f"   • {obj['object_id']:10s} | {obj['label']:20s} | "
                    f"conf={obj['confidence']:.2f} | "
                    f"pos=({pos.get('x',0):.3f}, {pos.get('y',0):.3f}, {pos.get('z',0):.3f})"
                )
            print()
        except Exception as e:
            print(f"❌ Scan failed: {e}\n")

    def _cmd_plan(self, command: str):
        if not self.last_scene_description:
            print("No scene data — run 'scan' first.\n")
            return
        print(f"Planning: '{command}'...")
        try:
            plan = self.planner.generate_plan(
                command, self.last_scene_description,
                workspace_bounds=self.config.get("execution", {}).get("workspace_bounds"),
            )
            self.last_task_plan = plan
            print(f"\n📋 Task Plan: {plan['task_summary']}")
            print(f"   Steps: {plan['total_steps']}")
            for step in plan["plan"]:
                params_str = ", ".join(f"{k}={v}" for k, v in step.get("params", {}).items())
                print(f"   {step['step']:3d}. {step['action']}({params_str})")
            print()
        except Exception as e:
            print(f"❌ Planning failed: {e}\n")

    def _cmd_parts(self):
        parts = self.db.get_all_parts()
        print(f"\n📦 Parts Database ({len(parts)} entries):")
        for p in parts:
            print(
                f"   {p['part_id']:30s} | {p['label']:20s} | "
                f"{p['category']:12s} | {p['material']:10s} | "
                f"{'FRAGILE' if p['fragile'] else 'sturdy':>7s}"
            )
        print()

    def _cmd_kits(self):
        kits = self.db.get_all_kits()
        print(f"\n📋 Kit Definitions ({len(kits)} kits):")
        for k in kits:
            parts_str = ", ".join(
                f"{r['quantity']}x {r['label']}" for r in k["required_parts"]
            )
            print(f"   {k['kit_name']:30s} → {parts_str}")
        print()

    def _cmd_history(self):
        summary = self.db.get_session_summary(self.session_id)
        print(f"\n📊 Session History ({self.session_id}):")
        print(f"   Total actions:  {summary['total_actions']}")
        print(f"   Successes:      {summary['successes']}")
        print(f"   Failures:       {summary['failures']}")
        print(f"   Success rate:   {summary['success_rate']}%")
        print(f"   Total time:     {summary['total_time_seconds']}s")
        print()

    def _cmd_scene(self):
        if self.last_scene_description:
            print(json.dumps(self.last_scene_description, indent=2))
        else:
            print("No scene data — run 'scan' first.\n")

    @staticmethod
    def _print_report(report: dict):
        print(f"\n{'═' * 50}")
        print(f"  {'✅ SUCCESS' if report['success'] else '❌ FAILED'}")
        print(f"  Command: {report['user_command']}")
        print(f"  Time: {report['total_time_seconds']}s")

        exe = report.get("steps", {}).get("execution", {})
        if isinstance(exe, dict) and "total_steps" in exe:
            print(f"  Actions: {exe.get('successful_steps', 0)}/{exe['total_steps']} succeeded")

        if "error" in report:
            print(f"  Error: {report['error']}")
        print(f"{'═' * 50}\n")

    def cleanup(self):
        """Clean up resources."""
        if self.db:
            self.db.close()
        log.info("Application cleanup complete")


# ═════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generative AI Robotic Kitting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                               # Interactive mode (standalone)
  python main.py -c "Pick all motor valves"     # Single command (standalone)
  python main.py --sim                          # Interactive mode (Isaac Sim)
  python main.py --ui                           # Launch Streamlit UI
        """,
    )

    parser.add_argument(
        "-c", "--command",
        type=str, default=None,
        help="Execute a single kitting command and exit",
    )
    parser.add_argument(
        "--config",
        type=str, default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Enable Isaac Sim mode (requires running inside Isaac Sim)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit web UI",
    )
    parser.add_argument(
        "--seed-db",
        action="store_true",
        help="Re-seed the parts database with sample data",
    )
    parser.add_argument(
        "--mock-image",
        type=str, default=None,
        help="Path to a mock workspace image for standalone testing",
    )

    args = parser.parse_args()

    # Launch Streamlit UI
    if args.ui:
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "streamlit_app.py")
        os.system(f"streamlit run {ui_path}")
        return

    # Create application
    app = KittingApplication(config_path=args.config)

    # Re-seed database if requested
    if args.seed_db:
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "knowledge", "kitting.db"
        )
        seed_database(db_path)
        print("Database re-seeded.")
        return

    # Initialise mode
    if args.sim:
        # Isaac Sim mode — must be run inside the simulator
        asyncio.get_event_loop().run_until_complete(app.init_sim())
    else:
        # Standalone mode
        app.init_standalone(mock_image_path=args.mock_image)

    # Execute
    try:
        if args.command:
            report = app.run_kitting_cycle(args.command)
            print(json.dumps(report, indent=2))
        else:
            app.interactive_mode()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
