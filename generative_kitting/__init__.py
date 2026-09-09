"""
Generative AI-based Robotic Kitting System
============================================

A neuro-symbolic framework that uses VLM for zero-shot object recognition
and LLM for natural language task orchestration, connected to a simulated
UR10 robotic arm in NVIDIA Isaac Sim.

Three-layer architecture:
    1. Perception Layer  — VLM-based scene understanding
    2. Orchestration Layer — LLM-based task planning
    3. Execution Layer   — Deterministic robot control via Isaac Sim
"""

import os
import sys


# Keep the existing flat imports working when this package is imported from the
# repository root (for example, with ``python -m generative_kitting.main``).
_package_root = os.path.dirname(os.path.abspath(__file__))
if _package_root not in sys.path:
    sys.path.insert(0, _package_root)


__version__ = "0.1.0"
__author__ = "Parth"
