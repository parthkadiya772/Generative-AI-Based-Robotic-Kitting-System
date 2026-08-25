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

__version__ = "2.0.1"
__author__ = "Parth"
