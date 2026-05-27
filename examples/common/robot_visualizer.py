#!/usr/bin/env python3
"""Shared robot visualizer for Piper robot demos.

This module acts as a facade, delegating 3D rendering to RobotVisualizerCore
and 2D UI elements to RobotVisualizerGUI.
"""

from typing import Any

from .visualizer_core import RobotVisualizerCore
from .visualizer_gui import RobotVisualizerGUI


class RobotVisualizer:
    """Shared visualizer facade for robot demos."""

    def __init__(self, urdf_path: str) -> None:
        """Initialize the visualizer components."""
        self._core = RobotVisualizerCore(urdf_path)
        self._gui = RobotVisualizerGUI(self._core.server)

    def __getattr__(self, name: str) -> Any:
        """Dynamically delegates method calls to the Core or GUI modules.

        If a top-level script calls `visualizer.add_basic_controls()`,
        this routes it to `_gui.add_basic_controls()`.
        """
        if hasattr(self._core, name):
            return getattr(self._core, name)
        if hasattr(self._gui, name):
            return getattr(self._gui, name)

        raise AttributeError(f"'RobotVisualizer' object has no attribute '{name}'")
