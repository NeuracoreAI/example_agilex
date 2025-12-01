#!/usr/bin/env python3
"""GUI-based Robot Control.

Control the Piper robot using a Tkinter GUI.
This file contains everything needed for GUI control in one place.
"""

import sys
from pathlib import Path

# Add repo root to path to import piper_controller
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.spatial.transform import Rotation

from piper_controller import PiperController


class GUIRobotControl:
    """GUI robot control - receives GUI commands and controls robot."""

    def __init__(
        self,
        can_interface: str = "can0",
        robot_rate: float = 100.0,
        debug_mode: bool = False,
    ) -> None:
        """Initialize GUI robot control.

        Args:
            can_interface: CAN interface for robot communication
            robot_rate: Robot control loop rate in Hz (default: 100)
            debug_mode: Enable debug logging
        """
        # Thread-safe command queue
        self.command_queue: queue.Queue = queue.Queue()
        self.running = threading.Event()
        self.running.set()

        # Track which buttons are currently pressed
        self.pressed_buttons = {
            # End-effector controls
            "x_pos": False,
            "x_neg": False,
            "y_pos": False,
            "y_neg": False,
            "z_pos": False,
            "z_neg": False,
            "roll_pos": False,
            "roll_neg": False,
            "pitch_pos": False,
            "pitch_neg": False,
            "yaw_pos": False,
            "yaw_neg": False,
            # Joint controls
            "j1_pos": False,
            "j1_neg": False,
            "j2_pos": False,
            "j2_neg": False,
            "j3_pos": False,
            "j3_neg": False,
            "j4_pos": False,
            "j4_neg": False,
            "j5_pos": False,
            "j5_neg": False,
            "j6_pos": False,
            "j6_neg": False,
            # Gripper controls
            "gripper_open": False,
            "gripper_close": False,
        }

        # Control speeds (per second)
        self.linear_speed = 50.0  # 5mm/s
        self.angular_speed = 50.0  # 50 degrees per second
        self.joint_speed = 50.0  # 50 degrees per second for joints
        self.gripper_speed = 1.0  # 1.0 normalized units per second (0.0 to 1.0 range)
        self.control_update_rate = robot_rate // 2  # Hz (half of the robot rate)

        # Store debug mode for command handler
        self.debug_mode = debug_mode

        # Initialize robot controller
        print("Initializing robot controller...")
        self.robot = PiperController(
            can_interface=can_interface, robot_rate=robot_rate, debug_mode=debug_mode
        )

        # Initialize GUI
        print("Initializing GUI...")
        self._create_gui()
        print("✓ GUI initialized!")

        print("\nGUI Robot Control ready!")
        print("Use the GUI to control the robot.")
        print("Hold buttons to move continuously.")

    def _create_gui(self) -> None:
        """Create the GUI elements."""
        self.root = tk.Tk()
        self.root.title("Robot Control GUI")
        self.root.geometry("450x1300")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))  # type: ignore[arg-type]

        # Title
        title_label = ttk.Label(
            main_frame, text="Robot Control - Hold to Move", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Control Mode", padding="10")
        mode_frame.grid(
            row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)  # type: ignore[arg-type]
        )

        self.control_mode_var = tk.StringVar(value="joint_space")

        ee_radio = ttk.Radiobutton(
            mode_frame,
            text="End-Effector Control",
            variable=self.control_mode_var,
            value="end_effector",
            command=self._on_mode_changed,
        )
        ee_radio.grid(row=0, column=0, padx=10, sticky=tk.W)

        joint_radio = ttk.Radiobutton(
            mode_frame,
            text="Joint Control",
            variable=self.control_mode_var,
            value="joint_space",
            command=self._on_mode_changed,
        )
        joint_radio.grid(row=0, column=1, padx=10, sticky=tk.W)

        # Instructions
        instructions = ttk.Label(
            main_frame, text="Hold buttons to move continuously", font=("Arial", 10)
        )
        instructions.grid(row=2, column=0, columnspan=3, pady=(0, 20))

        # Unified control area - switches between EE and Joint controls
        self.control_frame = ttk.LabelFrame(
            main_frame, text="Joint Control", padding="10"
        )
        self.control_frame.grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)  # type: ignore[arg-type]
        )

        # Gripper control section
        gripper_frame = ttk.LabelFrame(main_frame, text="Gripper Control", padding="10")
        gripper_frame.grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)  # type: ignore[arg-type]
        )

        # Gripper speed control
        ttk.Label(
            gripper_frame, text="Gripper Speed (%/s):", font=("Arial", 9, "bold")
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.gripper_speed_var = tk.StringVar(value=str(self.gripper_speed))
        gripper_speed_entry = ttk.Entry(
            gripper_frame, textvariable=self.gripper_speed_var, width=8
        )
        gripper_speed_entry.grid(row=0, column=1, padx=(0, 10))
        gripper_speed_entry.bind("<Return>", lambda e: self._update_gripper_speed())

        # Gripper buttons
        ttk.Label(
            gripper_frame,
            text="Gripper (Hold to Open/Close)",
            font=("Arial", 10, "bold"),
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 2))

        # Close button
        close_btn = tk.Button(
            gripper_frame,
            text="Close Gripper",
            width=15,
            height=2,
            bg="#ffcccc",
            activebackground="#ff9999",
        )
        close_btn.grid(row=2, column=0, padx=5, pady=5)
        close_btn.bind(
            "<ButtonPress-1>", lambda e: self._on_button_press("gripper_close")
        )
        close_btn.bind(
            "<ButtonRelease-1>", lambda e: self._on_button_release("gripper_close")
        )

        # Open button
        open_btn = tk.Button(
            gripper_frame,
            text="Open Gripper",
            width=15,
            height=2,
            bg="#ccffcc",
            activebackground="#99ff99",
        )
        open_btn.grid(row=2, column=1, padx=5, pady=5)
        open_btn.bind(
            "<ButtonPress-1>", lambda e: self._on_button_press("gripper_open")
        )
        open_btn.bind(
            "<ButtonRelease-1>", lambda e: self._on_button_release("gripper_open")
        )

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=10)

        # Robot status display - separate fields
        status_frame = ttk.LabelFrame(main_frame, text="Robot Status", padding="10")
        status_frame.grid(
            row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10)  # type: ignore[arg-type]
        )

        # Robot state
        ttk.Label(status_frame, text="Robot State:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.robot_state_label = ttk.Label(
            status_frame, text="DISABLED", foreground="red"
        )
        self.robot_state_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # Target position
        ttk.Label(
            status_frame, text="Target Position (mm):", font=("Arial", 10, "bold")
        ).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.target_pos_label = ttk.Label(status_frame, text="X: 0.0  Y: 0.0  Z: 0.0")
        self.target_pos_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # Target orientation
        ttk.Label(
            status_frame, text="Target Orientation (°):", font=("Arial", 10, "bold")
        ).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.target_rot_label = ttk.Label(
            status_frame, text="RX: 0.0  RY: 0.0  RZ: 0.0"
        )
        self.target_rot_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # Target gripper
        ttk.Label(
            status_frame, text="Target Gripper (°):", font=("Arial", 10, "bold")
        ).grid(row=3, column=0, sticky=tk.W, pady=2)
        self.target_gripper_label = ttk.Label(status_frame, text="0.0°")
        self.target_gripper_label.grid(
            row=3, column=1, sticky=tk.W, padx=(10, 0), pady=2
        )

        # Current position
        ttk.Label(
            status_frame, text="Current Position (mm):", font=("Arial", 10, "bold")
        ).grid(row=4, column=0, sticky=tk.W, pady=2)
        self.current_pos_label = ttk.Label(
            status_frame, text="No feedback", foreground="gray"
        )
        self.current_pos_label.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # Current orientation
        ttk.Label(
            status_frame, text="Current Orientation (°):", font=("Arial", 10, "bold")
        ).grid(row=5, column=0, sticky=tk.W, pady=2)
        self.current_rot_label = ttk.Label(
            status_frame, text="No feedback", foreground="gray"
        )
        self.current_rot_label.grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # Current gripper
        ttk.Label(
            status_frame, text="Current Gripper (°):", font=("Arial", 10, "bold")
        ).grid(row=6, column=0, sticky=tk.W, pady=2)
        self.current_gripper_label = ttk.Label(
            status_frame, text="No feedback", foreground="gray"
        )
        self.current_gripper_label.grid(
            row=6, column=1, sticky=tk.W, padx=(10, 0), pady=2
        )

        # Joint angles
        ttk.Label(
            status_frame, text="Joint Angles (°):", font=("Arial", 10, "bold")
        ).grid(row=7, column=0, sticky=tk.W, pady=2)
        self.joint_angles_label = ttk.Label(
            status_frame, text="No feedback", foreground="gray"
        )
        self.joint_angles_label.grid(row=7, column=1, sticky=tk.W, padx=(10, 0), pady=2)

        # Target joint angles
        ttk.Label(
            status_frame, text="Target Joint Angles (°):", font=("Arial", 10, "bold")
        ).grid(row=8, column=0, sticky=tk.W, pady=2)
        self.target_joint_angles_label = ttk.Label(
            status_frame, text="No target", foreground="gray"
        )
        self.target_joint_angles_label.grid(
            row=8, column=1, sticky=tk.W, padx=(10, 0), pady=2
        )

        # Control buttons (moved below status display)
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=20)

        # Stop/Resume button
        self.stop_resume_button = ttk.Button(
            button_frame, text="Stop Robot", command=self._on_stop_resume_pressed
        )
        self.stop_resume_button.grid(row=0, column=0, padx=10)

        # Home button
        self.home_button = ttk.Button(
            button_frame, text="Home", command=self._on_home_pressed
        )
        self.home_button.grid(row=0, column=1, padx=10)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Initialize control frame with joint controls
        self.control_frame.config(text=f"Joint Control ({self.joint_speed}°/s)")
        self._create_joint_controls()

        # Set initial control mode to joint space
        self.robot.set_control_mode(PiperController.ControlMode.JOINT_SPACE)

    def _on_button_press(self, key: str) -> None:
        """Called when a control button is pressed."""
        self.pressed_buttons[key] = True

    def _on_button_release(self, key: str) -> None:
        """Called when a control button is released."""
        self.pressed_buttons[key] = False

    def _on_mode_changed(self) -> None:
        """Called when control mode is changed."""
        mode_str = self.control_mode_var.get()
        if mode_str == "end_effector":
            mode = PiperController.ControlMode.END_EFFECTOR
        elif mode_str == "joint_space":
            mode = PiperController.ControlMode.JOINT_SPACE
        else:
            return

        # Update robot controller mode
        self.robot.set_control_mode(mode)

        # Clear the control frame and rebuild with appropriate controls
        for widget in self.control_frame.winfo_children():
            widget.destroy()

        if mode == PiperController.ControlMode.END_EFFECTOR:
            self.control_frame.config(text="End-Effector Control")
            self._create_end_effector_controls()
            self.status_label.config(
                text="Switched to End-Effector Control", foreground="blue"
            )
        else:
            self.control_frame.config(text=f"Joint Control ({self.joint_speed}°/s)")
            self._create_joint_controls()
            self.status_label.config(
                text="Switched to Joint Control", foreground="blue"
            )

        # Reset status after 2 seconds
        self.root.after(
            2000, lambda: self.status_label.config(text="Ready", foreground="green")
        )

    def _create_end_effector_controls(self) -> None:
        """Create end-effector control buttons in the control frame."""
        # Speed control section
        speed_frame = ttk.LabelFrame(
            self.control_frame, text="Speed Controls", padding="5"
        )
        speed_frame.grid(
            row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10)  # type: ignore[arg-type]
        )

        # Linear speed control
        ttk.Label(
            speed_frame, text="Linear Speed (mm/s):", font=("Arial", 9, "bold")
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.linear_speed_var = tk.StringVar(value=str(self.linear_speed))
        linear_speed_entry = ttk.Entry(
            speed_frame, textvariable=self.linear_speed_var, width=8
        )
        linear_speed_entry.grid(row=0, column=1, padx=(0, 10))
        linear_speed_entry.bind("<Return>", lambda e: self._update_linear_speed())

        # Angular speed control
        ttk.Label(
            speed_frame, text="Angular Speed (°/s):", font=("Arial", 9, "bold")
        ).grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        self.angular_speed_var = tk.StringVar(value=str(self.angular_speed))
        angular_speed_entry = ttk.Entry(
            speed_frame, textvariable=self.angular_speed_var, width=8
        )
        angular_speed_entry.grid(row=0, column=3, padx=(0, 5))
        angular_speed_entry.bind("<Return>", lambda e: self._update_angular_speed())

        # Linear buttons (X, Y, Z) with +/- for each
        linear_controls = [
            ("X (Forward/Back)", "x_neg", "x_pos", "← Back", "Forward →"),
            ("Y (Left/Right)", "y_neg", "y_pos", "← Left", "Right →"),
            ("Z (Down/Up)", "z_neg", "z_pos", "↓ Down", "Up ↑"),
        ]

        for i, (label, neg_key, pos_key, neg_text, pos_text) in enumerate(
            linear_controls
        ):
            ttk.Label(self.control_frame, text=label, font=("Arial", 10, "bold")).grid(
                row=i * 2, column=0, columnspan=2, sticky=tk.W, pady=(10, 2)
            )

            # Negative button
            neg_btn = tk.Button(
                self.control_frame,
                text=neg_text,
                width=15,
                height=2,
                bg="#ffcccc",
                activebackground="#ff9999",
            )
            neg_btn.grid(row=i * 2 + 1, column=0, padx=5, pady=5)
            neg_btn.bind(
                "<ButtonPress-1>", lambda e, k=neg_key: self._on_button_press(k)  # type: ignore[misc]
            )
            neg_btn.bind(
                "<ButtonRelease-1>", lambda e, k=neg_key: self._on_button_release(k)  # type: ignore[misc]
            )

            # Positive button
            pos_btn = tk.Button(
                self.control_frame,
                text=pos_text,
                width=15,
                height=2,
                bg="#ccffcc",
                activebackground="#99ff99",
            )
            pos_btn.grid(row=i * 2 + 1, column=1, padx=5, pady=5)
            pos_btn.bind(
                "<ButtonPress-1>", lambda e, k=pos_key: self._on_button_press(k)  # type: ignore[misc]
            )
            pos_btn.bind(
                "<ButtonRelease-1>", lambda e, k=pos_key: self._on_button_release(k)  # type: ignore[misc]
            )

        # Angular buttons (Roll, Pitch, Yaw) with +/- for each
        angular_controls = [
            ("Roll", "roll_neg", "roll_pos", "↶ CCW", "CW ↷"),
            ("Pitch", "pitch_neg", "pitch_pos", "↓ Down", "Up ↑"),
            ("Yaw", "yaw_neg", "yaw_pos", "← Left", "Right →"),
        ]

        for i, (label, neg_key, pos_key, neg_text, pos_text) in enumerate(
            angular_controls
        ):
            ttk.Label(self.control_frame, text=label, font=("Arial", 10, "bold")).grid(
                row=i * 2 + 6, column=0, columnspan=2, sticky=tk.W, pady=(10, 2)
            )

            # Negative button
            neg_btn = tk.Button(
                self.control_frame,
                text=neg_text,
                width=15,
                height=2,
                bg="#ffcccc",
                activebackground="#ff9999",
            )
            neg_btn.grid(row=i * 2 + 7, column=0, padx=5, pady=5)
            neg_btn.bind(
                "<ButtonPress-1>", lambda e, k=neg_key: self._on_button_press(k)  # type: ignore[misc]
            )
            neg_btn.bind(
                "<ButtonRelease-1>", lambda e, k=neg_key: self._on_button_release(k)  # type: ignore[misc]
            )

            # Positive button
            pos_btn = tk.Button(
                self.control_frame,
                text=pos_text,
                width=15,
                height=2,
                bg="#ccffcc",
                activebackground="#99ff99",
            )
            pos_btn.grid(row=i * 2 + 7, column=1, padx=5, pady=5)
            pos_btn.bind(
                "<ButtonPress-1>", lambda e, k=pos_key: self._on_button_press(k)  # type: ignore[misc]
            )
            pos_btn.bind(
                "<ButtonRelease-1>", lambda e, k=pos_key: self._on_button_release(k)  # type: ignore[misc]
            )

    def _create_joint_controls(self) -> None:
        """Create joint control buttons in the control frame."""
        # Speed control section
        speed_frame = ttk.LabelFrame(
            self.control_frame, text="Speed Controls", padding="5"
        )
        speed_frame.grid(
            row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10)  # type: ignore[arg-type]
        )

        # Joint speed control
        ttk.Label(
            speed_frame, text="Joint Speed (°/s):", font=("Arial", 9, "bold")
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.joint_speed_var = tk.StringVar(value=str(self.joint_speed))
        joint_speed_entry = ttk.Entry(
            speed_frame, textvariable=self.joint_speed_var, width=8
        )
        joint_speed_entry.grid(row=0, column=1, padx=(0, 5))
        joint_speed_entry.bind("<Return>", lambda e: self._update_joint_speed())

        # Joint buttons (J1-J6) with +/- for each
        joint_controls = [
            ("Joint 1", "j1_neg", "j1_pos", "← -", "+ →"),
            ("Joint 2", "j2_neg", "j2_pos", "← -", "+ →"),
            ("Joint 3", "j3_neg", "j3_pos", "← -", "+ →"),
            ("Joint 4", "j4_neg", "j4_pos", "← -", "+ →"),
            ("Joint 5", "j5_neg", "j5_pos", "← -", "+ →"),
            ("Joint 6", "j6_neg", "j6_pos", "← -", "+ →"),
        ]

        for i, (label, neg_key, pos_key, neg_text, pos_text) in enumerate(
            joint_controls
        ):
            ttk.Label(self.control_frame, text=label, font=("Arial", 10, "bold")).grid(
                row=i * 2 + 1, column=0, columnspan=2, sticky=tk.W, pady=(10, 2)
            )

            # Negative button
            neg_btn = tk.Button(
                self.control_frame,
                text=neg_text,
                width=15,
                height=2,
                bg="#ffcccc",
                activebackground="#ff9999",
            )
            neg_btn.grid(row=i * 2 + 2, column=0, padx=5, pady=5)
            neg_btn.bind(
                "<ButtonPress-1>", lambda e, k=neg_key: self._on_button_press(k)  # type: ignore[misc]
            )
            neg_btn.bind(
                "<ButtonRelease-1>", lambda e, k=neg_key: self._on_button_release(k)  # type: ignore[misc]
            )

            # Positive button
            pos_btn = tk.Button(
                self.control_frame,
                text=pos_text,
                width=15,
                height=2,
                bg="#ccffcc",
                activebackground="#99ff99",
            )
            pos_btn.grid(row=i * 2 + 2, column=1, padx=5, pady=5)
            pos_btn.bind(
                "<ButtonPress-1>", lambda e, k=pos_key: self._on_button_press(k)  # type: ignore[misc]
            )
            pos_btn.bind(
                "<ButtonRelease-1>", lambda e, k=pos_key: self._on_button_release(k)  # type: ignore[misc]
            )

    def _update_linear_speed(self) -> None:
        """Update linear speed from user input."""
        try:
            new_speed = float(self.linear_speed_var.get())
            if new_speed > 0:
                self.linear_speed = new_speed
                self.status_label.config(
                    text=f"Linear speed updated to {new_speed} mm/s", foreground="blue"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
            else:
                self.status_label.config(
                    text="Speed must be positive", foreground="red"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
        except ValueError:
            self.status_label.config(text="Invalid speed value", foreground="red")
            self.root.after(
                2000, lambda: self.status_label.config(text="Ready", foreground="green")
            )
            self.linear_speed_var.set(str(self.linear_speed))  # Reset to current value

    def _update_angular_speed(self) -> None:
        """Update angular speed from user input."""
        try:
            new_speed = float(self.angular_speed_var.get())
            if new_speed > 0:
                self.angular_speed = new_speed
                self.status_label.config(
                    text=f"Angular speed updated to {new_speed}°/s", foreground="blue"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
            else:
                self.status_label.config(
                    text="Speed must be positive", foreground="red"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
        except ValueError:
            self.status_label.config(text="Invalid speed value", foreground="red")
            self.root.after(
                2000, lambda: self.status_label.config(text="Ready", foreground="green")
            )
            self.angular_speed_var.set(
                str(self.angular_speed)
            )  # Reset to current value

    def _update_joint_speed(self) -> None:
        """Update joint speed from user input."""
        try:
            new_speed = float(self.joint_speed_var.get())
            if new_speed > 0:
                self.joint_speed = new_speed
                self.status_label.config(
                    text=f"Joint speed updated to {new_speed}°/s", foreground="blue"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
            else:
                self.status_label.config(
                    text="Speed must be positive", foreground="red"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
        except ValueError:
            self.status_label.config(text="Invalid speed value", foreground="red")
            self.root.after(
                2000, lambda: self.status_label.config(text="Ready", foreground="green")
            )
            self.joint_speed_var.set(str(self.joint_speed))  # Reset to current value

    def _update_gripper_speed(self) -> None:
        """Update gripper speed from user input."""
        try:
            new_speed = float(self.gripper_speed_var.get())
            if new_speed > 0:
                self.gripper_speed = new_speed
                self.status_label.config(
                    text=f"Gripper speed updated to {new_speed}°/s", foreground="blue"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
            else:
                self.status_label.config(
                    text="Speed must be positive", foreground="red"
                )
                self.root.after(
                    2000,
                    lambda: self.status_label.config(text="Ready", foreground="green"),
                )
        except ValueError:
            self.status_label.config(text="Invalid speed value", foreground="red")
            self.root.after(
                2000, lambda: self.status_label.config(text="Ready", foreground="green")
            )
            self.gripper_speed_var.set(
                str(self.gripper_speed)
            )  # Reset to current value

    def _on_home_pressed(self) -> None:
        """Called when home button is pressed."""
        self.status_label.config(text="Moving to home...", foreground="blue")
        self.root.after(
            2000, lambda: self.status_label.config(text="Ready", foreground="green")
        )

        self.command_queue.put(("reset", None, None))

    def _on_stop_resume_pressed(self) -> None:
        """Called when stop/resume button is pressed."""
        if self.robot.is_robot_enabled():
            self.status_label.config(text="Stopping robot...", foreground="orange")
            self.command_queue.put(("graceful_stop", None, None))
        else:
            self.status_label.config(text="Resuming robot...", foreground="orange")
            self.command_queue.put(("resume", None, None))

        self.root.after(
            2000, lambda: self.status_label.config(text="Ready", foreground="green")
        )

    def command_handler(self) -> None:
        """Handle commands from GUI in separate thread."""
        dt = 1.0 / self.control_update_rate

        while self.running.is_set():
            try:
                # Check for discrete commands (reset, stop, enable, disable)
                try:
                    command_type, _, _ = self.command_queue.get_nowait()

                    if command_type == "reset":
                        # Move to home position
                        self.robot.move_to_home()

                    elif command_type == "stop":
                        # Graceful stop
                        self.robot.graceful_stop()

                    elif command_type == "graceful_stop":
                        # Graceful stop
                        self.robot.graceful_stop()

                    elif command_type == "resume":
                        # Resume robot
                        self.robot.resume_robot()

                except queue.Empty:
                    pass

                # Calculate continuous movement based on pressed buttons
                linear_delta = np.zeros(3)
                angular_delta = np.zeros(3)
                joint_delta = np.zeros(6)
                gripper_delta = 0.0

                # Get current control mode
                current_mode = self.robot.get_control_mode()

                if current_mode == PiperController.ControlMode.END_EFFECTOR:
                    # Linear movements (X, Y, Z)
                    if self.pressed_buttons["x_pos"]:
                        linear_delta[0] += self.linear_speed * dt
                    if self.pressed_buttons["x_neg"]:
                        linear_delta[0] -= self.linear_speed * dt

                    if self.pressed_buttons["y_pos"]:
                        linear_delta[1] += self.linear_speed * dt
                    if self.pressed_buttons["y_neg"]:
                        linear_delta[1] -= self.linear_speed * dt

                    if self.pressed_buttons["z_pos"]:
                        linear_delta[2] += self.linear_speed * dt
                    if self.pressed_buttons["z_neg"]:
                        linear_delta[2] -= self.linear_speed * dt

                    # Angular movements (Roll, Pitch, Yaw)
                    if self.pressed_buttons["roll_pos"]:
                        angular_delta[0] += self.angular_speed * dt
                    if self.pressed_buttons["roll_neg"]:
                        angular_delta[0] -= self.angular_speed * dt

                    if self.pressed_buttons["pitch_pos"]:
                        angular_delta[1] += self.angular_speed * dt
                    if self.pressed_buttons["pitch_neg"]:
                        angular_delta[1] -= self.angular_speed * dt

                    if self.pressed_buttons["yaw_pos"]:
                        angular_delta[2] += self.angular_speed * dt
                    if self.pressed_buttons["yaw_neg"]:
                        angular_delta[2] -= self.angular_speed * dt

                    # Apply end-effector movement if any button is pressed
                    if np.any(linear_delta != 0) or np.any(angular_delta != 0):
                        if self.debug_mode:
                            print(
                                f"End-effector movement: linear={linear_delta}, angular={angular_delta}"
                            )
                        self.robot.update_target_pose(linear_delta, angular_delta)

                elif current_mode == PiperController.ControlMode.JOINT_SPACE:
                    # Joint movements (J1-J6)
                    if self.pressed_buttons["j1_pos"]:
                        joint_delta[0] += self.joint_speed * dt
                    if self.pressed_buttons["j1_neg"]:
                        joint_delta[0] -= self.joint_speed * dt

                    if self.pressed_buttons["j2_pos"]:
                        joint_delta[1] += self.joint_speed * dt
                    if self.pressed_buttons["j2_neg"]:
                        joint_delta[1] -= self.joint_speed * dt

                    if self.pressed_buttons["j3_pos"]:
                        joint_delta[2] += self.joint_speed * dt
                    if self.pressed_buttons["j3_neg"]:
                        joint_delta[2] -= self.joint_speed * dt

                    if self.pressed_buttons["j4_pos"]:
                        joint_delta[3] += self.joint_speed * dt
                    if self.pressed_buttons["j4_neg"]:
                        joint_delta[3] -= self.joint_speed * dt

                    if self.pressed_buttons["j5_pos"]:
                        joint_delta[4] += self.joint_speed * dt
                    if self.pressed_buttons["j5_neg"]:
                        joint_delta[4] -= self.joint_speed * dt

                    if self.pressed_buttons["j6_pos"]:
                        joint_delta[5] += self.joint_speed * dt
                    if self.pressed_buttons["j6_neg"]:
                        joint_delta[5] -= self.joint_speed * dt

                    # Apply joint movement if any button is pressed
                    if np.any(joint_delta != 0):
                        if self.debug_mode:
                            print(f"Joint movement: {joint_delta}")
                        self.robot.update_target_joint_angles(joint_delta)

                # Gripper movements (always available)
                if self.pressed_buttons["gripper_open"]:
                    gripper_delta += self.gripper_speed * dt
                if self.pressed_buttons["gripper_close"]:
                    gripper_delta -= self.gripper_speed * dt

                # Apply gripper movement if gripper button is pressed
                if gripper_delta != 0:
                    if self.debug_mode:
                        print(f"Gripper movement: {gripper_delta}")
                    self.robot.update_gripper_open_value(gripper_delta)

                # Sleep to maintain update rate
                time.sleep(dt)

            except Exception as e:
                print(f"Command handler error: {e}")

    def _update_status_display(self) -> None:
        """Update the robot status display with individual labels."""
        status = self.robot.get_robot_status()

        # Update stop/resume button text
        if status["enabled"]:
            self.stop_resume_button.config(text="Stop Robot")
            self.robot_state_label.config(text="ENABLED", foreground="green")
        else:
            self.stop_resume_button.config(text="Resume Robot")
            self.robot_state_label.config(text="DISABLED", foreground="red")

        # Update target pose
        target_pose = status["target_pose"]
        target_position = target_pose[:3, 3]
        target_rotation = Rotation.from_matrix(target_pose[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        self.target_pos_label.config(
            text=f"X: {target_position[0]:.1f}  Y: {target_position[1]:.1f}  Z: {target_position[2]:.1f}"
        )
        self.target_rot_label.config(
            text=f"RX: {target_rotation[0]:.1f}  RY: {target_rotation[1]:.1f}  RZ: {target_rotation[2]:.1f}"
        )

        # Update target joint angles
        target_joint_angles = status["target_joint_angles"]
        joint_text = f"J1: {target_joint_angles[0]:.1f}°  J2: {target_joint_angles[1]:.1f}°  J3: {target_joint_angles[2]:.1f}°"
        joint_text += f"\nJ4: {target_joint_angles[3]:.1f}°  J5: {target_joint_angles[4]:.1f}°  J6: {target_joint_angles[5]:.1f}°"
        self.target_joint_angles_label.config(text=joint_text, foreground="black")

        # Check if target gripper is valid
        gripper_open_value = status["gripper_open_value"]
        self.target_gripper_label.config(text=f"{gripper_open_value:.3f} (normalized)")

        # Update current pose (if available)
        if status["current_end_pose"] is not None:
            current_pose = status["current_end_pose"]
            current_position = current_pose[:3, 3]
            current_rotation = Rotation.from_matrix(current_pose[:3, :3]).as_euler(
                "xyz", degrees=True
            )
            self.current_pos_label.config(
                text=f"X: {current_position[0]:.1f}  Y: {current_position[1]:.1f}  Z: {current_position[2]:.1f}",
                foreground="black",
            )
            self.current_rot_label.config(
                text=f"RX: {current_rotation[0]:.1f}  RY: {current_rotation[1]:.1f}  RZ: {current_rotation[2]:.1f}",
                foreground="black",
            )
        else:
            self.current_pos_label.config(text="No feedback", foreground="gray")
            self.current_rot_label.config(text="No feedback", foreground="gray")

        # Update current gripper (if available)
        if status["current_gripper_open_value"] is not None:
            self.current_gripper_label.config(
                text=f"{status['current_gripper_open_value']:.3f} (normalized)",
                foreground="black",
            )
        else:
            self.current_gripper_label.config(text="No feedback", foreground="gray")

        # Update joint angles (if available)
        if status["current_joint_angles"] is not None:
            joint_angles = status["current_joint_angles"]
            joint_text = f"J1: {joint_angles[0]:.1f}°  J2: {joint_angles[1]:.1f}°  J3: {joint_angles[2]:.1f}°"
            joint_text += f"\nJ4: {joint_angles[3]:.1f}°  J5: {joint_angles[4]:.1f}°  J6: {joint_angles[5]:.1f}°"
            self.joint_angles_label.config(text=joint_text, foreground="black")
        else:
            self.joint_angles_label.config(text="No feedback", foreground="gray")

    def _schedule_status_update(self) -> None:
        """Schedule the next status update."""
        self._update_status_display()
        # Update every 500ms
        self.root.after(500, self._schedule_status_update)

    def run(self) -> None:
        """Run the GUI and control loops."""
        print("\nStarting GUI Robot Control...")
        print("Close the GUI window to exit.")
        print("-" * 60)

        # Start command handler thread
        command_thread = threading.Thread(target=self.command_handler, daemon=True)
        command_thread.start()

        # Start robot control loop thread
        robot_thread = threading.Thread(target=self.robot.control_loop, daemon=True)
        robot_thread.start()

        # Start status updates
        self._schedule_status_update()

        try:
            # Run GUI main loop (blocking)
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        print("\nCleaning up...")
        self.running.clear()

        # Cleanup GUI
        if hasattr(self, "root"):
            try:
                if self.root.winfo_exists():
                    self.root.destroy()
            except Exception as e:
                print(f"Error destroying GUI: {e}")

        # Cleanup robot
        if hasattr(self, "robot"):
            self.robot.cleanup()

        print("✓ Cleanup completed")


def main() -> None:
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="GUI Robot Control")
    parser.add_argument(
        "--can", type=str, default="can0", help="CAN interface (default: can0)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Robot control rate in Hz (default: 100.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Create and run the controller
    controller = GUIRobotControl(
        can_interface=args.can, robot_rate=args.rate, debug_mode=args.debug
    )

    controller.run()


if __name__ == "__main__":
    main()
