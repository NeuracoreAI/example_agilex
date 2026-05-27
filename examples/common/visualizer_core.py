import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf


class RobotVisualizerCore:
    """Handles the 3D rendering of the robot, ghost robot, and target frames."""

    def __init__(self, urdf_path: str) -> None:
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Load actual robot URDF
        urdf = yourdfpy.URDF.load(urdf_path)
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot_actual")

        # Load ghost robot URDF
        ghost_urdf = yourdfpy.URDF.load(urdf_path)
        self.ghost_robot_urdf = ViserUrdf(
            self.server,
            ghost_urdf,
            root_node_name="/robot_ghost",
            mesh_color_override=(1.0, 0.65, 0.0, 0.25),
        )

        self.controller_handle = None
        self.target_frame_handle = None
        self.rgb_image_handle = None

    def add_controller_visualization(self) -> None:
        self.controller_handle = self.server.scene.add_transform_controls(
            "/controller", scale=0.15, position=(0, 0, 0), wxyz=(1, 0, 0, 0)
        )

    def add_target_frame_visualization(self) -> None:
        self.target_frame_handle = self.server.scene.add_frame(
            "/target_goal", axes_length=0.1, axes_radius=0.003
        )

    def add_rgb_image_placeholder(self, height: int = 480, width: int = 640) -> None:
        if self.rgb_image_handle is None:
            dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
            self.rgb_image_handle = self.server.gui.add_image(
                dummy_image, label="RGB Camera", format="jpeg", jpeg_quality=85
            )

    def update_rgb_image(self, rgb_image: np.ndarray | None) -> None:
        if rgb_image is None:
            return
        if self.rgb_image_handle is None:
            self.add_rgb_image_placeholder(
                height=rgb_image.shape[0], width=rgb_image.shape[1]
            )
        self.rgb_image_handle.image = rgb_image

    def update_robot_pose(self, joint_config: np.ndarray) -> None:
        self.urdf_vis.update_cfg(joint_config)

    def update_ghost_robot_pose(self, joint_config: np.ndarray) -> None:
        if self.ghost_robot_urdf:
            self.ghost_robot_urdf.update_cfg(joint_config)

    def update_ghost_robot_visibility(self, flag: bool) -> None:
        if self.ghost_robot_urdf:
            self.ghost_robot_urdf.show_visual = flag

    def get_ghost_robot_visibility(self) -> bool:
        if self.ghost_robot_urdf:
            return self.ghost_robot_urdf.show_visual
        return False

    def set_ghost_robot_color(self, color: tuple[float, float, float, float]) -> None:
        if self.ghost_robot_urdf:
            self.ghost_robot_urdf.mesh_color_override = color

    def update_controller_visualization(self, transform: np.ndarray | None) -> None:
        if self.controller_handle is None or transform is None:
            return
        pos = transform[:3, 3]
        rot = Rotation.from_matrix(transform[:3, :3]).as_quat()
        self.controller_handle.position = tuple(pos)
        self.controller_handle.wxyz = (rot[3], rot[0], rot[1], rot[2])

    def update_target_visualization(self, transform: np.ndarray | None) -> None:
        if self.target_frame_handle is None or transform is None:
            return
        pos = transform[:3, 3]
        rot = Rotation.from_matrix(transform[:3, :3]).as_quat()
        self.target_frame_handle.position = tuple(pos)
        self.target_frame_handle.wxyz = (rot[3], rot[0], rot[1], rot[2])

    def stop(self) -> None:
        self.server.stop()
