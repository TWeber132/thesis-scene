import numpy as np

from .robot import Robot
from ..end_effectors.robotiq140 import Robotiq140


class UR10E(Robot):
    def __init__(self, assets_root, env, env_uid, base_joint_id) -> None:
        super().__init__(assets_root, env, env_uid, base_joint_id)
        self.home_j = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        self.robot_urdf_path = "robot/ur10e.urdf"  # Relative to assets_root
        self.load_robot()

        # URDF specific
        # Attach the end effector to
        self.ee_joint_name = "realsense_mount_base_joint"
        # Calculate ik to (including end effector)
        self.tcp_joint_name = "tcp_joint"
        self.check_joints()
        assert len(self.home_j) == self.n_joints

        # Initialize Endeffector
        self.ee = Robotiq140(assets_root=assets_root, env=self.env,
                             robot_uid=self.uid, base_joint_id=self.ee_joint_id)
