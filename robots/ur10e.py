import numpy as np

from .robot import Robot


class UR10E(Robot):
    def __init__(self, assets_root, env, env_uid, base_joint_id) -> None:
        super().__init__(assets_root, env, env_uid, base_joint_id)
        self.home_j = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        self.robot_urdf_path = "robots/ur10e.urdf"  # Relative to assets_root
        self.load_robot()

        # URDF specific
        # Attach the end effector to
        self.robot_ee_joint_name = "realsense_mount_base_joint"
        self.check_joints()
        assert len(self.home_j) == len(self.revolute_joint_list)
