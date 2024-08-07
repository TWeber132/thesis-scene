import os
import time
import numpy as np
import pybullet as p

from abc import ABC, abstractmethod
from typing import Any
from ..pybullet_utils.joint_info_list import JointInfoList


class Robot(ABC):
    assets_root: str
    env: Any
    robot_urdf_path: str
    uid: int
    joint_info_list: list
    revolute_joint_list: list
    robot_ee_joint_name: str
    robot_ee_joint_id: int
    ee: Any
    tcp_joint_id: int
    home_j: np.ndarray
    last_trajectory: list

    @abstractmethod
    def __init__(self, assets_root, env, env_uid, base_joint_id) -> None:
        self.assets_root = assets_root
        self.env = env
        _, _, _, _, self.base_pos, self.base_rot = p.getLinkState(
            env_uid, base_joint_id, computeForwardKinematics=True)
        self.last_trajectory = []

    def load_robot(self) -> None:
        self.uid = p.loadURDF(os.path.join(self.assets_root, self.robot_urdf_path), self.base_pos, self.base_rot,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

    def check_joints(self) -> None:
        self.joint_info_list = JointInfoList(self.uid)
        self.revolute_joint_list = self.joint_info_list.get_revolute_joint_list()

        self.robot_ee_joint_id = self.joint_info_list.get_joint_id(
            self.robot_ee_joint_name)

    def set_tcp_joint(self, tcp_joint_name):
        # Calculate ik to correct tcp joint (including end effector)
        # NOTE: tcp_joint_name depends on the chosen end effector
        self.tcp_joint_id = self.joint_info_list.get_joint_id(
            tcp_joint_name)

    def reset(self) -> None:
        for idx, joint in enumerate(self.revolute_joint_list):
            p.resetJointState(self.uid, joint.id, self.home_j[idx])
        self.ee.reset()

    def home(self) -> None:
        self.move_j(self.home_j)
        self.ee.open()

    def move_j(self, targ_j, speed=0.01, timeout=4.0) -> bool:
        if self.env.save_video:
            timeout = timeout * 50

        # Clear last trajectory list
        self.last_trajectory = []

        t0 = time.time()
        while (time.time() - t0) < timeout:
            curr_s = p.getJointStates(
                self.uid, [joint.id for joint in self.revolute_joint_list])
            curr_j = [current_state[0] for current_state in curr_s]
            curr_j = np.array(curr_j)

            # Save ee pose to last_trajectory:
            ee_pose = self.get_ee_pose()
            self.last_trajectory.append(ee_pose)

            diff_j = targ_j - curr_j
            if all(np.abs(diff_j) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diff_j)
            v = diff_j / norm if norm > 0 else 0
            step_j = curr_j + v * speed
            gains = np.ones(len(self.revolute_joint_list))
            p.setJointMotorControlArray(
                bodyIndex=self.uid,
                jointIndices=[joint.id for joint in self.revolute_joint_list],
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_j,
                positionGains=gains)

            self.env.step_counter += 1
            self.env.step_simulation()

        print(
            f'Warning: robot movej exceeded {timeout} second timeout. Skipping.')
        return True

    def move_p(self, pose, speed=0.01) -> bool:
        targ_j = self.solve_ik(pose)
        return self.move_j(targ_j, speed)

    def solve_ik(self, pose) -> np.ndarray:
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.uid,
            endEffectorLinkIndex=self.tcp_joint_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34] * 6,
            restPoses=np.float32(self.home_j).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def get_ee_pose(self) -> tuple:
        """This should be in "grippers.py" the moment the tcp link/joint no longer resides inside robot urdf for IK purposes"""
        _, _, _, _, base_pos, base_rot = p.getLinkState(
            self.uid, self.tcp_joint_id, computeForwardKinematics=True)
        return (np.array(base_pos), np.array(base_rot))
