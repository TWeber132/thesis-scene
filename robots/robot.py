import os
import time
import numpy as np
import pybullet as p

from abc import ABC, abstractmethod
from typing import Any
from ..pybullet_utils import JointInfo


class Robot(ABC):
    assets_root: str
    env: Any
    robot_urdf_path: str
    uid: int
    joints_info: list
    n_joints: int
    ee_joint_name: str
    ee_joint_id: int
    ee: Any
    tcp_joint_name: str
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
        # Save all controllable joints
        _n_urdf_joints = p.getNumJoints(self.uid)
        _urdf_joints_info = [p.getJointInfo(self.uid, i)
                             for i in range(_n_urdf_joints)]
        self.joints_info = [JointInfo(j[0], j[1].decode("utf-8"), j[10], j[11])
                            for j in _urdf_joints_info if j[2] == p.JOINT_REVOLUTE]
        self.n_joints = len(self.joints_info)

        self.ee_joint_id = [j[0] for j in _urdf_joints_info if j[1].decode(
            "utf-8") == self.ee_joint_name][0]
        self.tcp_joint_id = [j[0] for j in _urdf_joints_info if j[1].decode(
            "utf-8") == self.tcp_joint_name][0]

    def reset(self) -> None:
        for i in range(self.n_joints):
            p.resetJointState(self.uid, self.joints_info[i].id, self.home_j[i])
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
                self.uid, [joint.id for joint in self.joints_info])
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
            gains = np.ones(self.n_joints)
            p.setJointMotorControlArray(
                bodyIndex=self.uid,
                jointIndices=[joint.id for joint in self.joints_info],
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_j,
                positionGains=gains)

            self.env.step_counter += 1
            self.env.step_simulation()

        print(
            f'Warning: robot movej exceeded {timeout} second timeout. Skipping.')
        return True

    def move_p(self, pose, speed=0.01) -> bool:
        # self.env.add_object(
        #     urdf='util/coordinate_axes.urdf', pose=pose, category='fixed')
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