import os
import numpy as np
import pybullet as p
from scipy.ndimage import rotate

from typing import Any
from abc import ABC, abstractmethod
from ..pybullet_utils.joint_info_list import JointInfoList


class EndEffector(ABC):
    assets_root: str
    env: Any
    robot_uid: int
    base_joint_id: int
    ee_urdf_path: str
    uid: int
    tcp_joint_name: int
    joint_info_list: list
    revolute_joint_list: list
    contact_joint_names: list
    contact_joint_ids: list
    gripper_filter: np.ndarray
    home_j: np.ndarray

    @abstractmethod
    def __init__(self, assets_root, env, robot_uid, base_joint_id) -> None:
        self.assets_root = assets_root
        self.env = env
        self.robot_uid = robot_uid
        self.base_joint_id = base_joint_id
        _, _, _, _, self.base_pos, self.base_rot = p.getLinkState(
            robot_uid, base_joint_id, computeForwardKinematics=True)
        self.contact_joint_ids = []
        self.create_gripper_filter()

    def load_ee(self) -> None:
        self.uid = p.loadURDF(os.path.join(self.assets_root, self.ee_urdf_path), self.base_pos, self.base_rot,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        p.createConstraint(parentBodyUniqueId=self.robot_uid,
                           parentLinkIndex=self.base_joint_id,
                           childBodyUniqueId=self.uid,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,
                           jointAxis=(0, 0, 0),
                           parentFramePosition=(0, 0, 0),
                           childFramePosition=(0, 0, 0))

    def check_joints(self) -> None:
        self.joint_info_list = JointInfoList(self.uid)
        self.revolute_joint_list = self.joint_info_list.get_revolute_joint_list()

        for joint_name in self.contact_joint_names:
            joint_id = self.joint_info_list.get_joint_id(joint_name)
            p.enableJointForceTorqueSensor(bodyUniqueId=self.uid,
                                           jointIndex=joint_id,
                                           enableSensor=1)
            self.contact_joint_ids.append(joint_id)

    @abstractmethod
    def open(self) -> bool:
        ...

    @abstractmethod
    def close(self) -> bool:
        ...

    def reset(self) -> None:
        if len(self.revolute_joint_list) == 0:
            return
        for idx, joint in enumerate(self.revolute_joint_list):
            p.resetJointState(self.uid, joint.id, self.home_j[idx])

    @abstractmethod
    def create_gripper_filter(self) -> None:
        self.gripper_filter = np.ones([1, 10], dtype=np.int32)

    def get_gripper_filter(self, yaw=0):
        gripper_filter = rotate(self.gripper_filter,
                                yaw, reshape=False, order=0)
        return gripper_filter

    def in_contact_with_something(self) -> bool:
        for joint_id in self.contact_joint_ids:
            contact_points = p.getContactPoints(
                bodyA=self.uid, linkIndexA=joint_id)
            if contact_points:
                for contact_point in contact_points:
                    obj_uid = contact_point[2]
                    # End effector in contact with something but not itself
                    if obj_uid != self.uid:
                        return True
        # End effector in contact with nothing or itself
        return False

    def in_contact_with_obj(self, obj_uid) -> bool:
        for joint_id in self.contact_joint_ids:
            contact_points = p.getContactPoints(
                bodyA=self.uid, linkIndexA=joint_id, bodyB=obj_uid, linkIndexB=-1)
            # End effector in contact with object
            if contact_points:
                return True
        # End effector not in contact with object
        return False
