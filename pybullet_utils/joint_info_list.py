"""PyBullet utilities for loading assets."""
from dataclasses import dataclass
import pybullet as p


@dataclass(frozen=True)
class JointInfo:
    id: int
    name: str
    type: int
    max_force: float
    max_velocity: float


class JointInfoList:
    """Stores the joint information of the robot in a more accesible format"""
    _joint_info_list: list

    def __init__(self, body_uid: int) -> None:
        n_joints = p.getNumJoints(body_uid)
        p_joint_info_list = [p.getJointInfo(
            body_uid, i) for i in range(n_joints)]
        self._joint_info_list = [JointInfo(j[0], j[1].decode(
            "utf-8"), j[2], j[10], j[11]) for j in p_joint_info_list]

    def __iter__(self):
        return iter(self._joint_info_list)

    def __len__(self):
        return len(self._joint_info_list)

    def __getitem__(self, idx):
        return self._joint_info_list[idx]

    def get_joint_id(self, joint_name: str) -> int:
        for joint_info in self._joint_info_list:
            if joint_info.name == joint_name:
                return joint_info.id
        raise RuntimeError(
            f"There is no joint named {joint_name}, found {[joint_info.name for joint_info in self._joint_info_list]}.")

    def get_revolute_joint_list(self) -> list:
        return [joint_info for joint_info in self._joint_info_list if joint_info.type == p.JOINT_REVOLUTE]
