"""PyBullet utilities for loading assets."""
import os
import six
from dataclasses import dataclass
import pybullet as p


# BEGIN GOOGLE-EXTERNAL
def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    # Handles most general file open case.
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass

# END GOOGLE-EXTERNAL


@dataclass(frozen=True)
class JointInfo:
    """Stores the joint information of the robot in a more accesible format"""
    id: int
    name: str
    max_force: float
    max_velocity: float


def get_joint_id(body_uid: int, joint_name: str) -> int:
    _n_urdf_joints = p.getNumJoints(body_uid)
    _urdf_joints_info = [p.getJointInfo(body_uid, i)
                         for i in range(_n_urdf_joints)]
    joint_id = [j[0] for j in _urdf_joints_info if j[1].decode(
        "utf-8") == joint_name][0]
    return joint_id
