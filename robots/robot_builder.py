from typing import Type, Any

from .robot import Robot
from ..end_effectors.end_effector import EndEffector


class RobotBuilder:
    def __init__(self, env, robot, ee) -> None:
        self.robot = self._build_robot(env, robot, ee)

    def _build_robot(self, env, robot_type: Type[Robot], ee_type: Type[EndEffector]) -> Robot:
        robot = robot_type(assets_root=env.assets_root,
                           env=env,
                           env_uid=env.uid,
                           base_joint_id=env.env_robot_joint_id)
        ee = ee_type(assets_root=env.assets_root,
                     env=env,
                     robot_uid=robot.uid,
                     base_joint_id=robot.robot_ee_joint_id)
        robot.ee = ee
        robot.set_tcp_joint(ee.tcp_joint_name)
        return robot

    def get_robot(self) -> Robot:
        return self.robot
