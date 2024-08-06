from typing import Dict
import numpy as np


class Primitive:
    def __init__(self) -> None:
        self.trajectory = []

    def __call__(self, robot, action) -> bool:
        """Once the derived primitive is called it will execute the motion defined in each secial case.
        """
        self.trajectory = []
        raise NotImplementedError(
            "Primitive class can not be called directly!")

    def get_action_names(self) -> Dict:
        raise NotImplementedError(
            "Primitive class has no default action name")

    def save_last_robot_trajectory_to_list(self, robot, n_elements: int = 150):
        """Saves last robot trajectory to a list "self.trajectory" and extends the list with each new trajectory. List has to be reset for every new primitive call.

        Args:
            robot (UR10E_Robotiq140): the robot that drove the trajectory
            n_elements (int, optional): the number of elements/poses each trajectory should at most consist of, number can be lower when trajectory does not have enough elements. Defaults to 150.
        """
        has_n_elements = len(robot.last_movej_trajectory)
        n_elements = min(n_elements, has_n_elements)
        idx = np.round(np.linspace(
            0, has_n_elements - 1, n_elements)).astype(np.int32)
        robot_last_trajectory = np.array(
            robot.last_movej_trajectory, dtype=object)
        self.trajectory.extend(
            [traj for traj in robot_last_trajectory[idx]])
