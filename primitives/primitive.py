import numpy as np

from typing import Dict
from abc import ABC, abstractmethod


class Primitive(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.act_names = {}
        self.trajectory = []

    @abstractmethod
    def __call__(self, robot, action) -> bool:
        """Once the derived primitive is called it will execute the motion defined in each secial case.
        """
        # The trajectory has to be cleared everytime a new primitive is called
        self.trajectory = []
        # robot.do_things_with(action)

    def get_action_names(self) -> Dict:
        return self.act_names

    def save_last_trajectory_to_list(self, robot, max_elements: int = 150):
        """Saves last robot trajectory to a list "self.trajectory" and extends the list with each new trajectory. List has to be reset for every new primitive call.

        Args:
            robot (UR10E_Robotiq140): the robot that drove the trajectory
            n_elements (int, optional): the number of elements/poses each trajectory should at most consist of, number can be lower when trajectory does not have enough elements. Defaults to 150.
        """
        n_elements = len(robot.last_trajectory)
        n_elements = min(max_elements, n_elements)
        idx = np.round(np.linspace(
            0, n_elements - 1, n_elements)).astype(np.int32)
        robot_last_trajectory = np.array(
            robot.last_trajectory, dtype=object)
        self.trajectory.extend(
            [traj for traj in robot_last_trajectory[idx]])
