import numpy as np
from .base import Primitive


class Pick(Primitive):
    def __init__(self) -> None:
        self.type = "pick"
        self.app_height = 0.45
        self.pre_height = 0.10
        self.trajectory = []

    def __call__(self, robot, action) -> bool:
        if len(action) != 1:
            raise ValueError(
                "Action for pick primitive has to contain only one [pick]pose")
        self.trajectory = []
        pick_pos = np.array(action[0][0], dtype=np.float32)
        pick_rot = np.array(action[0][1], dtype=np.float32)
        app_pos = pick_pos + np.array([0.0, 0.0, self.app_height])
        app_rot = np.array([0.707, 0.707, 0.0, 0.0])
        pre_pos = pick_pos + np.array([0.0, 0.0, self.pre_height])
        pre_rot = pick_rot

        timeout = robot.ee.open()
        timeout |= robot.movep((app_pos, app_rot), speed=0.01)
        self.save_last_robot_trajectory_to_list(robot=robot, n_elements=20)
        timeout |= robot.movep((pre_pos, pre_rot), speed=0.01)
        self.save_last_robot_trajectory_to_list(robot=robot, n_elements=15)
        timeout |= robot.movep((pick_pos, pick_rot), speed=0.001)
        self.save_last_robot_trajectory_to_list(robot=robot, n_elements=5)
        timeout |= robot.ee.grasp()
        # To save time
        if timeout:
            return timeout
        timeout |= robot.movep((app_pos, app_rot), speed=0.001)
        return timeout

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

    def get_action_names(self):
        return {
            'train': [
                'pick up',
                'grasp',
                'take',
                'lift',
                'fetch',
                'collect',
                'grab',
                'go get',
                'get',
                'raise',
                'obtain',
                'hold'
            ],
            'valid': [
                'pick up',
                'grasp',
                'take',
                'lift',
                'fetch',
                'collect',
                'grab',
                'go get',
                'get',
                'raise',
                'obtain',
                'hold'
            ],
            'test': [
                'pick up',
                'grasp',
                'take',
                'lift',
                'fetch',
                'collect',
                'grab',
                'go get',
                'get',
                'raise',
                'obtain',
                'hold'
            ]
        }


class Place(Primitive):
    def __init__(self) -> None:
        self.type = "place"
        self.app_height = 0.45
        self.pre_height = 0.10
        self.trajectory = []

    def __call__(self, robot, action) -> bool:
        if len(action) != 1:
            raise ValueError(
                "Action for pick primitive has to contain only one [pick]pose")
        place_pos = np.array(action[0][0], dtype=np.float32)
        place_rot = np.array(action[0][1], dtype=np.float32)
        app_pos = place_pos + np.array([0.0, 0.0, self.app_height])
        app_rot = np.array([0.707, 0.707, 0.0, 0.0])
        pre_pos = place_pos + np.array([0.0, 0.0, self.pre_height])
        pre_rot = place_rot

        timeout = robot.movep((app_pos, app_rot), speed=0.001)
        timeout |= robot.movep((place_pos, place_rot), speed=0.001)
        timeout |= robot.ee.open()
        timeout |= robot.movep((app_pos, app_rot), speed=0.001)
        return timeout


class PickAndPlace(Primitive):
    def __init__(self) -> None:
        self.type = "pick_and_place"
        self.pick = Pick(at_key="pose0")
        self.place = Place(at_key="pose1")
        self.trajectory = []

    def __call__(self, robot, action) -> bool:
        robot.home()
        robot.ee.open()
        timeout = self.pick(robot, action)
        self.trajectory = self.pick.trajectory
        successful_grasp = robot.ee.check_grasp()
        if successful_grasp and not timeout:
            timeout |= self.place(robot, action)
        else:
            timeout = True
            print("Warning: Grasp failed.")
        return timeout
