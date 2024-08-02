import numpy as np
from .base import Primitive
from ..tasks import utils


class Push(Primitive):
    def __call__(self, robot, action) -> bool:
        """Execute pushing primitive.

        Args:
        movej: function to move robot joints.
        movep: function to move robot end effector pose.
        ee: robot end effector.
        pose0: SE(3) starting pose.
        pose1: SE(3) ending pose.

        Returns:
        timeout: robot movement timed out if True.
        """

        # Adjust push start and end positions.
        pos0 = np.float32((pose0[0][0], pose0[0][1], 0.005))
        pos1 = np.float32((pose1[0][0], pose1[0][1], 0.005))
        vec = np.float32(pos1) - np.float32(pos0)
        length = np.linalg.norm(vec)
        vec = vec / length
        pos0 -= vec * 0.02
        pos1 -= vec * 0.05

        # Align spatula against push direction.
        theta = np.arctan2(vec[1], vec[0])
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        over0 = (pos0[0], pos0[1], 0.31)
        over1 = (pos1[0], pos1[1], 0.31)

        # Execute push.
        timeout = robot.movep((over0, rot))
        timeout |= robot.movep((pos0, rot))
        n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
        for _ in range(n_push):
            target = pos0 + vec * n_push * 0.01
            timeout |= robot.movep((target, rot), speed=0.003)
        timeout |= robot.movep((pos1, rot), speed=0.003)
        timeout |= robot.movep((over1, rot))
        return timeout
