from agents.base import Agent
import numpy as np
import random


class RandomAgent(Agent):
    def act(self, obs, info):
        pick_pose = ((random.uniform(0.0, 0.8), random.uniform(-0.5, 0.5), random.uniform(0.95, 1.2)),
                     (1.0, 1.0, 0.0, 0.0))
        place_pose = ((random.uniform(0.0, 0.8), random.uniform(-0.5, 0.5), random.uniform(0.95, 1.2)),
                      (1.0, 1.0, 0.0, 0.0))
        return {'pose0': pick_pose, 'pose1': place_pose}
