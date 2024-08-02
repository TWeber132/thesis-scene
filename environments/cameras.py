"""Camera configs."""

import numpy as np
import pybullet as p
from tasks.utils import get_pose_on_sphere


class RealSenseD415():
    """Default configuration RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

    # Set default camera poses.
    front_position = (0.8, 0, 1.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)

    # Default camera configs.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': front_position,
        'rotation': front_rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]


class NeRFCamera():
    # Mimic RealSense D415 RGB-D camera parameters.
    def __init__(self, pose) -> None:
        image_size = (480, 640)
        intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)
        depth_range = (0.01, 10)

        position = pose[0]
        rotation = pose[1]

        self.CONFIG = [{
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': position,
            'rotation': rotation,
            'zrange': depth_range,
            'noise': False
        }]


class NeRFCameraFactory():
    def __init__(self,
                 min_azimuth=-np.pi + np.pi / 6,
                 max_azimuth=np.pi - np.pi / 6,
                 min_polar=np.pi / 6,
                 max_polar=np.pi / 3,
                 radius=0.8,
                 sph_pos=np.array([0.655, 0.0, 0.0]),
                 resolution=(480, 640),
                 intrinsics=(450, 0, 320, 0, 450, 240, 0, 0, 1),
                 depth_range=(0.01, 10.0)):

        self.min_azimuth = min_azimuth
        self.max_azimuth = max_azimuth
        self.min_polar = min_polar
        self.max_polar = max_polar
        self.radius = radius
        self.sph_pos = sph_pos
        self.resolution = resolution
        self.intrinsics = intrinsics
        self.depth_range = depth_range

    def create(self):
        azimuth = np.random.uniform(self.min_azimuth, self.max_azimuth)
        cos_polar = np.random.uniform(
            np.cos(self.max_polar), np.cos(self.min_polar))
        polar = np.arccos(cos_polar)
        pose = get_pose_on_sphere(
            azimuth, polar, self.radius, sph_pos=self.sph_pos)
        return NeRFCamera(pose)


class Oracle():
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
    # position over the middle of the "board_link"
    position = (0.655, 0, 1000.)
    rotation = p.getQuaternionFromEuler((np.pi, 0, np.pi / 2))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (999.7, 1001.),
        'noise': False
    }]


class RS200Gazebo():
    """Gazebo Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (554.3826904296875, 0.0, 320.0, 0.0,
                  554.3826904296875, 240.0, 0.0, 0.0, 1.0)
    position = (0.5, 0, 1.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, np.pi / 2))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]


class KinectFranka():
    """Kinect Franka Camera"""

    # Near-orthographic projection.
    image_size = (424, 512)
    intrinsics = (365.57489013671875, 0.0, 257.5205078125, 0.0,
                  365.57489013671875, 205.26710510253906, 0.0, 0.0, 1.0)
    position = (1.082, -0.041, 1.027)
    rotation = p.getQuaternionFromEuler((-2.611, 0.010, 1.553))

    # Camera config.
    CONFIG = [{
        'image_size': image_size,
        'intrinsics': intrinsics,
        'position': position,
        'rotation': rotation,
        'zrange': (0.01, 10.),
        'noise': False
    }]


if __name__ == "__main__":
    camera_factory = NeRFCameraFactory()
    for i in range(50):
        camera = camera_factory.create()
