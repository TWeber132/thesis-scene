"""Base Task class."""

import os
import random
import string
import tempfile
import cv2
import numpy as np
import pybullet as p

from ..environments import cameras
from ..primitives.pick_and_place import PickAndPlace
from ..agents.oracle_agent import OracleAgent
from ..tasks import utils


class Task():
    """Base Task class."""

    def __init__(self, info=None):
        self.from_info = info
        self.mode = 'train'
        self.sixdof = False
        self.primitive = PickAndPlace
        self.oracle_cams = cameras.Oracle.CONFIG

        # Evaluation epsilons (for pose evaluation metric).
        self.pos_eps = 0.01
        self.rot_eps = np.deg2rad(15)

        # Workspace bounds.
        self.pix_size = 0.003125
        self.bounds = np.array([[0.32, 0.98],
                                [-0.32, 0.32],
                                [-0.1, 0.3]])

        self.goals = []
        self.lang_goals = []
        self.task_completed_desc = "task completed."
        self.progress = 0
        self._rewards = 0

        self.assets_root = None

    def reset(self, env):
        if not self.assets_root:
            raise ValueError('assets_root must be set for task, '
                             'call set_assets_root().')
        self.goals = []
        self.lang_goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self._rewards = 0  # Cumulative returned rewards.

        if self.from_info:
            self.recreate_from_info()
            return

    def recreate_from_info(self):
        for key, item in self.from_info:
            self.add_object(item[-1], pose=(item[0], item[1]))
        print("Recreated Task from info")

    # -------------------------------------------------------------------------
    # Oracle Agent
    # -------------------------------------------------------------------------

    def create_oracle_agent(self, env) -> OracleAgent:
        # Define solution act methode
        def act(obs, info):
            raise NotImplementedError

        return OracleAgent(act)

    # -------------------------------------------------------------------------
    # Reward Function and Task Completion Metrics
    # -------------------------------------------------------------------------

    def reward(self):
        raise NotImplementedError

    def done(self):
        """Check if the task is done or has failed.

        Returns:
          True if the episode should be considered a success, which we
            use for measuring successes, which is particularly helpful for tasks
            where one may get successes on the very last time step, e.g., getting
            the cloth coverage threshold on the last alllowed action.
            However, for bag-items-easy and bag-items-hard (which use the
            'bag-items' metric), it may be necessary to filter out demos that did
            not attain sufficiently high reward in external code. Currently, this
            is done in `main.py` and its ignore_this_demo() method.
        """

        # # For tasks with self.metric == 'pose'.
        # if hasattr(self, 'goal'):
        # goal_done = len(self.goal['steps']) == 0  # pylint:
        # disable=g-explicit-length-test
        return (len(self.goals) == 0) or (self._rewards > 0.99)  # pylint: disable=g-explicit-length-test
        # return zone_done or defs_done or goal_done

    # -------------------------------------------------------------------------
    # Environment Helper Functions
    # -------------------------------------------------------------------------

    def is_match(self, pose0, pose1, symmetry):
        """Check if pose0 and pose1 match within a threshold."""

        # Get translational error.
        diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
        dist_pos = np.linalg.norm(diff_pos)

        # Get rotational error around z-axis (account for symmetries).
        diff_rot = 0
        if symmetry > 0:
            rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
            rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
            diff_rot = np.abs(rot0 - rot1) % symmetry
            if diff_rot > (symmetry / 2):
                diff_rot = symmetry - diff_rot

        return (dist_pos < self.pos_eps) and (diff_rot < self.rot_eps)

    def get_true_image(self, env):
        """Get RGB-D orthographic heightmaps and segmentation masks."""

        # Capture near-orthographic RGB-D images and segmentation masks.
        color, depth, segm = env.render_camera(self.oracle_cams[0])

        # Combine color with masks for faster processing.
        color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

        # Reconstruct real orthographic projection from point clouds.
        hmaps, cmaps = utils.reconstruct_heightmaps(
            [color], [depth], self.oracle_cams, self.bounds, self.pix_size)

        # Split color back into color and masks.
        cmap = np.uint8(cmaps)[0, Ellipsis, :3]
        hmap = np.float32(hmaps)[0, Ellipsis]
        mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
        return cmap, hmap, mask

    def get_random_pose(self, env, obj_size):
        """Get random collision-free object pose within workspace bounds."""

        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))

        _, hmap, obj_mask = self.get_true_image(env)

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))

        if np.sum(free) == 0:
            return None, None
        pix = utils.sample_distribution(np.float32(free))
        pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        # raise pos by obj_size / 2
        pos = (pos[0], pos[1], pos[2] + obj_size[2] / 2)
        theta = np.random.rand() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def get_lang_goal(self):
        if len(self.lang_goals) == 0:
            return self.task_completed_desc
        else:
            return self.lang_goals[0]

    def get_reward(self):
        return float(self._rewards)

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def fill_template(self, template, replace):
        """Read a file and replace key strings."""
        full_template_path = os.path.join(self.assets_root, template)
        with open(full_template_path, 'r') as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
        alphabet = string.ascii_lowercase + string.digits
        rname = ''.join(random.choices(alphabet, k=16))
        tmpdir = tempfile.gettempdir()
        template_filename = os.path.split(template)[-1]
        fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
        with open(fname, 'w') as file:
            file.write(fdata)
        return fname

    def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = np.random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return tuple(size)

    def get_box_object_points(self, obj):
        obj_shape = p.getVisualShapeData(obj)
        obj_dim = obj_shape[0][3]
        obj_dim = tuple(d for d in obj_dim)
        xv, yv, zv = np.meshgrid(
            np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
            np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
            np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
            sparse=False, indexing='xy')
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

    def get_mesh_object_points(self, obj):
        mesh = p.getMeshData(obj)
        mesh_points = np.array(mesh[1])
        mesh_dim = np.vstack(
            (mesh_points.min(axis=0), mesh_points.max(axis=0)))
        xv, yv, zv = np.meshgrid(
            np.arange(mesh_dim[0][0], mesh_dim[1][0], 0.02),
            np.arange(mesh_dim[0][1], mesh_dim[1][1], 0.02),
            np.arange(mesh_dim[0][2], mesh_dim[1][2], 0.02),
            sparse=False, indexing='xy')
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

    def color_random_brown(self, obj):
        shade = np.random.rand() + 0.5
        color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
        p.changeVisualShape(obj, -1, rgbaColor=color)

    def set_assets_root(self, assets_root):
        self.assets_root = assets_root
