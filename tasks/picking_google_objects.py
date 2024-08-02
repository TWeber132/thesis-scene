import os

import numpy as np
from tasks.task import Task
from primitives.pick_and_place import Pick
from agents.oracle_agent import OracleAgent
from scipy.signal import correlate
from tasks import utils
import matplotlib.pyplot as plt

import pybullet as p
import time


class PickingSeenGoogleObjectsSeq(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 1
        self.n_tries = 50
        self.primitive = Pick()
        self.lang_template = "{act} {obj}"
        self.obj_names = self.get_object_names()
        self.act_names = self.primitive.get_action_names()
        self.task_completed_desc = "done picking objects."
        self.loaded_obj_names = {}
        self.best_yaws = []  # cache for oracle
        self.pick_masks = {}  # cache for oracle

    def get_object_names(self):
        return {
            'train': [
                'alarm clock',
                'android toy',
                # 'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                # 'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                # 'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                # 'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                # 'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                # 'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                # 'scissors',
                'screwdriver',
                'silver tape',
                # 'spatula with purple head',
                'spiderman figure',
                # 'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'valid': [
                'alarm clock',
                'android toy',
                # 'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                # 'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                # 'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                # 'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                # 'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                # 'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                # 'scissors',
                'screwdriver',
                'silver tape',
                # 'spatula with purple head',
                'spiderman figure',
                # 'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
            'test': [
                'alarm clock',
                'android toy',
                # 'ball puzzle',
                'black and blue sneakers',
                'black boot with leopard print',
                'black fedora',
                'black razer mouse',
                'black sandal',
                'black shoe with green stripes',
                'black shoe with orange stripes',
                'brown fedora',
                'bull figure',
                'butterfinger chocolate',
                'c clamp',
                'can opener',
                'crayon box',
                'dinosaur figure',
                'dog statue',
                'frypan',
                # 'green and white striped towel',
                'grey soccer shoe with cleats',
                'hammer',
                'hard drive',
                # 'honey dipper',
                'light brown boot with golden laces',
                'lion figure',
                # 'magnifying glass',
                'mario figure',
                'nintendo 3ds',
                'nintendo cartridge',
                'office depot box',
                'orca plush toy',
                'pepsi gold caffeine free box',
                'pepsi max box',
                'pepsi next box',
                'pepsi wild cherry box',
                'porcelain cup',
                # 'porcelain salad plate',
                'porcelain spoon',
                'purple tape',
                'red and white flashlight',
                # 'red and white striped towel',
                'red cup',
                'rhino figure',
                'rocket racoon figure',
                # 'scissors',
                'screwdriver',
                'silver tape',
                # 'spatula with purple head',
                'spiderman figure',
                # 'tablet',
                'toy school bus',
                'toy train',
                'unicorn toy',
                'white razer mouse',
                'yoshi figure',
            ],
        }

    def reset(self, env):
        super().reset(env)

        # object names
        obj_names = self.obj_names[self.mode]
        act_names = self.act_names[self.mode]

        # Add Google Scanned Objects to scene.
        obj_uids = []
        obj_descs = []
        act_descs = []

        n_objs = np.random.randint(1, 6)
        size = (0.1, 0.1, 0.1)
        obj_scale = 0.5
        obj_template = 'google/object-template.urdf'
        chosen_objs = self.choose_objects(obj_names, n_objs)
        chosen_acts = self.choose_actions(act_names, n_objs)
        self.loaded_obj_names = {}

        for i in range(n_objs):

            pose = self.get_random_pose(env, size)

            # Add object only if valid pose found.
            if pose[0] is not None:
                # Initialize with a slightly tilted pose so that the objects aren't always erect.
                slight_tilt = utils.q_mult(
                    pose[1], (-0.1736482, 0, 0, 0.9848078))
                ps = ((pose[0][0], pose[0][1], pose[0][2]+0.05), slight_tilt)

                obj_name = chosen_objs[i]
                act_name = chosen_acts[i]
                obj_name_with_underscore = obj_name.replace(" ", "_")
                mesh_file = os.path.join(self.assets_root,
                                         'google',
                                         'meshes_fixed',
                                         f'{obj_name_with_underscore}.obj')
                texture_file = os.path.join(self.assets_root,
                                            'google',
                                            'textures',
                                            f'{obj_name_with_underscore}.png')

                try:
                    replace = {'FNAME': (mesh_file,),
                               'SCALE': [obj_scale, obj_scale, obj_scale],
                               'COLOR': (0.2, 0.2, 0.2)}
                    urdf = self.fill_template(obj_template, replace)
                    obj_uid = env.add_object(urdf, ps)
                    if os.path.exists(urdf):
                        os.remove(urdf)
                    obj_uids.append(obj_uid)

                    texture_id = p.loadTexture(texture_file)
                    p.changeVisualShape(
                        obj_uid, -1, textureUniqueId=texture_id)
                    p.changeVisualShape(obj_uid, -1, rgbaColor=[1, 1, 1, 1])

                    obj_descs.append(obj_name)
                    act_descs.append(act_name)
                    self.loaded_obj_names[obj_uid] = obj_name_with_underscore

                except Exception as e:
                    print("Failed to load Google Scanned Object in PyBullet")
                    print(obj_name_with_underscore, mesh_file, texture_file)
                    print(f"Exception: {e}")

        self.set_goals(obj_uids, obj_descs, act_descs)

        for i in range(480):
            p.stepSimulation()

    def choose_objects(self, object_names, k):
        return np.random.choice(object_names, k, replace=False)

    def choose_actions(self, action_names, k):
        return np.random.choice(action_names, k, replace=False)

    def set_goals(self, obj_uids, obj_descs, act_descs):
        n_pick_objs = 1
        obj_uids = obj_uids[:n_pick_objs]

        for obj_idx, obj_uid in enumerate(obj_uids):
            self.goals.append(
                {"obj_uid": obj_uid, "max_reward": (1/len(obj_uids))})
            self.lang_goals.append(
                self.lang_template.format(act=act_descs[obj_idx], obj=obj_descs[obj_idx]))

    def reward(self, env):
        """Get delta rewards for current timestep.

        Returns:
          A tuple consisting of the scalar (delta) reward, plus `extras`
            dict which has extra task-dependent info from the process of
            computing rewards that gives us finer-grained details. Use
            `extras` for further data analysis.
        """
        # Unpack next goal step.
        obj_uid = self.goals[0]["obj_uid"]
        max_reward = self.goals[0]["max_reward"]

        def obj_airborne(obj_uid):
            obj_z = p.getBasePositionAndOrientation(obj_uid)[0][2]
            if obj_z > 0.2:
                return True
            return False

        # Move to next goal step if current goal step is complete.
        if obj_airborne(obj_uid) and env.robot.ee.obj_grasped(obj_uid):
            self.goals.pop(0)
            self.lang_goals.pop(0)
            self._rewards += max_reward

        return self._rewards

    # -------------------------------------------------------------------------
    # Oracle Agent
    # -------------------------------------------------------------------------

    def create_oracle_agent(self, env) -> OracleAgent:

        def act(obs, info):
            t0 = time.time()

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            _, hmap, obj_mask = self.get_true_image(env)

            # Unpack next goal step.
            obj_uid = self.goals[0]["obj_uid"]

            # Determine best gripper yaw angles
            if obs is not None:
                step_size = -2  # Clock wise
                self.pick_masks = {}
                valid_yaws = {}

                """    -90
                    -> * | *
                  -      |
                0 -______|_______
                  -      |    
                   -     |  
                     - * | *       Gripper (*)
                        90
                """
                for yaw in range(90, -90, step_size):
                    # NOTE: The gripper filter heavily relies on the chosen end effector of the robot
                    gripper_filter = env.robot.ee.get_gripper_filter(yaw)
                    # plt.imshow(gripper_filter)
                    # plt.show()
                    obj_mask_fil = correlate(
                        obj_mask, gripper_filter, mode='same')
                    obj_mask_fil = np.clip(obj_mask_fil, 0, 255)
                    # plt.imshow(obj_mask_fil)
                    # plt.show()
                    pick_mask = np.uint8(obj_mask_fil == obj_uid)
                    pick_mask_score = np.sum(pick_mask)
                    if pick_mask_score > 0:
                        valid_yaws[yaw] = pick_mask_score
                        self.pick_masks[yaw] = pick_mask
                yaws = [yaw for yaw, score in sorted(
                    valid_yaws.items(), key=lambda item: item[1], reverse=True)]

                # Trigger task reset if no object is visible.
                if len(yaws) == 0:
                    self.goals = []
                    self.lang_goals = []
                    print('Object for pick is not visible. Skipping demonstration.')
                    return

                # Get picking yaw
                if len(yaws) < 20:
                    self.best_yaws = yaws[:]
                else:
                    self.best_yaws = yaws[:20]
            pick_yaw = np.random.choice(self.best_yaws)
            # Get picking pitch
            pick_pitch = np.random.randint(-10, 10)
            # Get picking pix
            pick_prob = np.float32(self.pick_masks[pick_yaw])
            pick_pix = utils.sample_distribution(pick_prob)
            # Get pick pose
            filter_to_pick_pos = (0.0, 0.0, 0.0)
            filter_to_pick_rot = p.getQuaternionFromEuler(
                (np.pi, 0.0, np.pi/2))
            filter_to_pick_pose = (filter_to_pick_pos, filter_to_pick_rot)
            pick_pos = np.array(utils.pix_to_xyz(
                pick_pix, hmap, self.bounds, self.pix_size))
            pick_rot = p.getQuaternionFromEuler(
                (0.0, pick_pitch*np.pi/180, pick_yaw*np.pi/180))
            pick_pose = (pick_pos, pick_rot)
            pick_pose = p.multiplyTransforms(*pick_pose, *filter_to_pick_pose)
            # Chose most feasible z for pick pose
            obj_z = p.getBasePositionAndOrientation(obj_uid)[0][2]
            pick_pose_z = pick_pose[0][2]
            if obj_z < pick_pose_z:
                pick_pose = ((pick_pose[0][0], pick_pose[0][1], obj_z),
                             (pick_pose[1][0], pick_pose[1][1], pick_pose[1][2], pick_pose[1][3]))

            print(f"Oracle took: {round((time.time()-t0), 4)} s")
            return [pick_pose]

        def act_random(obs, info):
            t0 = time.time()

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            _, hmap, obj_mask = self.get_true_image(env)

            # Unpack next goal step.
            obj_uid = self.goals[0]["obj_uid"]

            # Get pick_mask
            pick_mask = np.uint8(obj_mask == obj_uid)
            # Trigger task reset if no object is visible.
            if np.sum(pick_mask) == 0:
                self.goals = []
                self.lang_goals = []
                print('Object for pick is not visible. Skipping demonstration.')
                return

            # Get pick position
            pick_prob = np.float32(pick_mask)
            pick_pix = utils.sample_distribution(pick_prob)
            pick_pos = np.array(utils.pix_to_xyz(
                pick_pix, hmap, self.bounds, self.pix_size))

            # Get random orientation
            min_azimuth = -np.pi / 2
            max_azimuth = np.pi / 2
            min_polar = 0
            max_polar = np.pi / 4
            azimuth = np.random.uniform(
                min_azimuth, max_azimuth)
            cos_polar = np.random.uniform(
                np.cos(max_polar), np.cos(min_polar))
            polar = np.arccos(cos_polar)

            # Generate poses from pix and random orientation
            # NOTE: radius is almost 0 because radius of 0 would lead to xyz = (0, 0, 0) which is bad if you want to get a direction of z_axis
            pick_pose = utils.get_pose_on_sphere(
                azimuth, polar, radius=1e-6, sph_pos=pick_pos)
            # Chose most feasible z for pick pose
            obj_z = p.getBasePositionAndOrientation(obj_uid)[0][2]
            pick_pose_z = pick_pose[0][2]
            if obj_z < pick_pose_z:
                pick_pose = ((pick_pose[0][0], pick_pose[0][1], obj_z),
                             (pick_pose[1][0], pick_pose[1][1], pick_pose[1][2], pick_pose[1][3]))

            print(f"Random oracle took: {round((time.time()-t0), 4)} s")
            return [pick_pose]

        return OracleAgent(act)
