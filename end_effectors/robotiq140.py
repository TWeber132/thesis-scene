import time
import numpy as np
import pybullet as p

from .end_effector import EndEffector


class Robotiq140(EndEffector):
    leading_joint_range: list
    leading_joint_id: int

    def __init__(self, assets_root, env, robot_uid, base_joint_id):
        super().__init__(assets_root, env, robot_uid, base_joint_id)
        self.home_j = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.leading_joint_range = [0.0, 0.7]

        self.ee_urdf_path = "robot/robotiq140.urdf"
        self.load_ee()

        # URDF specific
        self.contact_joint_names = ['left_inner_finger_pad_joint',
                                    'right_inner_finger_pad_joint']
        self.check_joints()
        assert len(self.home_j) == self.n_joints
        self.mimic_leading_joint()

    def mimic_leading_joint(self) -> None:
        leading_joint_name = 'left_outer_knuckle_joint'
        self.leading_joint_id = [
            joint.id for joint in self.joints_info if joint.name == leading_joint_name][0]

        following_joint_names_and_multiplier = {'right_outer_knuckle_joint': 1,
                                                'left_inner_knuckle_joint': 1,
                                                'right_inner_knuckle_joint': 1,
                                                'left_inner_finger_joint': -1,
                                                'right_inner_finger_joint': -1}
        following_joint_ids_and_multiplier = {
            joint.id: following_joint_names_and_multiplier[joint.name] for joint in self.joints_info if joint.name in following_joint_names_and_multiplier}

        for joint_id, multiplier in following_joint_ids_and_multiplier.items():
            c = p.createConstraint(parentBodyUniqueId=self.uid,
                                   parentLinkIndex=self.leading_joint_id,
                                   childBodyUniqueId=self.uid,
                                   childLinkIndex=joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            # Note: the mysterious `erp` is of EXTREME importance
            p.changeConstraint(c, gearRatio=multiplier, maxForce=10000, erp=1)

    def move_m(self) -> bool:
        # Moves leading joint until the measured moment exceeds a threshold
        targ_m = 0.05
        timeout = 2.0
        t0 = time.time()
        while (time.time() - t0) < timeout:
            # Check if the combined moment is reached
            for joint_id in self.contact_joint_ids:
                curr_ft = p.getJointState(
                    bodyUniqueId=self.uid, jointIndex=joint_id)[2]
                _, _, _, curr_mx, curr_my, curr_mz = curr_ft
                sum_curr_m = abs(curr_mx) + abs(curr_my) + abs(curr_mz)
                if (sum_curr_m > targ_m):
                    return False

            p.setJointMotorControl2(bodyUniqueId=self.uid,
                                    jointIndex=self.leading_joint_id,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=2.0,
                                    force=1.0)
            self.env.step_simulation()
        print(
            f'Warning: gripper move_m exceeded {timeout} second timeout. Skipping.')
        return True

    def move_j(self, targ_j: float = 0.0) -> bool:
        # Moves leading joint to target joint position
        timeout = 2.0
        t0 = time.time()
        while (time.time() - t0) < timeout:
            targ_j = np.clip(targ_j, *self.leading_joint_range)
            # Control the mimic gripper joint(s)
            curr_j = p.getJointState(
                bodyUniqueId=self.uid, jointIndex=self.leading_joint_id)[0]
            diff_j = targ_j - curr_j
            if abs(diff_j) < 1e-3:
                return False
            p.setJointMotorControl2(bodyUniqueId=self.uid,
                                    jointIndex=self.leading_joint_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=targ_j,
                                    force=5,
                                    maxVelocity=10)
            self.env.step_simulation()
        print(
            f'Warning: gripper move_j exceeded {timeout} second timeout. Skipping.')
        return True

    def close(self) -> bool:
        return self.move_j(targ_j=self.leading_joint_range[1])

    def open(self) -> bool:
        return self.move_j(targ_j=self.leading_joint_range[0])

    def grasp(self) -> bool:
        return self.move_m()

    def create_gripper_filter(self) -> None:
        # NOTE: width and height have to be odd
        gripper_width_px = 45  # ~ real_width / env.pixel_size
        gripper_height_px = 7
        # Max possible width of image if rotated by 45 degrees
        pad_to = int(gripper_width_px * np.sqrt(2)) + 1
        # NOTE: pad_to has to be odd, so we have a center point to rotate around
        if pad_to % 2 == 0:
            pad_to += 1
        pad_w_by = pad_to - gripper_width_px  # Will always be even
        pad_h_by = pad_to - gripper_height_px

        gripper_filter = np.zeros(
            (gripper_height_px, gripper_width_px), dtype=np.int32)
        gripper_filter[gripper_height_px // 2,
                       gripper_width_px // 2] = 1
        gripper_filter[:, 0] = -255
        gripper_filter[:, -1] = -255
        gripper_filter = np.pad(gripper_filter, ((
            pad_h_by // 2, pad_h_by // 2), (pad_w_by // 2, pad_w_by // 2)), 'constant')
        self.gripper_filter = gripper_filter
