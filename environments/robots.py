import os
import time
import numpy as np
import pybullet as p


from environments.grippers import Robotiq140
from environments.pybullet_utils import JointInfo


UR10E_URDF_PATH = "robot/ur10e.urdf"


class UR10E_Robotiq140:
    def __init__(self, assets_root: str, env, env_uid, base_id) -> None:
        # Load robot
        _, _, _, _, base_pos, base_rot = p.getLinkState(
            env_uid, base_id, computeForwardKinematics=True)
        self.assets_root = assets_root
        self.env = env
        self.robot_urdf_path = UR10E_URDF_PATH
        self.uid = p.loadURDF(os.path.join(self.assets_root, self.robot_urdf_path), base_pos, base_rot,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        # Save all controllable joints
        _n_urdf_joints = p.getNumJoints(self.uid)
        _urdf_joints_info = [p.getJointInfo(self.uid, i)
                             for i in range(_n_urdf_joints)]
        self.joints_info = [JointInfo(j[0], j[1].decode("utf-8"), j[10], j[11])
                            for j in _urdf_joints_info if j[2] == p.JOINT_REVOLUTE]

        self.n_joints = len(self.joints_info)

        # Get id of flange
        self.flange_joint_id = [j[0] for j in _urdf_joints_info if j[1].decode(
            "utf-8") == "realsense_mount_base_joint"][0]
        # Get id of tcp
        self.tcp_joint_id = [j[0] for j in _urdf_joints_info if j[1].decode(
            "utf-8") == "tcp_joint"][0]

        # Initialize gripper
        self.ee = Robotiq140(assets_root=assets_root, env=self.env,
                             robot_uid=self.uid, flange_id=self.flange_joint_id)

        # Define home joint positions
        self.homej = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        self.last_trajectory = []

    def reset(self):
        for i in range(self.n_joints):
            p.resetJointState(self.uid, self.joints_info[i].id, self.homej[i])
        self.ee.reset()

    def home(self):
        self.movej(self.homej)
        self.ee.open()

    def movej(self, targj, speed=0.01, timeout=4.0):
        """Move UR5 to target joint configuration."""
        if self.env.save_video:
            timeout = timeout * 50

        # Clear last trajectory list
        self.last_movej_trajectory = []

        t0 = time.time()
        while (time.time() - t0) < timeout:
            currs = p.getJointStates(
                self.uid, [joint.id for joint in self.joints_info])
            currj = [current_state[0] for current_state in currs]
            currj = np.array(currj)

            # Save ee pose to last_trajectory:
            ee_pose = self.get_ee_pose()
            self.last_movej_trajectory.append(ee_pose)

            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(self.n_joints)
            p.setJointMotorControlArray(
                bodyIndex=self.uid,
                jointIndices=[joint.id for joint in self.joints_info],
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains)

            self.env.step_counter += 1
            self.env.step_simulation()

        print(
            f'Warning: robot movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movep(self, pose, speed=0.01):
        # self.env.add_object(
        #     urdf='util/coordinate_axes.urdf', pose=pose, category='fixed')
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.uid,
            endEffectorLinkIndex=self.tcp_joint_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34] * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def get_ee_pose(self):
        """This should be in "grippers.py" the moment the tcp link/joint no longer resides inside robot urdf for IK purposes"""
        _, _, _, _, base_pos, base_rot = p.getLinkState(
            self.uid, self.tcp_joint_id, computeForwardKinematics=True)
        return (np.array(base_pos), np.array(base_rot))
