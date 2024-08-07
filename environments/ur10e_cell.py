from .environment import Environment
from .cameras import NeRFCameraFactory


class UR10ECell(Environment):
    def __init__(self, assets_root, task=None, disp=False, hz=240, record_cfg=None):
        super().__init__(assets_root, task, disp, hz, record_cfg)

        self.env_urdf_path = "robot/ur10e_cell.urdf"
        cam_factory = NeRFCameraFactory()
        self.agent_cams = [cam_factory.create().CONFIG[0] for i in range(50)]
        self.robot_joint_name = "platform_joint"
