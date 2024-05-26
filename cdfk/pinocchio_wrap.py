from dataclasses import dataclass
from typing import List

import numpy as np
import pinocchio as pin


@dataclass
class EndEffectorConfig:
    name: str
    parent_frame: str
    trans: np.ndarray


class PinocchioWrap:
    def __init__(self, urdf_path, end_effector_configs: List[EndEffectorConfig]) -> None:
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())

        ee_fid_list = []
        for ef_config in end_effector_configs:
            relative = pin.SE3(np.eye(3), ef_config.trans)
            parent_joint_idx = model.frames[model.getFrameId(ef_config.parent_frame)].parent
            new_frame = pin.Frame(
                ef_config.name, parent_joint_idx, 0, relative, pin.FrameType.OP_FRAME
            )
            model.addFrame(new_frame)
            frame_id = model.getFrameId(ef_config.name)
            ee_fid_list.append(frame_id)

        self.ee_fid_list = ee_fid_list
        self.model = model
        self.data = model.createData()

    def forward_kinematics(self, q) -> None:
        pin.forwardKinematics(self.model, self.data, q)

    def get_frame_coords(self, fid: int) -> pin.SE3:
        pin.updateFramePlacement(self.model, self.data, fid)
        return self.data.oMf[fid]

    def get_frame_id(self, frame_name) -> int:
        return self.model.getFrameId(frame_name)
