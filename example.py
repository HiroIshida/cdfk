import time

import numpy as np
import pinocchio as pin
from robot_descriptions.jaxon_description import URDF_PATH as JAXON_URDF_PATH
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis, Box

from cdfk.pinocchio_wrap import EndEffectorConfig, PinocchioWrap
from cdfk.transcription import EqConst
from cdfk.viewer_utils import Viewer

manip_pose = {
    "RLEG_JOINT0": 0.0,
    "RLEG_JOINT1": 0.0,
    "RLEG_JOINT2": -0.349066,
    "RLEG_JOINT3": 0.698132,
    "RLEG_JOINT4": -0.349066,
    "RLEG_JOINT5": 0.0,
    "LLEG_JOINT0": 0.0,
    "LLEG_JOINT1": 0.0,
    "LLEG_JOINT2": -0.349066,
    "LLEG_JOINT3": 0.698132,
    "LLEG_JOINT4": -0.349066,
    "LLEG_JOINT5": 0.0,
    "CHEST_JOINT0": 0.0,
    "CHEST_JOINT1": 0.0,
    "CHEST_JOINT2": 0.0,
    "RARM_JOINT0": 0.0,
    "RARM_JOINT1": 0.959931,
    "RARM_JOINT2": -0.349066,
    "RARM_JOINT3": -0.261799,
    "RARM_JOINT4": -1.74533,
    "RARM_JOINT5": -0.436332,
    "RARM_JOINT6": 0.0,
    "RARM_JOINT7": -0.785398,
    "LARM_JOINT0": 0.0,
    "LARM_JOINT1": 0.959931,
    "LARM_JOINT2": 0.349066,
    "LARM_JOINT3": 0.261799,
    "LARM_JOINT4": -1.74533,
    "LARM_JOINT5": 0.436332,
    "LARM_JOINT6": 0.0,
    "LARM_JOINT7": -0.785398,
}


efconf1 = EndEffectorConfig("rfoot1", "RLEG_LINK5", np.array([0.125, 0.05, -0.1]))
efconf2 = EndEffectorConfig("rfoot2", "RLEG_LINK5", np.array([0.125, -0.065, -0.1]))
efconf3 = EndEffectorConfig("rfoot3", "RLEG_LINK5", np.array([-0.10, 0.05, -0.1]))
efconf4 = EndEffectorConfig("rfoot4", "RLEG_LINK5", np.array([-0.10, -0.065, -0.1]))

efconf5 = EndEffectorConfig("lfoot1", "LLEG_LINK5", np.array([0.125, -0.05, -0.1]))
efconf6 = EndEffectorConfig("lfoot2", "LLEG_LINK5", np.array([0.125, 0.065, -0.1]))
efconf7 = EndEffectorConfig("lfoot3", "LLEG_LINK5", np.array([-0.10, -0.05, -0.1]))
efconf8 = EndEffectorConfig("lfoot4", "LLEG_LINK5", np.array([-0.10, 0.065, -0.1]))
efconfs = [efconf1, efconf2, efconf3, efconf4, efconf5, efconf6, efconf7, efconf8]


pinwrap = PinocchioWrap(JAXON_URDF_PATH, efconfs)
const = EqConst(30, pinwrap)

# create joint_name to index mapping
q = pin.neutral(pinwrap.model)
q[2] = 1.0
for name, angle in manip_pose.items():
    jid = pinwrap.model.getJointId(name)
    q[jid + 5] = angle

const(np.zeros(const.var_range_table.ndim), True)

pinwrap.forward_kinematics(q)
axes = []
for i in range(4):
    fid = pinwrap.get_frame_id(f"rfoot{i+1}")
    coords = pinwrap.get_frame_coords(fid)
    skcoords = Coordinates(pos=coords.translation, rot=coords.rotation)
    axis = Axis.from_coords(skcoords)
    axes.append(axis)

v = Viewer(JAXON_URDF_PATH, pinwrap.model)
ground = Box([2.0, 2.0, 0.1], pos=[0.0, 0.0, -0.05])
v.update(q)
v.viewer.add(ground)
for a in axes:
    v.viewer.add(a)
v.show()
import time

time.sleep(1000)
