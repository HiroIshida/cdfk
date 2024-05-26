import time

import numpy as np
import pinocchio as pin
from robot_descriptions.jaxon_description import URDF_PATH as JAXON_URDF_PATH
from robot_descriptions.loaders.pinocchio import load_robot_description
from skrobot.model.primitives import Box

from cdfk.transcription import (
    EndEffectorConfig,
    EqConst,
    EqConstRangeTable,
    VarnameRangeTable,
)
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


efconf1 = EndEffectorConfig("rleg1", "RLEG_LINK5", np.array([0.125, 0.05, -0.1]))
efconf2 = EndEffectorConfig("rleg2", "RLEG_LINK5", np.array([0.125, -0.065, -0.1]))
efconf3 = EndEffectorConfig("rleg3", "RLEG_LINK5", np.array([-0.10, 0.05, -0.1]))
efconf4 = EndEffectorConfig("rleg4", "RLEG_LINK5", np.array([-0.10, -0.065, -0.1]))

efconf5 = EndEffectorConfig("lleg1", "LLEG_LINK5", np.array([0.125, -0.05, -0.1]))
efconf6 = EndEffectorConfig("lleg2", "LLEG_LINK5", np.array([0.125, 0.065, -0.1]))
efconf7 = EndEffectorConfig("lleg3", "LLEG_LINK5", np.array([-0.10, -0.05, -0.1]))
efconf8 = EndEffectorConfig("lleg4", "LLEG_LINK5", np.array([-0.10, 0.065, -0.1]))
efconfs = [efconf1, efconf2, efconf3, efconf4, efconf5, efconf6, efconf7, efconf8]


vrtable = VarnameRangeTable.create(10, 3, 2)
eqtable = EqConstRangeTable.create(10, 3, 2)

robot: pin.RobotWrapper = load_robot_description(
    "jaxon_description", root_joint=pin.JointModelFreeFlyer()
)

# joint names
joint_names = robot.model.names[1:]
for i, name in enumerate(joint_names):
    # show joint spec
    jid = robot.model.getJointId(name)
    joint = robot.model.joints[jid]
    print(f"Joint {i}: jid {jid} {name}, type: {joint.shortname()}, nq: {joint.nq}")

# create joint_name to index mapping
q = pin.neutral(robot.model)
q[2] = 1.0
for name, angle in manip_pose.items():
    jid = robot.model.getJointId(name)
    print(jid)
    # set angle
    # if jid = 1 then actual index is [0, 1, 2, 3, 4, 5, 6] 7
    q[jid + 5] = angle
    joint = robot.model.joints[jid]
    print(f"{name}, type: {joint.shortname()}, nq: {joint.nq}")


const = EqConst(20, robot, efconfs)

v = Viewer(JAXON_URDF_PATH, robot.model)
ground = Box([2.0, 2.0, 0.1], pos=[0.0, 0.0, -0.05])
v.update(q)
v.viewer.add(ground)
v.show()
import time

time.sleep(1000)


# ts = time.time()
# for _ in range(1000):
#     const(np.zeros(const.var_range_table.ndim), True)
# print("Time elapsed: ", (time.time() - ts) / 1000)
