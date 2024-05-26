import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import quaternion2matrix
from skrobot.model.robot_model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.viewers import PyrenderViewer


class Viewer:
    viewer: PyrenderViewer
    robot: RobotModel
    joint_id_map: np.ndarray

    def __init__(self, urdf_path, pin_model):
        self.viewer = PyrenderViewer()
        self.robot: RobotModel = RobotModelFromURDF(urdf_file=urdf_path)
        skrobot_joint_names = self.robot.joint_names

        # create joint map from pinocchio joint id to skrobot joint id
        len(self.robot.joint_list)
        table = {}

        for name in pin_model.names[1:]:
            jid = pin_model.getJointId(name)
            print(f"joint name: {name}, jid: {jid}")
            joint = pin_model.joints[jid]
            if name == "root_joint":
                assert joint.nq == 7, f"joint.nq = {joint.nq}"
            else:
                joint.nq == 1
                table[jid + 5] = skrobot_joint_names.index(name)

        self.joint_id_map = table
        self.viewer.add(self.robot)

    def update(self, q: np.ndarray):
        # q in pinocchio order
        # convert q to skrobot order
        q_root = q[:7]
        x, y, z, qx, qy, qz, qw = q_root
        mat = quaternion2matrix([qw, qx, qy, qz])
        co = Coordinates(pos=[x, y, z], rot=mat)
        print(co)
        self.robot.newcoords(co)

        q_joint = np.array([q[jid] for jid in self.joint_id_map])
        self.robot.angle_vector(q_joint)

    def show(self):
        self.viewer.show()
