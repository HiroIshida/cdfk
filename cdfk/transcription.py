from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pinocchio as pin

from cdfk.pinocchio_wrap import PinocchioWrap


@dataclass
class VarnameRangeTable:
    q_list: List[range]  # joint configuration
    qd_list: List[range]  # v in Dai's paper
    r_list: List[range]  # com position
    rd_list: List[range]  # com velocity
    rdd_list: List[range]  # com acceleration
    c_list_list: List[List[range]]  # contact point
    F_list_list: List[List[range]]  # contact force
    betas_list_list: List[List[range]]  # linear combination
    h_list: List[range]  # momentum
    hd_list: List[range]  # momentum rate
    dt_list: List[range]  # time step
    ndim: int  # total number of optimization variables

    @classmethod
    def create(cls, T: int, dof: int, n_contact: int) -> "VarnameRangeTable":
        head = 0
        q_list = []
        qd_list = []
        r_list = []
        rd_list = []
        rdd_list = []
        c_list_list = []
        F_list_list = []
        betas_list_list = []
        h_list = []
        hd_list = []
        dt_list = []
        for i in range(T):
            q_list.append(range(head, head + dof))
            head += dof
            # dof - 1 because quaternion is represented by 4 variables
            # but omega is represented by 3 variables
            qd_list.append(range(head, head + (dof - 1)))
            head += dof - 1
            r_list.append(range(head, head + 3))
            head += 3
            rd_list.append(range(head, head + 3))
            head += 3
            rdd_list.append(range(head, head + 3))
            head += 3

            c_list = []
            F_list = []
            betas_list = []
            for j in range(n_contact):
                c_list.append(range(head, head + 3))
                head += 3
                F_list.append(range(head, head + 3))
                head += 3
                betas_list.append(range(head, head + 4))
                head += 4

            c_list_list.append(c_list)
            F_list_list.append(F_list)
            betas_list_list.append(betas_list)

            h_list.append(range(head, head + 3))
            head += 3
            hd_list.append(range(head, head + 3))
            head += 3
            dt_list.append(range(head, head + 1))
            head += 1
        return cls(
            q_list,
            qd_list,
            r_list,
            rd_list,
            rdd_list,
            c_list_list,
            F_list_list,
            betas_list_list,
            h_list,
            hd_list,
            dt_list,
            head,
        )


@dataclass
class EqConstRangeTable:
    com_dyn_list: List[range]  # (7a) com dynamics
    com_mom_list: List[range]  # (7b) com momentum
    cam_list: List[range]  # (7c)  centroidal angular momentum
    qd_euler_list: List[range]  # (7d)
    hd_euler_list: List[range]  # (7e)
    rd_euler_list: List[range]  # (7f)
    rdd_euler_list: List[range]  # (7g)
    com_list: List[range]  # (7h)
    contact_point_kin_list_list: List[List[range]]  # (7i)
    fixed_contact_list_list: List[List[range]]  # (7j)
    lincomb_vec_list_list: List[List[range]]
    n_const: int  # total number of equality constraints

    @classmethod
    def create(cls, T: int, dof: int, n_contact: int) -> "EqConstRangeTable":
        com_dyn_list = []
        com_mom_list = []
        cam_list = []
        qd_euler_list = []
        hd_euler_list = []
        rd_euler_list = []
        rdd_euler_list = []
        com_list = []
        contact_point_kin_list_list = []
        fixed_contact_list_list = []
        lincomb_vec_list_list = []

        head = 0
        for i in range(T):
            com_dyn_list.append(range(head, head + 3))
            head += 3
            com_mom_list.append(range(head, head + 3))
            head += 3
            cam_list.append(range(head, head + 3))
            head += 3
            qd_euler_list.append(range(head, head + dof))
            head += dof
            hd_euler_list.append(range(head, head + 3))
            head += 3
            rd_euler_list.append(range(head, head + 3))
            head += 3
            rdd_euler_list.append(range(head, head + 3))
            head += 3
            com_list.append(range(head, head + 3))
            head += 3
            contact_point_kin_list_list.append([])
            for j in range(n_contact):
                contact_point_kin_list_list[i].append(range(head, head + 3))
                head += 3
            fixed_contact_list_list.append([])
            for j in range(n_contact):
                fixed_contact_list_list[i].append(range(head, head + 3))
                head += 3

            # considering we approximate the friction cone by a four-sided pyramid
            lincomb_vec_list = []
            for j in range(n_contact):
                lincomb_vec_list.append(range(head, head + 3))
                head += 3
            lincomb_vec_list_list.append(lincomb_vec_list)

        return cls(
            com_dyn_list,
            com_mom_list,
            cam_list,
            qd_euler_list,
            hd_euler_list,
            rd_euler_list,
            rdd_euler_list,
            com_list,
            contact_point_kin_list_list,
            fixed_contact_list_list,
            lincomb_vec_list_list,
            head,
        )


@dataclass
class EqConst:
    var_range_table: VarnameRangeTable
    eq_range_table: EqConstRangeTable
    T: int
    m: float
    pinwrap: PinocchioWrap
    ef_points_list: List[np.ndarray]
    friction_pyramid: np.ndarray

    def __init__(self, T: int, pinwrap: PinocchioWrap, q_ef_hold: np.ndarray):
        dof = pinwrap.model.nq
        n_contact = len(pinwrap.ee_fid_list)

        pinwrap.forward_kinematics(q_ef_hold)
        ef_points_list = []
        for ee_fid in pinwrap.ee_fid_list:
            co = pinwrap.get_frame_coords(ee_fid)
            ef_points_list.append(co.translation)

        self.var_range_table = VarnameRangeTable.create(T, dof, n_contact)
        self.eq_range_table = EqConstRangeTable.create(T, dof, n_contact)
        self.T = T
        self.m = pin.computeTotalMass(pinwrap.model)
        self.pinwrap = pinwrap
        self.ef_points_list = ef_points_list

        mu = 0.5
        friction_pyramid = np.array(
            [
                [mu / np.sqrt(2), mu / np.sqrt(2), 1],
                [-mu / np.sqrt(2), mu / np.sqrt(2), 1],
                [-mu / np.sqrt(2), -mu / np.sqrt(2), 1],
                [mu / np.sqrt(2), -mu / np.sqrt(2), 1],
            ]
        )
        self.friction_pyramid = friction_pyramid

    @property
    def n_contact(self) -> int:
        return len(self.pinwrap.ee_fid_list)

    @staticmethod
    def dquad_dt(q: np.ndarray, omega: np.ndarray):
        qw, qx, qy, qz = q
        wx, wy, wz = omega
        dqdt = 0.5 * np.array(
            [
                -qx * wx - qy * wy - qz * wz,
                qw * wx + qy * wz - qz * wy,
                qw * wy - qx * wz + qz * wx,
                qw * wz + qx * wy - qy * wx,
            ]
        )
        return dqdt

    def __call__(self, vec: np.ndarray, with_jacobian: bool) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros(self.eq_range_table.n_const)
        for i in range(self.T):
            q_i = vec[self.var_range_table.q_list[i]]
            qd_i = vec[self.var_range_table.qd_list[i]]
            r_i = vec[self.var_range_table.r_list[i]]
            rd_i = vec[self.var_range_table.rd_list[i]]
            rdd_i = vec[self.var_range_table.rdd_list[i]]
            h_i = vec[self.var_range_table.h_list[i]]
            hd_i = vec[self.var_range_table.hd_list[i]]
            dt_i = vec[self.var_range_table.dt_list[i]]

            self.pinwrap.forward_kinematics(q_i)

            # (7a) com dynamics
            var = self.m * rdd_i - self.m * np.array([0, 0, 9.8])
            for j in range(self.n_contact):
                F_ij = vec[self.var_range_table.F_list_list[i][j]]
                var -= F_ij
            out[self.eq_range_table.com_dyn_list[i]] = var

            # (7b) com momentum
            var = hd_i
            for j in range(self.n_contact):
                F_ij = vec[self.var_range_table.F_list_list[i][j]]
                c_ij = vec[self.var_range_table.c_list_list[i][j]]
                var -= np.cross(c_ij - r_i, F_ij)
            out[self.eq_range_table.com_mom_list[i]] = var

            # (7c) centroidal angular momentum
            Ag = pin.computeCentroidalMap(self.pinwrap.model, self.pinwrap.data, q_i)[3:, :]
            var = h_i - np.dot(Ag, qd_i)
            out[self.eq_range_table.cam_list[i]] = var

            if i > 0:
                # (7d) qd_euler
                _, quad_i, _ = q_i[:3], q_i[3:7], q_i[7:]
                v_i, omega_i, jointd_i = qd_i[:3], qd_i[3:6], qd_i[6:]
                dq = np.hstack([v_i, self.dquad_dt(quad_i, omega_i), jointd_i]) * dt_i
                q_im = vec[self.var_range_table.q_list[i - 1]]
                out[self.eq_range_table.qd_euler_list[i]] = (q_i - q_im) - dq

                # (7e) hd_euler
                h_im = vec[self.var_range_table.h_list[i - 1]]
                out[self.eq_range_table.hd_euler_list[i]] = (h_i - h_im) - dt_i * hd_i

                # (7f) rd_euler
                r_im = vec[self.var_range_table.r_list[i - 1]]
                rd_im = vec[self.var_range_table.rd_list[i - 1]]
                out[self.eq_range_table.rd_euler_list[i]] = (r_i - r_im) - dt_i * (rd_i + rd_im) / 2

                # (7g) rdd_euler
                out[self.eq_range_table.rdd_euler_list[i]] = (rd_i - rd_im) - dt_i * rdd_i

            # (7h) com
            com = pin.centerOfMass(self.pinwrap.model, self.pinwrap.data, q_i)
            out[self.eq_range_table.com_list[i]] = r_i - com

            # (7i) contact point kinematics
            for j in range(self.n_contact):
                c_ij = vec[self.var_range_table.c_list_list[i][j]]
                p_ij = self.pinwrap.get_frame_coords(self.pinwrap.ee_fid_list[j]).translation
                out[self.eq_range_table.contact_point_kin_list_list[i][j]] = c_ij - p_ij

            # (7j) fixed contact
            for j in range(self.n_contact):
                c_ij = vec[self.var_range_table.c_list_list[i][j]]
                out[self.eq_range_table.fixed_contact_list_list[i][j]] = (
                    c_ij - self.ef_points_list[j]
                )

            # force as linear combination of friction pyramid
            for j in range(self.n_contact):
                F_ij = vec[self.var_range_table.F_list_list[i][j]]
                beta_ij = vec[self.var_range_table.betas_list_list[i][j]]
                out[self.eq_range_table.lincomb_vec_list_list[i][j]] = (
                    beta_ij @ self.friction_pyramid - F_ij
                )
