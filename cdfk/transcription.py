from dataclasses import dataclass
from typing import List



@dataclass
class VarnameRangeTable:
    q_list: List[range]  # joint configuration
    qd_list: List[range]  # v in Dai's paper
    r_list: List[range]  # com position
    rd_list: List[range]  # com velocity
    rdd_list: List[range]  # com acceleration
    F_list: List[range]  # contact force
    h_list: List[range]  # momentum
    hd_list: List[range]  # momentum rate
    dt_list: List[range]  # time step
    ndim: int  # total number of optimization variables

    @classmethod
    def create(cls, T: int, dof: int, n_contact: int) -> 'VarnameRangeTable':
        head = 0
        q_list = []
        qd_list = []
        r_list = []
        rd_list = []
        rdd_list = []
        F_list = []
        h_list = []
        hd_list = []
        dt_list = []
        for i in range(T):
            q_list.append(range(head, head + dof))
            head += dof
            qd_list.append(range(head, head + dof))
            head += dof
            r_list.append(range(head, head + 3))
            head += 3
            rd_list.append(range(head, head + 3))
            head += 3
            rdd_list.append(range(head, head + 3))
            head += 3
            for j in range(n_contact):
                F_list.append(range(head, head + 3))
                head += 3
            h_list.append(range(head, head + 3))
            head += 3
            hd_list.append(range(head, head + 3))
            head += 3
            dt_list.append(range(head, head + 1))
            head += 1
        return cls(q_list, qd_list, r_list, rd_list, rdd_list, F_list, h_list, hd_list, dt_list, head)


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
    contact_point_kin_list: List[List[range]]  # (7i)
    fixed_contact_list: List[List[range]]
    n_const: int  # total number of equality constraints

    @classmethod
    def create(cls, T: int, dof: int, n_contact: int) -> 'EqConstRangeTable':
        com_dyn_list = []
        com_mom_list = []
        cam_list = []
        qd_euler_list = []
        hd_euler_list = []
        rd_euler_list = []
        rdd_euler_list = []
        com_list = []
        contact_point_kin_list = []
        fixed_contact_list = []

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
            contact_point_kin_list.append([])
            for j in range(n_contact):
                contact_point_kin_list[i].append(range(head, head + 3))
                head += 3
            fixed_contact_list.append([])
            for j in range(n_contact):
                fixed_contact_list[i].append(range(head, head + 3))
                head += 3
        return cls(com_dyn_list, com_mom_list, cam_list, qd_euler_list, hd_euler_list, rd_euler_list, rdd_euler_list,
                   com_list, contact_point_kin_list, fixed_contact_list, head)
