#!/usr/bin/env python

from __future__ import print_function
import os
import numpy as np

import time
from utils.pybullet_tools.kuka_primitives3 import BodyPose, BodyConf, Register
from utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, disconnect, get_bodies, HideOutput, \
    create_box, \
    load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, set_joint_positions, \
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer, p

from utils.pybullet_tools.body_utils import draw_frame

from copy import copy


def display_scenario():
    connect(use_gui=True)

    dirname = os.path.dirname(__file__)
    file = 'cembre_description/cembre_simplified.urdf'
    load_file = os.path.join(dirname, file)

    scn = load_pybullet(load_file, fixed_base=True)

    # _link_name_to_index = {p.getBodyInfo(scn)[0].decode('UTF-8'): -1, }
    # for _id in range(p.getNumJoints(scn)):
    #     _name = p.getJointInfo(scn, _id)[12].decode('UTF-8')
    #     _link_name_to_index[_name] = _id
    #
    # base_link_pose = p.getLinkState(scn, _link_name_to_index['base_link'])[:2]
    # print(base_link_pose)

    file = 'cembre_description/ur10e.urdf'
    load_file = os.path.join(dirname, file)
    r = load_pybullet(load_file, fixed_base=True)
    # set_pose(r, (base_link_pose[0],base_link_pose[1]))
    set_pose(r, ((-0.25499999999999995, 0.06099999999999994, 2.0580000000000003), (0.7071067811882787, 0.7071067811848163, -7.31230107716731e-14, -7.312301077203115e-14)))


    for i in range(10000):
        step_simulation()
        time.sleep(0.1)

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    display_scenario()
