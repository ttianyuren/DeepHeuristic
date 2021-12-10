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
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer

from utils.pybullet_tools.body_utils import draw_frame

from copy import copy


def display_scenario():
    connect(use_gui=True)

    dirname = os.path.dirname(__file__)
    file = 'cembre_description/cembre_simplified.urdf'
    load_file = os.path.join(dirname, file)

    scn = load_pybullet(load_file, fixed_base=True)

    for i in range(10000):
        step_simulation()
        time.sleep(0.1)

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    display_scenario()
