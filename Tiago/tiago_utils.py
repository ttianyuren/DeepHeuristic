import math
import os
import random
import re
from collections import namedtuple
from itertools import combinations

import numpy as np

from utils.pybullet_tools.utils import multiply, get_link_pose, joint_from_name, set_joint_position, joints_from_names, \
    set_joint_positions, get_joint_positions, get_min_limit, get_max_limit, quat_from_euler, read_pickle, set_pose, \
    set_base_values, \
    get_pose, euler_from_quat, link_from_name, has_link, point_from_pose, invert, Pose, \
    unit_pose, joints_from_names, PoseSaver, get_aabb, get_joint_limits, get_joints, \
    ConfSaver, get_bodies, create_mesh, remove_body, single_collision, unit_from_theta, angle_between, violates_limit, \
    violates_limits, add_line, get_body_name, get_num_joints, approximate_as_cylinder, \
    approximate_as_prism, unit_quat, unit_point, clip, get_joint_info, tform_point, get_yaw, \
    get_pitch, wait_for_user, quat_angle_between, angle_between, quat_from_pose, compute_jacobian, \
    movable_from_joints, quat_from_axis_angle, LockRenderer, Euler, get_links, get_link_name, \
    draw_point, draw_pose, get_extend_fn, get_moving_links, link_pairs_collision, draw_point, get_link_subtree, \
    clone_body, get_all_links, set_color, pairwise_collision, tform_point, wait_for_duration, add_body_name, RED, GREEN, \
    YELLOW, apply_alpha


#Webot https://cyberbotics.com/doc/guide/tiago-steel
Tiago_GROUPS = {
    'base': ['x', 'y', 'theta'],
    'torso': ['torso_lift_joint'],                                              #ID: 24
    'head': ['head_1_joint', 'head_2_joint'], 
    'arm': ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint',         #ID: 34, 35, 36
            'arm_5_joint', 'arm_6_joint', 'arm_7_joint'],                       #ID: 37, 38, 39
    'gripper': ['gripper_left_finger_joint', 'gripper_right_finger_joint'], 
    'wheel': ['wheel_left_joint', 'wheel_right_joint']
}

#####################################################
########### NOT NEEDED ##############################

Tiago_arm_limits = {
    'arm_joint_1': [0.07, 2.68],
    'arm_joint_2': [-1.5, 1.02],
    'arm_joint_3': [-3.46, 1.5],
    'arm_joint_4': [-0.32, 2.27],
    'arm_joint_5': [-2.07, 2.07],
    'arm_joint_6': [-1.39, 1.39],
    'arm_joint_7': [-2.07, 2.07], 
}

Tiago_head_limits = {
    'joint_1': [-1.24, 1.24], 
    'joint_2': [-0.98, 0.79]
}

Tiago_gripper_limits = {
    'joint_1': [0, 0.05],
    'joint_2': [0, 0.05]
}

Tiago_torso_limits = {
    'joint_1': [0, 0.35]
}

Tiago_wheel_limits = {
    'left': [-3.14, 3.14],
    'right': [-3.14, 3.14]
}

#####################################################
#####################################################


Tiago_Base_Link = 'base_footprint'

Tiago_URDF = "tiago_description/tiago_single.urdf"

# Special Arm configurations
REST_ARM = [0.4303, -1.4589, -0.5566, 2.0267, -1.3867, 1.3321, 0.1261]
WEAR_OBJECT = []        #TODO


INITIAL_GRASP_POSITIONS = {
    'rest': REST_ARM,
    'wear': WEAR_OBJECT
}

CARRY_ARM_CONF = []

def get_initial_conf(grasp_type):
    return INITIAL_GRASP_POSITIONS[grasp_type]


def get_joints_from_body(robot, body_part):
    return joints_from_names(robot, Tiago_GROUPS[body_part])


def open_arm(robot):
    for joint in get_joints_from_body(robot, "gripper"):
        set_joint_position(robot, joint, get_max_limit(robot, joint))

def close_arm(robot):
    for joint in get_joints_from_body(robot, "gripper"):
        set_joint_position(robot, joint, get_min_limit(robot, joint))


def set_group_conf(robot, body_part, positions):
    set_joint_positions(robot, get_joints_from_body(robot, body_part), positions)




# Box grasps

# GRASP_LENGTH = 0.04
GRASP_LENGTH = 0.
# GRASP_LENGTH = -0.01

# MAX_GRASP_WIDTH = 0.07
MAX_GRASP_WIDTH = np.inf

# Arm tool poses
TOOL_POSE = Pose(euler=Euler(pitch=np.pi / 2)) 

def get_top_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=unit_pose(),
                   max_width=MAX_GRASP_WIDTH, grasp_length=GRASP_LENGTH):
    pass