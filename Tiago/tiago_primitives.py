from __future__ import print_function

import copy
import pybullet as p
import random
import time
from itertools import islice

import numpy as np

#from ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
#from ikfast.utils import USE_CURRENT, USE_ALL
#from pr2_problems import get_fixed_bodies
from Tiago.tiago_utils import open_arm, joints_from_names
from utils.pybullet_tools.utils import *

BASE_EXTENT = 3.5  # 2.5
BASE_LIMITS = (-BASE_EXTENT * np.ones(2), BASE_EXTENT * np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False


#######################################################

#Diese Classe läd nur alle Objekte (Tisch, Boxen und boden) in 
class sdg_sample_place(object):
    def __init__(self, scn):
        self.all_bodies = scn.all_bodies

    def __call__(self, input_tuple, seed=None):
        body, surface = input_tuple     #robot, oberfläche???
        others = list(set(self.all_bodies) - {body, surface})
        """1) Generation"""
        pose = sample_placement_seed(body, surface, seed)
        """2) Validation"""
        if (pose is None) or any(pairwise_collision(body, b) for b in others):
            return None

        body_pose = BodyPose(body, pose)
        return (body_pose,)  # return a tuple


class sdg_sample_grasp(object):
    def __init__(self, scn):
        self.robot, self.arm, self.grasp_type = scn.pr2, scn.arms[0], scn.grasp_type

    def search(self, input_tuple, seed=None):
        """return the ee_frame wrt the measure_frame of the object"""
        body, = input_tuple  # grasp_dir defined in ellipsoid_frame of the body

        grasps = []

        if 'top' == self.grasp_type:
            approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0])
            grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_LEFT_ARM)
                          for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        if 'side' == self.grasp_type:
            approach_vector = APPROACH_DISTANCE * get_unit_vector([2, 0, -1])
            grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_LEFT_ARM)
                          for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))
        filtered_grasps = []
        for grasp in grasps:
            grasp_width = 0.0
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)

        random.shuffle(filtered_grasps)
        body_grasp = filtered_grasps[0]

        return (body_grasp,)  # return a tuple

    def __call__(self, input_tuple, seed=None):
        return self.search(input_tuple, seed=None)


class sdg_ik_grasp(object):
    def __init__(self, scn, max_attempts=25, learned=True, teleport=False, **kwargs):
        self.max_attempts = max_attempts

        self.ir_sampler = get_ir_sampler(scn, learned=learned, max_attempts=1, **kwargs)
        self.ik_fn = get_ik_fn(scn, teleport=teleport, **kwargs)

    def search(self, input_tuple, seed=None):
        b, a, p, g = input_tuple
        ir_generator = self.ir_sampler(*input_tuple)
        attempts = 0

        for i in range(self.max_attempts):
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return None
            if ir_outputs is None:
                continue
            ik_outputs = self.ik_fn(*(input_tuple + ir_outputs))
            if ik_outputs is None:
                continue
            # print('IK attempts:', attempts)
            result = ir_outputs + ik_outputs
            return result

        return None

    def __call__(self, input_tuple, seed=None):
        return self.search(input_tuple, seed=None)


class sdg_motion_base_joint(object):
    def __init__(self, scn, max_attempts=25, custom_limits={}, teleport=False, **kwargs):
        self.max_attempts = max_attempts
        self.teleport = teleport
        self.custom_limits = custom_limits

        self.robot = scn.pr2
        self.obstacles = list(set(scn.env_bodies) | set(scn.regions))

        self.saver = BodySaver(self.robot)

    def search(self, input_tuple, seed=None):
        bq1, bq2 = input_tuple
        self.saver.restore()
        bq1.assign()

        for i in range(self.max_attempts):
            if self.teleport:
                path = [bq1, bq2]
            elif is_drake_pr2(self.robot):
                raw_path = plan_joint_motion(self.robot, bq2.joints, bq2.values, attachments=[],
                                             obstacles=self.obstacles, custom_limits=self.custom_limits,
                                             self_collisions=SELF_COLLISIONS,
                                             restarts=4, iterations=50, smooth=50)
                if raw_path is None:
                    print('Failed motion plan!')
                    continue
                path = [Conf(self.robot, bq2.joints, q) for q in raw_path]
            else:
                goal_conf = base_values_from_pose(bq2.value)
                raw_path = plan_base_motion(self.robot, goal_conf, BASE_LIMITS, obstacles=self.obstacles)
                if raw_path is None:
                    print('Failed motion plan!')
                    continue
                path = [BodyPose(self.robot, pose_from_base_values(q, bq1.value)) for q in raw_path]
            bt = Trajectory(path)
            cmd = Commands(State(), savers=[BodySaver(self.robot)], commands=[bt])
            return (cmd,)
        return None

    def __call__(self, input_tuple, seed=None):
        return self.search(input_tuple, seed=None)

#######################################################

class BodyPose(object):
    # def __init__(self, position, orientation):
    #    self.position = position
    #    self.orientation = orientation
    def __init__(self, body, value=None, support=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.init = init

    def assign(self):
        set_pose(self.body, self.value)

    def iterate(self):
        yield self

    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)

    def __repr__(self):
        return 'p{}'.format(id(self) % 1000)



class Conf(object):
    def __init__(self, body, joints, values=None, init=False):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init

    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)

    def iterate(self):
        yield self

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)