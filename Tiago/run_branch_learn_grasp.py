#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import os
import cProfile
import pstats
import argparse
import pickle as pk
import pybullet as p
import time
from etamp.actions import ActionInfo
from etamp.stream import StreamInfo


from Tiago.tiago_utils import open_arm, close_arm, set_group_conf, get_initial_conf, get_joints_from_body, Tiago_limits

from Tiago.tiago_primitives import BodyPose, sdg_sample_place, sdg_sample_grasp, sdg_ik_grasp, sdg_motion_base_joint,\
    GraspDirection, sdg_plan_free_motion, sdg_sample_grasp_dir, sdg_sample_base_position, BodyConf, BodyInfo

from utils.pybullet_tools.pr2_primitives import  Conf, get_ik_ir_gen, get_motion_gen, \
    get_stable_gen, get_grasp_gen, Attach, Detach, Clean, Cook, control_commands, \
    get_gripper_joints, GripperCommand, apply_commands, State, Command

from utils.pybullet_tools.body_utils import get_raytest_scatter3, get_ellipsoid_frame
from utils.pybullet_tools.pr2_utils import get_arm_joints, ARM_NAMES,  get_group_joints, get_group_conf
from utils.pybullet_tools.utils import WorldSaver, is_connected,step_simulation,  connect, get_pose, set_pose, get_configuration, is_placement, \
    disconnect, get_bodies, connect, get_pose, is_placement, point_from_pose, \
    disconnect, user_input, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, \
    get_distance, LockRenderer, get_min_limit, get_max_limit
# from etamp.progressive3 import solve_progressive, solve_progressive2
from etamp.pddlstream.utils import read, INF, get_file_path, find_unique
from etamp.pddlstream.language.constants import pAtom

from etamp.p_uct2 import PlannerUCT
from etamp.tree_node2 import ExtendedNode
from etamp.env_sk_branch import SkeletonEnv
from Tiago.build_world_learn_grasp import BuildWorldScenario
from etamp.topk_skeleton import EXE_Action, EXE_Stream
from etamp.pddlstream.language.object import Object, OptimisticObject, EXE_Object, EXE_OptimisticObject, get_hash


def get_fixed(robot, movable):
    rigid = [body for body in get_bodies() if body != robot]
    fixed = [body for body in rigid if body not in movable]
    return fixed


def place_movable(certified):
    placed = []
    for literal in certified:
        if literal[0] == 'not':
            fact = literal[1]
            if fact[0] == 'trajcollision':
                _, b, p = fact[1:]
                set_pose(b, p.pose)
                placed.append(b)
    return placed


def extract_motion(action_plan):
    """
    Return a list of robot motions
    each of which corresponds to a motor action in the action_plan.
    """
    list_motion = []
    for name, args, _ in action_plan:
        # args are instances of classes
        cmd = args[-1]
        if name == 'place':
            """Since 'place' is the reversed motion of 'pick',
               its path is simply the reverse of what generated by 'pick'."""
            reversed_cmd = cmd.reverse()
            list_motion += reversed_cmd.body_paths
        elif name in ['move', 'move_free', 'move_holding', 'pick']:
            list_motion += cmd.body_paths
    print('list of paths ----------------------------')
    print(action_plan)
    print(list_motion)
    print('----------------------------------')
    return list_motion


def move_cost_fn(*args):
    """
    :param c: Commands
    """
    c = args[-1]  # objects
    [t] = c.value.body_paths
    distance = t.distance()
    return distance + 0.1


def get_const_cost_fn(cost):
    def fn(*args):
        return cost

    return fn


def get_action_cost(plan):
    cost = None
    if plan:
        cost = 0
        for paction in plan:
            if callable(paction.pa_info.cost_fn):
                cost += paction.pa_info.cost_fn(*paction.args)
        # print('Action Cost ====================== ', cost)
    return cost


def get_update_env_reward_fn(scn, action_info):
    def get_actions_cost(exe_plan):
        cost = None
        if exe_plan:
            cost = 0
            for action in exe_plan:
                if action.name not in action_info:
                    continue
                cost_fn = action_info[action.name].cost_fn
                if callable(cost_fn):
                    cost += cost_fn(*action.parameters)
        return cost

    def fn(list_exe_action):

        cost = get_actions_cost(list_exe_action)

        """Execution uncertainty will be implemented here."""
        with LockRenderer():
            for action in list_exe_action:
                for patom in action.add_effects:
                    if patom.name.lower() == "AtBConf".lower():
                        body_config = patom.args[0].value
                        body_config.assign()
                    elif patom.name.lower() == "AtPose".lower():
                        body_pose = patom.args[1].value
                        body_pose.assign()
                    elif patom.name.lower() == "AtGrasp".lower():
                        body_grasp = patom.args[2].value
                        attachment = body_grasp.attachment(scn.pr2, scn.arm_left)
                        attachment.assign()

        if cost is False:
            return None

        return 0.1 * np.exp(-cost)

    return fn


def postprocess_plan(scn, exe_plan, teleport=False):
    if exe_plan is None:
        return None
    commands = []
    for i, (name, args) in enumerate(exe_plan):
        if name == 'move_base':
            c = args[-1]
            new_commands = c.commands
        elif name == 'pick':
            a, b, p, g, _, c = args
            [t] = c.commands
            close_gripper = GripperCommand(scn.robots[0], a, g.grasp_width, teleport=teleport)
            attach = Attach(scn.robots[0], a, g, b)
            new_commands = [t, close_gripper, attach, t.reverse()]
        elif name == 'place':
            a, b, p, g, _, c = args
            [t] = c.commands
            gripper_joint = get_gripper_joints(scn.robots[0], a)[0]
            position = get_max_limit(scn.robots[0], gripper_joint)
            open_gripper = GripperCommand(scn.robots[0], a, position, teleport=teleport)
            detach = Detach(scn.robots[0], a, b)
            new_commands = [t, detach, open_gripper, t.reverse()]
        else:
            raise ValueError(name)
        # print(i, name, args, new_commands)
        commands += new_commands
    return commands


def play_commands(commands):
    use_control = False
    if use_control:
        control_commands(commands)
    else:
        apply_commands(State(), commands, time_step=0.01)


def get_op_plan(scn, target, path=None):
    """         SETUP: Position and Orientation of Box, Table, robot, IDs are bodys                 """
    robot = scn.robots[0]
    floor = scn.env_bodies[0]
    box_pose = BodyPose(target)
    box_info = BodyInfo(scn, target)
    robot_conf = BodyConf(robot)

    if path is not None and os.path.exists(path):
        with open(path, 'rb') as f:
            op_plan = pk.load(f)
    else:
        # initial variables
        oBody = EXE_Object(pddl='o' + str(box_pose.body), value=box_pose.body)
        oBodyPose = EXE_Object(pddl='pInit' + str(box_pose.body), value=box_pose)
        oFloorID = EXE_Object(pddl='surface', value=floor)
        oConf = EXE_Object(pddl='qInit', value=robot_conf)

        # TAMP variables
        voG0 = EXE_OptimisticObject(pddl='#g0', repr_name='#g0', value=None)
        voG1 = EXE_OptimisticObject(pddl='#g1', repr_name='#g1', value=None)
        voQ0 = EXE_OptimisticObject(pddl='#q0', repr_name='#q0', value=None)
        voT1 = EXE_OptimisticObject(pddl='#t1', repr_name='#t1', value=None)
        voQ11 = EXE_OptimisticObject(pddl='#q11', repr_name='#q11', value=None)
        voT37 = EXE_OptimisticObject(pddl='#t37', repr_name='#t37', value=None)

        op_plan = [EXE_Stream(inputs=(oBody,),
                              name='sample-grasp-direction',
                              outputs=(voG0,)),

                   EXE_Stream(inputs=(oBody, voG0),
                              name='sample-grasp',
                              outputs=(voG1,)),

                   EXE_Stream(inputs=(voG1, oFloorID, voG0),
                              name='sample-base-position',
                              outputs=(voQ0,)),

                   EXE_Stream(inputs=(oConf, voQ0,),
                              name='plan-free-motion',
                              outputs=(voT37,)),

                   EXE_Action(add_effects=(pAtom(name='atbconf', args=(voQ0,)),),
                              name='move-free-base',
                              parameters=(oConf, voQ0, voT37,)),


                   EXE_Stream(inputs=(oBody, oBodyPose, voG1,),
                              name='inverse-kinematics',
                              outputs=(voQ11, voT1,)),

                   EXE_Action(add_effects=(pAtom(name='atbconf', args=(voQ11,)),),
                              name='move-free-arm',
                              parameters=(voQ0, voQ11, voT1,)),

                   EXE_Action(add_effects=(pAtom(name='atbpose', args=(oBody, voG0,)),),
                              name='pick',
                              parameters=(oBody, oBodyPose, voG0, voQ0, voT1,))
                   ]

    return op_plan



#######################################################
def main():
    run()
    print("NN: ", True)
    run(nn=True)


def run(cnn=True, nn=False, target=0):
    visualization = 1
    connect(use_gui=visualization)
    scn = BuildWorldScenario()
    robot = scn.robots[0]
    target = scn.movable_bodies[target]
    target_info = scn.dic_body_info[target]
    ellipsoid_frame, obj_extent, list_dist, list_dir_jj, list_z_jj = get_ellipsoid_frame(target, target_info, robot)
    mat_image = get_raytest_scatter3(target, ellipsoid_frame, obj_extent, robot)
    """TODO: Here operators should be implemented"""
    stream_info = {'sample-grasp-direction': StreamInfo(seed_gen_fn=sdg_sample_grasp_dir(cnn=cnn, img=mat_image), free_generator=True, discrete=True, p1=[0, 1, 2, 3, 4], p2=[.2, .2, .2, .2, .2]),
                   'sample-grasp': StreamInfo(seed_gen_fn=sdg_sample_grasp(robot, target_info),
                                              free_generator=False),
                   'sample-base-position': StreamInfo(seed_gen_fn=sdg_sample_base_position(scn.all_bodies, nn, list_dist, list_dir_jj, list_z_jj),
                                                      every_layer=15, free_generator=True, discrete=False, p1=[1, 1, 1], p2=[.2, .2, .2]),
                   'inverse-kinematics': StreamInfo(seed_gen_fn=sdg_ik_grasp(scn.robots[0], all_bodies=scn.all_bodies),
                                                    free_generator=False),
                   'plan-free-motion': StreamInfo(seed_gen_fn=sdg_plan_free_motion(scn.robots[0], scn.all_bodies),
                                                  free_generator=False),
                   }

    action_info = {'move-free-base': ActionInfo(optms_cost_fn=get_const_cost_fn(1), cost_fn=get_const_cost_fn(1)),
                   'move-free-arm': ActionInfo(optms_cost_fn=get_const_cost_fn(1), cost_fn=get_const_cost_fn(1)),
                   'pick': ActionInfo(optms_cost_fn=get_const_cost_fn(1), cost_fn=get_const_cost_fn(1)),
                   'place': ActionInfo(optms_cost_fn=get_const_cost_fn(1), cost_fn=get_const_cost_fn(1))
                   }
    op_plan = get_op_plan(scn, target)
    e_root = ExtendedNode()
    assert op_plan is not None

    """Here use tree search to bind open variables"""
    skeleton_env = SkeletonEnv(e_root.num_children, op_plan,
                               get_update_env_reward_fn(scn, action_info),
                               stream_info, scn)
    selected_branch = PlannerUCT(skeleton_env)

    st = time.time()

    concrete_plan = selected_branch.think(900, visualization)

    thinking_time = time.time() - st

    if concrete_plan is None:
        print('TAMP is failed.', concrete_plan)
        disconnect()
        return

    print('TAMP is successful. think_time: ', thinking_time)
    time.sleep(0)
    disconnect()
    """
    exe_plan = None
    if concrete_plan is not None:
        exe_plan = []
    for action in concrete_plan:
        exe_plan.append((action.name, [arg.value for arg in action.parameters]))

    with open('exe_plan.pk', 'wb') as f:
        pk.dump((scn, exe_plan), f)

    if exe_plan is None:
        disconnect()
        return

    with LockRenderer():
        commands = postprocess_plan(scn, exe_plan)

    play_commands(commands)
    """
    print('Finished.')


if __name__ == '__main__':
    main()
