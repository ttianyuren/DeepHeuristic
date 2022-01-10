#!/usr/bin/env python

from __future__ import print_function

import math
import os
import numpy as np
import pickle as pk
import torch
import time
from utils.pybullet_tools.kuka_primitives3 import BodyPose, BodyConf, Register
from utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, disconnect, get_bodies, HideOutput, \
    create_box, \
    load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, set_joint_positions, \
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer, p, set_color, draw_point, multiply, \
    pairwise_collision, link_from_name, inverse_kinematics, get_sample_fn, get_link_pose

from utils.pybullet_tools.body_utils import draw_frame, place_objects

from copy import copy
from mesh.mesh import icosphere
from train_DH import get_arg, model_setup
import torch.nn.functional as F
from generate_dataset_cembre import DataGenerator


def get_rand(low=0.08, high=0.20):
    return np.random.uniform(low, high)


def draw_sphere(radius, pos, color=[0.5, 0.5, 0.5, 0.5]):
    visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius)
    outSphere = p.createMultiBody(baseMass=0,
                                  baseVisualShapeIndex=visualShapeId)
    set_pose(outSphere, Pose(pos))
    set_color(outSphere, color)


def draw_pointCloud(points, color=(0, 0, 0)):
    with HideOutput():
        for p in points:
            draw_point(p, color=color)


def get_area_p(list_region):
    sum_area = 0
    list_area = []
    for r in list_region:
        _, extent = get_center_extent(r)
        # area = extent[0] * extent[1]
        area = np.sqrt(extent[0] ** 2 + extent[1] ** 2)
        sum_area += area
        list_area.append(area)
    results = []
    for a in list_area:
        results.append(a / sum_area)
    return results


# class DataGenerator(object):
#     def __init__(self, visualization=True):
#         env = load_pybullet('/cembre_description/cembre_simplified.urdf', fixed_base=True)
#
#         self.env_bodies = [env]
#         self.visualization = visualization
#         self.robot = load_pybullet('/cembre_description/ur10e.urdf', fixed_base=True)
#         self.movable_joints = get_movable_joints(self.robot)
#
#         self.wrist = load_pybullet('/cembre_description/end_link.urdf', fixed_base=False)
#
#         # set_pose(r, (base_link_pose[0],base_link_pose[1]))
#         set_pose(self.robot, ((-0.25499999999999995, 0.06099999999999994, 2.0580000000000003),
#                               (0.7071067811882787, 0.7071067811848163, -7.31230107716731e-14, -7.312301077203115e-14)))
#
#         self.reset_robot()
#
#         region1 = load_pybullet('/cembre_description/region_small.urdf', fixed_base=True)
#         set_pose(region1, Pose((-0.28, -0.1, 0.94)))
#         region2 = load_pybullet('/cembre_description/region_small.urdf', fixed_base=True)
#         set_pose(region2, Pose((0.34, 0.1, 1.067)))
#
#         self.regions = [region1, region2]
#         self.region_p = get_area_p(self.regions)
#
#         self.reset_containers()
#         self.remove_wrist()
#
#     def reset_containers(self):
#         self.target_obj = None
#         self.movable_bodies = []
#         self.all_bodies = list(set(self.env_bodies) | set(self.regions))
#         self.dic_body_info = {}
#
#     def reset_robot(self):
#         initial_jts = np.array([0.0, -0.1, -2.7, 1.5, 0.5 * np.pi, 0])
#         config_left = BodyConf(self.robot, initial_jts)
#         config_left.assign()
#
#     def sample_region(self):
#         idx = np.random.choice(len(self.regions), 1, p=self.region_p)[0]
#         # return idx
#         return self.regions[idx]
#
#     def remove_wrist(self):
#         set_pose(self.wrist, Pose((0, 0, -20)))
#
#     def reset(self):
#         if self.movable_bodies:
#             for b in self.movable_bodies:
#                 p.removeBody(b)
#         self.reset_containers()
#
#         obj1 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.859, 0.192, 0.306, 1.0))
#         obj2 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.271, 0.706, 0.490, 1.0))
#         obj3 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.647, 0.498, 0.894, 1.0))
#         self.target_obj = obj1
#         self.movable_bodies = [obj1, obj2, obj3]
#         self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))
#
#         for b in self.movable_bodies:
#             obj_center, obj_extent = get_center_extent(b)
#             r_outSphere = np.linalg.norm(obj_extent) / 2 * 1.01
#             body_pose = get_pose(b)
#             body_frame = tform_from_pose(body_pose)
#             center_frame = tform_from_pose((obj_center, body_pose[1]))
#             relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))
#
#             self.dic_body_info[b] = (obj_extent, r_outSphere, relative_frame_center)
#
#         region = self.sample_region()
#         list_remove = place_objects(self.movable_bodies, region, self.all_bodies)
#         for b in list_remove:
#             self.movable_bodies.remove(b)
#             self.all_bodies.remove(b)
#
#     def get_X(self):
#         p_inner_outer = 0.25
#         max_range = 0.6  # meters
#         unit_vertices = icosphere(level=4).vertices  # radius=1
#         centroid = np.array(get_pose(self.target_obj)[0])
#         r_inner = self.dic_body_info[self.target_obj][1] * p_inner_outer
#         inner_vertices = unit_vertices * r_inner + centroid
#         inner_ends = unit_vertices * (max_range + r_inner) + centroid
#         # draw_sphere(r_inner, centroid, [0, 1, 1, 0.5])
#
#         """Temporarily move away the objects that are not involved in rayTest"""
#         offset = [0, 0, -5]
#         obj_pose = get_pose(self.target_obj)
#         robot_pose = get_pose(self.robot)
#         temp_obj_pose = multiply(Pose(Point(*offset)), obj_pose)
#         set_pose(self.target_obj, temp_obj_pose)
#         temp_robot_pose = multiply(Pose(Point(*offset)), robot_pose)
#         set_pose(self.robot, temp_robot_pose)
#
#         temp_centroid = centroid + np.array(offset)
#         r_outer = self.dic_body_info[self.target_obj][1]
#         outer_vertices = unit_vertices * r_outer + temp_centroid
#         outer_ends = np.dstack([temp_centroid] * len(outer_vertices))[0].transpose((1, 0))
#         # draw_sphere(r_outer, temp_centroid, [1, 1, 0, 0.5])
#
#         # draw_pointCloud(outer_ends, color=[0, 1, 1])
#         # draw_pointCloud(outer_vertices, color=[1, 1, 0])
#         #
#         # draw_pointCloud(inner_ends, color=[0, 1, 1])
#         # draw_pointCloud(inner_vertices, color=[1, 0, 1])
#
#         rays_env = p.rayTestBatch(inner_vertices.tolist(), inner_ends.tolist())
#         rays_shape = p.rayTestBatch(outer_vertices.tolist(), outer_ends.tolist())
#
#         """Put the target object and the robot back"""
#         set_pose(self.robot, robot_pose)
#         set_pose(self.target_obj, obj_pose)
#
#         icosphere_features = []
#         for re, rs, v_norm in zip(rays_env, rays_shape, unit_vertices):
#             objectUniqueId1, linkIndex1, hitFractionb1, hitPosition1, hitNormal1 = re
#             objectUniqueId2, linkIndex2, hitFractionb2, hitPosition2, hitNormal2 = rs
#             cos1 = np.dot(hitNormal1, -v_norm)
#             cos2 = np.dot(hitNormal2, v_norm)
#             icosphere_features.append([r_outer, hitFractionb1, cos1, hitFractionb2, cos2])
#
#         return icosphere_features
#
#     def get_wrist_poses(self, pos, v_norm):
#         v_norm = np.array(v_norm)
#         v_norm = v_norm / np.linalg.norm(v_norm)
#
#         # draw_sphere(0.1, pos)
#
#         align_norm = Pose(pos, euler=(0, math.acos(v_norm[2]), math.atan2(v_norm[1], v_norm[0])))
#         swap = Pose(euler=(-np.pi / 2, 0, 0))
#
#         poses = []
#
#         for i in range(8):
#             rotation = Pose(euler=(0, np.pi * 0.25 * i, 0))  # 8 trails
#             wrist_pose = multiply(align_norm, swap, rotation, Pose(point=(0., -0.11, -0.12)))
#             poses.append(wrist_pose)
#
#         return poses
#
#     def test_reachability(self, pos, v_norm):
#         r = 0
#         wrist_poses = self.get_wrist_poses(pos, v_norm)
#         for p in wrist_poses:
#             set_pose(self.wrist, p)
#             if self.visualization:
#                 time.sleep(0.1)
#
#             no_collision = not any(pairwise_collision(self.wrist, b,
#                                                       visualization=False,
#                                                       max_distance=0.)
#                                    for b in self.all_bodies)
#             if no_collision:
#                 r = 1
#                 break
#
#         self.remove_wrist()
#         return r
#
#     def get_ee_poses(self, pos, v_norm):
#         wrist_poses = self.get_wrist_poses(pos, v_norm)
#         ee_poses = []
#         for wp in wrist_poses:
#             set_pose(self.wrist, wp)
#             ep = get_link_pose(self.wrist, link_from_name(self.wrist, 'ee_link'))
#             ee_poses.append(ep)
#
#         self.remove_wrist()
#         return ee_poses
#
#     def get_ik(self, pos, v_norm):
#         ee_poses = self.get_ee_poses(pos, v_norm)
#         sample_fn = get_sample_fn(self.robot, self.movable_joints)
#         for ee_pose in ee_poses:
#             for _ in range(10):
#                 sampled_conf = sample_fn()
#                 set_joint_positions(self.robot, self.movable_joints, sampled_conf)  # Random seed
#
#                 q_approach = inverse_kinematics(self.robot, link_from_name(self.wrist, 'ee_link'), ee_pose)
#
#                 if q_approach:
#                     """Reachable"""
#                     set_joint_positions(self.robot, self.movable_joints, q_approach)
#                     no_collision = not any(pairwise_collision(self.robot, b,
#                                                               visualization=False,
#                                                               max_distance=0) for b in self.all_bodies)
#                     if no_collision:
#                         return q_approach
#         return None
#
#     def solve_ik_random(self):
#         outer_vertices, unit_vertices, outer_ends = self.get_icosphere_info(self.target_obj)
#         for v, v_norm in zip(outer_vertices, unit_vertices):
#             q_approach = self.get_ik(v, v_norm)
#             if q_approach:
#                 return q_approach
#         return None
#
#     def solve_ik_heuristic(self, v_heuristic):
#         outer_vertices, unit_vertices, outer_ends = self.get_icosphere_info(self.target_obj)
#         for v, v_norm, h in zip(outer_vertices, unit_vertices, v_heuristic):
#             if v_heuristic == 1:
#                 q_approach = self.get_ik(v, v_norm)
#                 if q_approach:
#                     return q_approach
#         return None
#
#     def get_icosphere_info(self, obj):
#         unit_vertices = icosphere(level=1).vertices  # radius=1
#         centroid = np.array(get_pose(obj)[0])
#         r_outer = self.dic_body_info[obj][1] * 1.05
#         outer_vertices = unit_vertices * r_outer + centroid
#         outer_ends = np.dstack([centroid] * len(outer_vertices))[0].transpose((1, 0))
#
#         return outer_vertices, unit_vertices, outer_ends
#
#     def get_labels(self):
#
#         outer_vertices, unit_vertices, outer_ends = self.get_icosphere_info(self.target_obj)
#
#         rays_shape = p.rayTestBatch(outer_vertices.tolist(), outer_ends.tolist())
#
#         labels = []
#         for rs, vtx, v_norm in zip(rays_shape, outer_vertices, unit_vertices):
#             objectUniqueId2, linkIndex2, hitFractionb2, hitPosition2, hitNormal2 = rs
#             if objectUniqueId2 != self.target_obj:
#                 labels.append([0])
#             else:
#                 r = self.test_reachability(vtx, v_norm)
#                 labels.append([r])
#
#         return labels


if __name__ == '__main__':

    print('Loading model...')
    model = model_setup(get_arg())
    model.load_state_dict(torch.load('trained_model_para.pkl'))
    print('...Completed!')

    visualization = True  # True False

    connect(use_gui=visualization)

    dg = DataGenerator(visualization=visualization, train=False)

    list_X = []
    list_labels = []

    list_cost_random = []
    list_cost_heuristic = []

    success_count = 0
    failure_count = 0
    N = 500

    file_result = 'test_propose_ik.pk'

    # with HideOutput():
    for i in range(N):
        dg.reset()
        X = dg.get_X()
        """Random IK search"""
        s = time.time()
        q_approach_random = dg.solve_ik_random()
        cost_random = time.time() - s
        """IK search with learned heuristic"""
        s = time.time()
        with torch.no_grad():
            inputX = torch.FloatTensor([X]).cuda()
            output = torch.squeeze(model(inputX))
            prediction = torch.sigmoid(output).data.round().cpu()
            prediction = prediction.type(torch.int)
        q_approach_heuristic = dg.solve_ik_heuristic(prediction)
        cost_heuristic = time.time() - s

        if (q_approach_random is None and q_approach_heuristic is None) or (
                q_approach_random is not None and q_approach_heuristic is not None):
            """the two IK solvers come to the same results, i.e., not false negative"""
            list_cost_heuristic.append(cost_heuristic)
            list_cost_random.append(cost_random)
            successful_count += 1

        if len(list_cost_random) > 0 and (i + 1) % 10 == 0:
            print("{}, Accuracy rate: {:.3f}, Cost ratio: {:.3f}".format(i, successful_count / float(i + 1),
                                                                         sum(list_cost_heuristic) / sum(
                                                                             list_cost_random)))
            with open(file_result, 'wb') as f:
                pk.dump((N, successful_count, list_cost_random, list_cost_heuristic), f)

    disconnect()
    print("Accuracy rate: {:.3f}, Cost ratio: {:.3f}".format(successful_count / float(N),
                                                             sum(list_cost_heuristic) / sum(
                                                                 list_cost_random)))
    with open(file_result, 'wb') as f:
        pk.dump((N, successful_count, list_cost_random, list_cost_heuristic), f)
    print('Finished.')
