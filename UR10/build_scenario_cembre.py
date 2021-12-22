#!/usr/bin/env python

from __future__ import print_function

import math
import os
import numpy as np
import pickle as pk

import time
from utils.pybullet_tools.kuka_primitives3 import BodyPose, BodyConf, Register
from utils.pybullet_tools.utils import WorldSaver, connect, dump_world, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, disconnect, get_bodies, HideOutput, \
    create_box, \
    load_pybullet, step_simulation, Euler, get_links, get_link_info, get_movable_joints, set_joint_positions, \
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer, p, set_color, draw_point, multiply, \
    pairwise_collision

from utils.pybullet_tools.body_utils import draw_frame, place_objects

from copy import copy
from mesh.mesh import icosphere


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


class DataGenerator(object):
    def __init__(self, visualization=True):
        env = load_pybullet('/cembre_description/cembre_simplified.urdf', fixed_base=True)

        self.env_bodies = [env]
        self.visualization = visualization
        self.robot = load_pybullet('/cembre_description/ur10e.urdf', fixed_base=True)
        self.ee = load_pybullet('/cembre_description/end_link.urdf', fixed_base=False)

        # set_pose(r, (base_link_pose[0],base_link_pose[1]))
        set_pose(self.robot, ((-0.25499999999999995, 0.06099999999999994, 2.0580000000000003),
                              (0.7071067811882787, 0.7071067811848163, -7.31230107716731e-14, -7.312301077203115e-14)))
        initial_jts = np.array([0.0, -0.1, -2.7, 1.5, 0.5 * np.pi, 0])
        config_left = BodyConf(self.robot, initial_jts)
        config_left.assign()

        region1 = load_pybullet('/cembre_description/region1.urdf', fixed_base=True)
        set_pose(region1, Pose((-0.29, 0, 0.94)))
        region2 = load_pybullet('/cembre_description/region2.urdf', fixed_base=True)
        set_pose(region2, Pose((0.32, 0, 1.067)))

        self.regions = [region1, region2]
        self.region_p = get_area_p(self.regions)

        self.reset_containers()
        self.remove_ee()

    def reset_containers(self):
        self.target_obj = None
        self.movable_bodies = []
        self.all_bodies = list(set(self.env_bodies) | set(self.regions))
        self.dic_body_info = {}

    def sample_region(self):
        idx = np.random.choice(len(self.regions), 1, p=self.region_p)[0]
        # return idx
        return self.regions[idx]

    def remove_ee(self):
        set_pose(self.ee, Pose((0, 0, -20)))

    def reset(self):
        if self.movable_bodies:
            for b in self.movable_bodies:
                p.removeBody(b)
        self.reset_containers()

        obj1 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.859, 0.192, 0.306, 1.0))
        obj2 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.271, 0.706, 0.490, 1.0))
        obj3 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.647, 0.498, 0.894, 1.0))
        self.target_obj = obj1
        self.movable_bodies = [obj1, obj2, obj3]
        self.all_bodies = list(set(self.movable_bodies) | set(self.env_bodies) | set(self.regions))

        for b in self.movable_bodies:
            obj_center, obj_extent = get_center_extent(b)
            r_outSphere = np.linalg.norm(obj_extent) / 2 * 1.01
            body_pose = get_pose(b)
            body_frame = tform_from_pose(body_pose)
            center_frame = tform_from_pose((obj_center, body_pose[1]))
            relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

            self.dic_body_info[b] = (obj_extent, r_outSphere, relative_frame_center)

        region = self.sample_region()
        list_remove = place_objects(self.movable_bodies, region, self.all_bodies)
        for b in list_remove:
            self.movable_bodies.remove(b)
            self.all_bodies.remove(b)

    def get_X(self):
        p_inner_outer = 0.25
        max_range = 0.6  # meters
        unit_vertices = icosphere(level=4).vertices  # radius=1
        centroid = np.array(get_pose(self.target_obj)[0])
        r_inner = self.dic_body_info[self.target_obj][1] * p_inner_outer
        inner_vertices = unit_vertices * r_inner + centroid
        inner_ends = unit_vertices * (max_range + r_inner) + centroid
        # draw_sphere(r_inner, centroid, [0, 1, 1, 0.5])

        """Temporarily move away the objects that are not involved in rayTest"""
        offset = [0, 0, -5]
        obj_pose = get_pose(self.target_obj)
        robot_pose = get_pose(self.robot)
        temp_obj_pose = multiply(Pose(Point(*offset)), obj_pose)
        set_pose(self.target_obj, temp_obj_pose)
        temp_robot_pose = multiply(Pose(Point(*offset)), robot_pose)
        set_pose(self.robot, temp_robot_pose)

        temp_centroid = centroid + np.array(offset)
        r_outer = self.dic_body_info[self.target_obj][1]
        outer_vertices = unit_vertices * r_outer + temp_centroid
        outer_ends = np.dstack([temp_centroid] * len(outer_vertices))[0].transpose((1, 0))
        # draw_sphere(r_outer, temp_centroid, [1, 1, 0, 0.5])

        # draw_pointCloud(outer_ends, color=[0, 1, 1])
        # draw_pointCloud(outer_vertices, color=[1, 1, 0])
        #
        # draw_pointCloud(inner_ends, color=[0, 1, 1])
        # draw_pointCloud(inner_vertices, color=[1, 0, 1])

        rays_env = p.rayTestBatch(inner_vertices.tolist(), inner_ends.tolist())
        rays_shape = p.rayTestBatch(outer_vertices.tolist(), outer_ends.tolist())

        """Put the target object and the robot back"""
        set_pose(self.robot, robot_pose)
        set_pose(self.target_obj, obj_pose)

        icosphere_features = []
        for re, rs, v_norm in zip(rays_env, rays_shape, unit_vertices):
            objectUniqueId1, linkIndex1, hitFractionb1, hitPosition1, hitNormal1 = re
            objectUniqueId2, linkIndex2, hitFractionb2, hitPosition2, hitNormal2 = rs
            cos1 = np.dot(hitNormal1, -v_norm)
            cos2 = np.dot(hitNormal2, v_norm)
            icosphere_features.append([r_outer, hitFractionb1, cos1, hitFractionb2, cos2])

        return icosphere_features

    def test_reachability(self, pos, v_norm):

        v_norm = np.array(v_norm)
        v_norm = v_norm / np.linalg.norm(v_norm)

        # draw_sphere(0.1, pos)

        align_norm = Pose(pos, euler=(0, math.acos(v_norm[2]), math.atan2(v_norm[1], v_norm[0])))
        swap = Pose(euler=(-np.pi / 2, 0, 0))

        r = 0

        for i in range(8):
            rotation = Pose(euler=(0, np.pi * 0.25 * i, 0))  # 8 trails
            set_pose(self.ee, multiply(align_norm, swap, rotation, Pose(point=(0., -0.11, -0.12))))
            if self.visualization:
                time.sleep(0.1)

            no_collision = not any(pairwise_collision(self.ee, b,
                                                      visualization=False,
                                                      max_distance=0.)
                                   for b in self.all_bodies)
            if no_collision:
                r = 1
                break

        self.remove_ee()
        return r

    def get_labels(self):
        unit_vertices = icosphere(level=1).vertices  # radius=1
        centroid = np.array(get_pose(self.target_obj)[0])
        r_outer = self.dic_body_info[self.target_obj][1] * 1.05
        outer_vertices = unit_vertices * r_outer + centroid
        outer_ends = np.dstack([centroid] * len(outer_vertices))[0].transpose((1, 0))

        rays_shape = p.rayTestBatch(outer_vertices.tolist(), outer_ends.tolist())

        labels = []
        for rs, vtx, v_norm in zip(rays_shape, outer_vertices, unit_vertices):
            objectUniqueId2, linkIndex2, hitFractionb2, hitPosition2, hitNormal2 = rs
            if objectUniqueId2 != self.target_obj:
                labels.append([0])
            else:
                r = self.test_reachability(vtx, v_norm)
                labels.append([r])

        return labels


def display_scenario():
    connect(use_gui=True)

    env = load_pybullet('/cembre_description/cembre_simplified.urdf', fixed_base=True)

    env_bodies = [env]

    robot = load_pybullet('/cembre_description/ur10e.urdf', fixed_base=True)
    # set_pose(r, (base_link_pose[0],base_link_pose[1]))
    set_pose(robot, ((-0.25499999999999995, 0.06099999999999994, 2.0580000000000003),
                     (0.7071067811882787, 0.7071067811848163, -7.31230107716731e-14, -7.312301077203115e-14)))
    initial_jts = np.array([0.0, -0.1, -2.7, 1.5, 0.5 * np.pi, 0])
    config_left = BodyConf(robot, initial_jts)
    config_left.assign()

    region1 = load_pybullet('/cembre_description/region1.urdf', fixed_base=True)
    set_pose(region1, Pose((-0.29, 0, 0.94)))
    region2 = load_pybullet('/cembre_description/region2.urdf', fixed_base=True)
    set_pose(region2, Pose((0.32, 0, 1.067)))

    regions = [region1, region2]

    obj1 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.859, 0.192, 0.306, 1.0))
    obj2 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.271, 0.706, 0.490, 1.0))
    obj3 = create_box(get_rand(), get_rand(), get_rand(low=0.14), mass=0.5, color=(0.647, 0.498, 0.894, 1.0))

    movable_bodies = [obj1, obj2, obj3]

    all_bodies = list(set(movable_bodies) | set(env_bodies) | set(regions))

    dic_body_info = {}

    for b in movable_bodies:
        obj_center, obj_extent = get_center_extent(b)
        r_outSphere = np.linalg.norm(obj_extent) / 2 * 1.01
        body_pose = get_pose(b)
        body_frame = tform_from_pose(body_pose)
        center_frame = tform_from_pose((obj_center, body_pose[1]))
        relative_frame_center = np.dot(center_frame, np.linalg.inv(body_frame))

        dic_body_info[b] = (obj_extent, r_outSphere, relative_frame_center)

    list_remove = place_objects(movable_bodies, region1, all_bodies)
    for b in list_remove:
        movable_bodies.remove(b)
        all_bodies.remove(b)

    # visualShapeId = p.createVisualShape(p.GEOM_SPHERE, dic_body_info[obj1][1])
    # outSphere = p.createMultiBody(baseMass=0,
    #                               baseVisualShapeIndex=visualShapeId)
    # set_pose(outSphere, get_pose(obj1))
    # set_color(outSphere, [0.9, 0.5, 0.5, 0.5])

    unit_vertices = icosphere(level=4).vertices  # radius=1
    centroid = np.array(get_pose(obj1)[0])
    r_inner = dic_body_info[obj1][1] * 0.25
    inner_vertices = unit_vertices * r_inner + centroid
    inner_ends = unit_vertices * (0.6 + r_inner) + centroid
    # draw_sphere(r_inner, centroid, [0, 1, 1, 0.5])

    """Temporarily move away the objects that are not involved in rayTest"""
    offset = [0, 0, -5]
    obj_pose = get_pose(obj1)
    robot_pose = get_pose(robot)
    temp_obj_pose = multiply(Pose(Point(*offset)), obj_pose)
    set_pose(obj1, temp_obj_pose)
    temp_obj_pose = multiply(Pose(Point(*offset)), robot_pose)
    set_pose(robot, temp_obj_pose)

    temp_centroid = centroid + np.array(offset)
    r_outer = dic_body_info[obj1][1]
    outer_vertices = unit_vertices * r_outer + temp_centroid
    outer_ends = np.dstack([temp_centroid] * len(outer_vertices))[0].transpose((1, 0))
    # draw_sphere(r_outer, temp_centroid, [1, 1, 0, 0.5])

    # draw_pointCloud(outer_ends, color=[0, 1, 1])
    # draw_pointCloud(outer_vertices, color=[1, 1, 0])
    #
    # draw_pointCloud(inner_ends, color=[0, 1, 1])
    # draw_pointCloud(inner_vertices, color=[1, 0, 1])

    rays_env = p.rayTestBatch(inner_vertices.tolist(), inner_ends.tolist())
    rays_shape = p.rayTestBatch(outer_vertices.tolist(), outer_ends.tolist())

    """Set back those objects"""
    # set_pose(obj1, obj_pose)
    # set_pose(robot, robot_pose)

    for i in range(10000):
        step_simulation()
        time.sleep(0.1)

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    visualization = False

    connect(use_gui=visualization)

    dg = DataGenerator(visualization=visualization)

    list_X = []
    list_labels = []

    # with HideOutput():
    for i in range(9000):
        dg.reset()
        X = dg.get_X()
        label = dg.get_labels()
        list_X.append(X)
        list_labels.append(label)
        if i != 0 and i % 5 == 0:
            print('Generated scene number: ', i)

    list_X = np.array(list_X)
    list_labels = np.array(list_labels)

    file_data = 'Xs_labels.pk'
    with open(file_data, 'wb') as f:
        pk.dump((list_X, list_labels), f)

    # for i in range(10000):
    #     step_simulation()
    #     time.sleep(0.1)

    disconnect()
    print('Finished.')
