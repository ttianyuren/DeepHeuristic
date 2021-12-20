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
    set_camera, get_center_extent, tform_from_pose, attach_viewcone, LockRenderer, p, set_color, draw_point

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

    draw_sphere(dic_body_info[obj1][1], get_pose(obj1)[0], [0.9, 0.9, 0.1, 0.5])

    unit_vertices = icosphere(level=1).vertices  # radius=1
    centroid = np.array(get_pose(obj1)[0])
    outer_vertices = unit_vertices * dic_body_info[obj1][1] + centroid
    outer_ends = np.dstack([centroid] * len(outer_vertices))[0].transpose((1, 0))
    r_inner = dic_body_info[obj1][1] * 0.25
    inner_vertices = unit_vertices * r_inner + centroid
    inner_ends = unit_vertices * (0.6 + r_inner) + centroid
    draw_pointCloud(outer_ends, color=[1, 1, 0])
    draw_pointCloud(inner_ends, color=[0, 1, 1])

    for i in range(10000):
        step_simulation()
        time.sleep(0.1)

    disconnect()
    print('Finished.')


if __name__ == '__main__':
    display_scenario()
