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
    pairwise_collision

from utils.pybullet_tools.body_utils import draw_frame, place_objects

from copy import copy
from mesh.mesh import icosphere
from train_DH import get_arg, model_setup
import torch.nn.functional as F
from generate_dataset_cembre import DataGenerator

if __name__ == '__main__':

    print('Loading model...')
    model = model_setup(get_arg())
    model.load_state_dict(torch.load('trained_model_para.pkl'))
    print('...Completed!')

    visualization = True

    connect(use_gui=visualization)

    dg = DataGenerator(visualization=visualization)

    list_X = []
    list_labels = []

    # with HideOutput():
    for i in range(10):
        dg.reset()
        X = dg.get_X()

        with torch.no_grad():
            inputX = torch.FloatTensor([X]).cuda()
            output = torch.squeeze(model(inputX))
            prediction = torch.sigmoid(output).data.round().cpu()
            prediction = prediction.type(torch.int)

        print('pred: ', prediction.tolist())
        label = np.array(dg.get_labels()).squeeze().tolist()
        print('true: ', label)
        print()
        time.sleep(1.)

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
