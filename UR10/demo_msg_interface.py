#!/usr/bin/env python

from __future__ import print_function


import pickle as pk
import torch

from utils.pybullet_tools.utils import connect, disconnect
from train_DH import get_arg, model_setup

from generate_dataset_cembre import DataGenerator

if __name__ == '__main__':

    print('Loading model...')
    model = model_setup(get_arg())
    model.load_state_dict(torch.load('trained_model_para.pkl'))
    print('...Completed!')

    visualization = True

    connect(use_gui=visualization)

    dg = DataGenerator(visualization=visualization)

    """Input:  
        [(pose,dim) for each movable object]"""
    msg_customBox = dg.get_test_msg()  # TODO: read this from CustomBox.msg. See https://github.com/JRL-CARI-CNR-UNIBS/deep_heuristic_for_grasping_sharework/blob/master/heuristic_grasping_sharework/msg/CustomBox.msg
    dg.input_msg(msg_customBox)

    """Output:  
        [(feasibility_estimate, (approach_point, approach_direction)) for each movable object]"""
    msg_feasibility = []

    for b in dg.movable_bodies:
        dg.target_obj = b
        X = dg.get_X()

        with torch.no_grad():
            inputX = torch.FloatTensor([X]).cuda()
            output = torch.squeeze(model(inputX))
            prediction = torch.sigmoid(output).data.round().cpu()
            prediction = prediction.type(torch.int)

        print('pred: ', prediction.tolist())
        # label = np.array(dg.get_labels()).squeeze().tolist()
        # print('true: ', label)

        outer_vertices, unit_vertices, _ = dg.get_icosphere_info(dg.target_obj)
        msg_feasibility.append((float(sum(prediction)) / float(len(prediction)), (outer_vertices, unit_vertices)))

    print(msg_feasibility) # TODO: send this out as Feasibility.msg. See https://github.com/JRL-CARI-CNR-UNIBS/deep_heuristic_for_grasping_sharework/blob/master/heuristic_grasping_sharework/msg/Feasibility.msg

    disconnect()
    print('Finished.')
