#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flower-in-vase task.
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
import pyvista as pv

from utils_learning import tp
from frame_weighted_motion_generation import FraMoGen


def gripper_to_flower(right_gripper_pose):
    flower_local_coord = np.array([-0.217, 0, 0.054]) # the local coordinates of flower end in the gripper frame
    gripper_orien_mat = R.from_quat(right_gripper_pose[3:, :].T).as_matrix()
    flower_global_coord = gripper_orien_mat@flower_local_coord.reshape((-1, 1))
    return flower_global_coord.squeeze().T+right_gripper_pose[:3, :]

def plot_vase(pl, frame_pose: tp, right_robot=True, colors=None,
              with_bottom=True,
              body_height=0.2, inner_radius=0.04/np.sqrt(3), outer_radius=0.046/np.sqrt(3),
              bottom_height=0.005, bottom_radius=0.07/np.sqrt(3)):
    gripper_orien_mat = frame_pose.A
    vase_orien_mat = gripper_orien_mat
    vase_z_vec = vase_orien_mat[:, -1] # direction perpendicular to the gripper
    vase_body_centroid = frame_pose.b.squeeze()+vase_z_vec*(body_height/2)
    if right_robot:
        vase_color = ["cyan", "royalblue"]
    else:
        vase_color = ["orangered", "royalblue"]
    if colors != None:
        vase_color = colors
    vase_body = pv.CylinderStructured([inner_radius, outer_radius], body_height, vase_body_centroid, vase_z_vec, 7)
    pl.add_mesh(vase_body, show_edges=False, color=vase_color[0])

    if with_bottom:
        vase_bottom_centroid = vase_body_centroid-vase_z_vec*(body_height/2+bottom_height/2)
        vase_bottom = pv.CylinderStructured([0, bottom_radius], bottom_height, vase_bottom_centroid, vase_z_vec, 7)
        pl.add_mesh(vase_bottom, show_edges=False, color=vase_color[1])


def main(root_folder, known_tp_pose, local_tp_pose, training_set, validation_set, generalization):
    training_demo_id = training_set
    if generalization == "training":
        validation_demo_id = training_set
    elif generalization == "validation":
        validation_demo_id = validation_set
    model = FraMoGen(root_folder, known_tp_pose, local_tp_pose, training_demo_id, validation_demo_id) #TODO: model
    repro = model.reproduce_multiple_demos()
    print(f"The {generalization} loss is {model.calculate_repro_loss(repro)}.")
    model.vis_reproduction(repro, gripper_to_flower, plot_vase, vis_obj=False, with_truth=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conduct the flower-in-vase task")
    parser.add_argument("-g", "--generalization", default="validation", type=str, choices = ["training", "validation"])
    args = parser.parse_args()

    root_folder = os.path.join(os.path.dirname(__file__), "3D_flower")
    num_state = 6
    known_tp_pose = np.array([0.688, 0.092, 0.003, 0, 0, 0]) # the pose of the starting vase
    local_tp_pose = np.array([-0.0325, 0, 0, 0, 90, 0])
    training_demo_id = [4, 7]
    validation_demo_id = [10, 11, 12, 13]
    augmen_demo_id = [0, 1, 2, 3, 5, 8, 9]

    main(root_folder, known_tp_pose, local_tp_pose, training_demo_id, validation_demo_id, args.generalization)