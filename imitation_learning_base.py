#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class definition for Imitation Learning algorithms.
"""

import os
from typing import List

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from dtw import dtw

from utils_learning import tp, demo
from utils_learning import remove_similar_points, smooth_curve, cal_frame_pose, interpolate_orien


class ImitationLearningBase:
    """
    Base class for Imitation Learning (IL) algorithms.

    This class initializes the essential parameters and organizes datasets
    for imitation learning tasks.
    """
    def __init__(self, root_folder: str,
                 known_tp_pose: np.ndarray, local_tp_pose: np.ndarray,
                 training_demo_id: List[int], validation_demo_id: List[int], 
                 num_frame: int = 2, sim_exp: bool = False):
        '''
        Initialize the base class for imitation learning.

        Parameters
        ----------
        root_folder : str
            the folder containing recorded demonstrations
        known_tp_pose : np.ndarray
            to define the pose of the starting frame,
            vector of length 6, position(3)+orientation(3, extrinsic euler angle)
        local_tp_pose : np.ndarray
            to define the local pose of the ending frame relative to the robot end-effector,
            vector of length 6, position(3)+orientation(3, extrinsic euler angle)
        training_demo_id : List[int]
            training datasets
        validation_demo_id : List[int]
            validation datasets
        num_frame : int, optional
            the number of reference frames, by default 2
        sim_exp : bool, optional
            to specify if it is a simulation task, by default False,
            it determines how to deal with collected demonstrations for dataset generation
        '''
        self.root_folder = root_folder
        self.demo_folder = os.path.join(root_folder, "learning_demos")
        self.homo_matrix = np.load(os.path.join(root_folder, "left_to_right.npy"))
        self.known_tp_pose = known_tp_pose
        self.local_tp_pose = local_tp_pose
        self.training_demo_id = training_demo_id
        self.validation_demo_id = validation_demo_id
        self.num_frame = num_frame
        self.sim_exp = sim_exp

        self.num_demo = 0
        self.num_static_demo = 0
        self.num_dynamic_demo = 0

        self.demo_file = []
        self.static_demo_file = []
        self.dynamic_demo_file = []
        self.demo_dataset = []

        self.sort_demo_file()
        self.generate_demo_set(no_repeat=True, smooth=False)

    def sort_demo_file(self):
        '''
        Sort the demo files by the indexes in their names
        '''
        demo_file = pd.Series(os.listdir(self.demo_folder))
        static_demo_selection = demo_file.str.startswith("static").to_list()
        self.static_demo_file = demo_file.loc[static_demo_selection].to_list()
        dynamic_demo_selection = demo_file.str.startswith("dynamic").to_list()
        self.dynamic_demo_file = demo_file.loc[dynamic_demo_selection].to_list()
        sorting_key = lambda x:int(x.split("_")[-1][:-4]) # e.g., static_1.npy
        self.static_demo_file.sort(key=sorting_key)
        self.dynamic_demo_file.sort(key=sorting_key)
        self.num_static_demo = len(self.static_demo_file)
        self.num_dynamic_demo = len(self.dynamic_demo_file)
        self.demo_file = self.static_demo_file+self.dynamic_demo_file
        self.num_demo = len(self.demo_file)

        print("=============================================")
        print(f"There are {self.num_demo} demos in total.")
        print(f"{self.num_static_demo} of them are static.")
        print(f"{self.num_dynamic_demo} of them are dynamic.")
        print(f"The training demos are {[self.demo_file[i] for i in self.training_demo_id]}.")
        print("=============================================")


    def generate_demo_set(self, no_repeat: bool, smooth: bool):
        '''Process collected demonstrations to generate demo objects

        Parameters
        ----------
        no_repeat : bool
            whether to remove very close points from recorded trajectories
        smooth : bool
            whether smooth recorded trajectories
        '''
        known_tp_A = R.from_euler("xyz", self.known_tp_pose[3:], degrees=True).as_matrix()
        known_tp = tp(known_tp_A, self.known_tp_pose[:3].reshape((-1, 1)))
        for f in self.demo_file:
            # generate demo arrays, np.array([num_dim, num_timesteps]), num_dim is 14 or 16
            # num_dim = 14 ==> [left_arm_pos, left_arm_orien(quat), right_arm_pos, right_arm_orien(quat)]
            if self.sim_exp:
                # for simulation, the homo matrix is a identity matrix
                demo_dict = loadmat(os.path.join(self.demo_folder, f)) # dict_keys(['__header__', '__version__', '__globals__', 'demo'])
                demo_matlab = demo_dict["demo"].squeeze() # np.array([num_timesteps]), each entry is the type of <class 'numpy.void'>
                demo_data = np.zeros((14, demo_matlab.shape[0]))
                for i, d in enumerate(demo_matlab):
                    d_list = list(d)
                    ''' 
                    list of arrays [smooth_data, noisy_data, frame1_orientation, frame1_position, frame2_orientation, frame2_position, goal]
                    e.g., [array([[-0.8],[-0.83162278]]),
                           array([[-0.8], [-0.83162278]]),demo_data 
                           array([[0]], dtype=uint8), 
                           array([[-0.8], [-0.8]]),
                           array([[108]], dtype=uint8), 
                           array([[0.00363636], [0.09393939]]),
                           array([[0.03371141], [0.10371137]])]
                    '''
                    demo_data[:2, i] = d_list[5].squeeze() # frame 2 position
                    # the frame angle is relative to the positive y-axis, change it to be relative to the positive x-axis
                    demo_data[3:7, i] = R.from_euler("z", d_list[4].squeeze()+90, True).as_quat() # frame 2 orientation
                    demo_data[7:9, i] = d_list[0].squeeze() # demo trajectory
            else:
                demo_data = np.load(os.path.join(self.demo_folder, f))
            demo_len = demo_data.shape[1]

            # the length of pose vector is 8 or 7 (with or without timestamp)
            if demo_data.shape[0] % 2 == 0:
                pose_length = int(demo_data.shape[0]/2)
            else:
                raise ValueError(f"The shape of data is {demo_data.shape}")

            # data pre-processing
            if no_repeat:
                # pay attention to the distance threshold for different tasks
                selected_index = remove_similar_points(demo_data[pose_length:pose_length+3])
                demo_data = demo_data[:, selected_index]
                demo_len = demo_data.shape[1]
            left_pose = demo_data[:pose_length, :][:7, :]
            # For robot experiments, the end-effector pose of the left robot is relative to its base.
            # We need to transform them to the right robot base frame.
            # Two robot bases have the same orientation but different position, so we just need to change positions.
            left_pose[:3, :] = self.homo_matrix[:3, :]@np.vstack((left_pose[:3, :], np.ones(demo_len)))
            right_pose = demo_data[pose_length:][:7, :]
            if smooth:
                right_pose[:3, :] = smooth_curve(right_pose[:3, :])

            # generate demo object
            tp_mat = np.empty(shape=(self.num_frame, demo_len), dtype=object)
            for i in range(demo_len):
                tp_mat[0, i] = known_tp
                tp_mat[1, i] = cal_frame_pose(left_pose[:, i], self.local_tp_pose)

            self.demo_dataset.append(demo(tp_mat, right_pose))

    def vis_reproduction(self, repro, get_obj_fun, plot_obj_fun, with_truth=True, vis_obj=False, exp_2d=False):
        '''visualize reproductions of models

        Parameters
        ----------
        repro : List[demo]
            list of reproductions
        get_obj_fun : function
            the function to calculate the pose of the manipulated object (end-effector ---> object)
        plot_obj_fun : function
            the function to plot the vase (flower-in-vase task) or holder (roller-on-holder task) 
        with_truth : bool, optional
            whether to plot the ground truth, by default True
        vis_obj : bool, optional
            whether to visualize the trajectory of the manipulated object, by default False
        exp_2d : bool, optional
            to specify if it is a 2D task, by default False
        '''
        num_repro = len(repro)
        num_col = int(np.ceil(num_repro/2))
        # we use pyplot for 2D tasks and pyvista for 3D tasks
        if exp_2d:
            _, ax = plt.subplots(2, num_col, figsize=[30, 15])
            if num_col == 1:
                ax = ax[:, None]
            for i, repro_demo in enumerate(repro):
                sub_ax = ax[i%2, i//2] # [0, 0], [1, 0], [0, 1], [1, 1], ...
                fig_title = f"Demo {self.validation_demo_id[i]}"

                sub_ax.plot(repro_demo.path_pos[0, :], repro_demo.path_pos[1, :], "g", linewidth=2, label="reproduction", zorder=5)

                plot_obj_fun(sub_ax, repro_demo.tp_mat[0, 0], right_robot=True)
                plot_obj_fun(sub_ax, repro_demo.tp_mat[1, 0], right_robot=False)
                if with_truth:
                    truth_demo = self.demo_dataset[self.validation_demo_id[i]]
                    sub_ax.plot(truth_demo.path_pos[0, :], truth_demo.path_pos[1, :], "r", linewidth=2, label="ground truth")
                    dtw_similarity = dtw(repro_demo.path_pos[:2, :].T, truth_demo.path_pos[:2, :].T).normalizedDistance
                    fig_title = f"Demo {self.validation_demo_id[i]} & DTW similarity: {np.around(dtw_similarity, 6)}"

                sub_ax.set_title(fig_title)
                sub_ax.axis("equal") # important to patch drawing
                sub_ax.axis("square")
                sub_ax.set_facecolor("white")
                sub_ax.grid(False)
                sub_ax.legend(facecolor="white")
                sub_ax.spines['top'].set_color("black")
                sub_ax.spines['bottom'].set_color("black")
                sub_ax.spines['left'].set_color("black")
                sub_ax.spines['right'].set_color("black")
            plt.show()
        else:
            pl = pv.Plotter(window_size=[3000, 2000], shape=[2, num_col])
            pl.background_color = "white"
            for i, repro_demo in enumerate(repro):
                truth_demo = self.demo_dataset[self.validation_demo_id[i]]
                truth_orien = truth_demo.path_orien
                truth_gripper_path = truth_demo.path_pos

                # add the orientation for the reproduction
                repro_gripper_path = repro_demo.path_pos
                path_len = repro_gripper_path.shape[1]

                repro_demo.path_orien = interpolate_orien(truth_orien[:, 0], truth_orien[:, -1], path_len)
                repro_orien = repro_demo.path_orien


                # # ================= TODO: only for flower-in-vase task =====================
                # from utils_learning import cal_gripper_pose
                # # calculate gripper pose using frame pose
                # gripper_pose = cal_gripper_pose(repro_demo.tp_mat[1, 0], self.local_tp_pose)
                # # calculate right robot gripper pose using (left) gripper pose
                # T = R.from_euler("Y", -13, degrees=True).as_matrix()
                # left_orien_mat = R.from_quat(gripper_pose[3:]).as_matrix()
                # right_orien_mat = left_orien_mat@T
                # right_orien_quat_final = R.from_matrix(right_orien_mat).as_quat()

                # # calculate the initial right robot gripper pose
                # right_orien_quat_initial = R.from_euler("xyz", [180, -90, 60], degrees=True).as_quat()

                # repro_demo.path_orien = interpolate_orien(right_orien_quat_initial, right_orien_quat_final, path_len)
                # repro_orien = repro_demo.path_orien
                # # =================================================================================

                if vis_obj:
                    repro_obj_path = get_obj_fun(np.vstack((repro_gripper_path, repro_orien)))
                    repro_mesh = pv.MultipleLines(repro_obj_path.T)
                    truth_obj_path = get_obj_fun(np.vstack((truth_gripper_path, truth_orien)))
                    truth_mesh = pv.MultipleLines(truth_obj_path.T)
                else:
                    repro_mesh = pv.MultipleLines(repro_gripper_path.T)
                    truth_mesh = pv.MultipleLines(truth_gripper_path.T)

                pl.subplot(i%2, i//2) # [0, 0], [1, 0], [0, 1], [1, 1], ...
                fig_title = f"Demo {self.validation_demo_id[i]}"
                if with_truth:
                    pl.add_mesh(truth_mesh, line_width=6, color="r", label="ground truth")
                    dtw_similarity = dtw(truth_gripper_path.T, repro_gripper_path.T).normalizedDistance
                    fig_title = f"Demo {self.validation_demo_id[i]} & DTW similarity: {np.around(dtw_similarity, 6)}"
                pl.add_mesh(repro_mesh, line_width=6, color="g", label="reproduction")
                plot_obj_fun(pl, repro_demo.tp_mat[0, 0], right_robot=True)
                plot_obj_fun(pl, repro_demo.tp_mat[1, 0], right_robot=False)
                # pl.add_legend(size=(0.3, 0.3), bcolor="w")

                pl.add_title(fig_title, color="black")
                pl.camera.azimuth = -45.0
                pl.camera.elevation = -25
            pl.show()

    def calculate_repro_loss(self, repro: List[demo]):
        '''
        calculate the model errors using DTW distance between reproductions and ground truths
        '''
        dtw_similarity = 0
        for i, repro_demo in enumerate(repro):
            truth_demo = self.demo_dataset[self.validation_demo_id[i]]
            dtw_similarity += dtw(repro_demo.path_pos.T, truth_demo.path_pos.T).normalizedDistance
        # TODO: use average value for three_frame_simulation and dressing_exp
        return dtw_similarity/len(repro)

    def vis_repo_one_image(self, plot_obj_fun, repro: List[demo]=None):
        '''visualize reproductions together in a figure, only for 2D experiments

        Parameters
        ----------
        plot_obj_fun : function
            the function to plot the box (pen-to-box task)
        repro : List[demo], optional
            list of reproductions, by default None
            None for training and validation sets visualization
            Not None for the visualization of reproductions on the validation set
        '''
        _, ax = plt.subplots(figsize=[15, 15])
        # display training and validation sets
        if repro==None:
            for i, id in enumerate(self.training_demo_id+self.validation_demo_id):
                truth_demo = self.demo_dataset[id]
                if i < len(self.training_demo_id):
                    ax.plot(truth_demo.path_pos[0, :], truth_demo.path_pos[1, :], "red", linewidth=5, zorder=40) # label=str(id)
                else:
                    ax.plot(truth_demo.path_pos[0, :], truth_demo.path_pos[1, :], "green", linewidth=5, zorder=40) # label=str(id)
                plot_obj_fun(ax, truth_demo.tp_mat[0, 0], right_robot=True)
                plot_obj_fun(ax, truth_demo.tp_mat[1, 0], right_robot=False, alpha=0.6)
        # display reproductions compared to ground truths
        else:
            for i, id in enumerate(self.validation_demo_id):
                truth_demo = self.demo_dataset[id]
                ax.plot(truth_demo.path_pos[0, :], truth_demo.path_pos[1, :], "red", linewidth=5, zorder=40)
                ax.plot(repro[i].path_pos[0, :], repro[i].path_pos[1, :], "green", linewidth=5, zorder=50)
                plot_obj_fun(ax, truth_demo.tp_mat[0, 0], right_robot=True)
                plot_obj_fun(ax, truth_demo.tp_mat[1, 0], right_robot=False, alpha=0.6)

        ax.axis("equal") # important to patch drawing
        ax.axis("square")
        ax.set_facecolor("white")
        ax.grid(False)
        ax.spines['top'].set_color("black")
        ax.spines['bottom'].set_color("black")
        ax.spines['left'].set_color("black")
        ax.spines['right'].set_color("black")

        plt.show()