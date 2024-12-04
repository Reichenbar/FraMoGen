#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The class definition for Frame-weighted motion generation method.
"""

import sys
import os
from typing import List

import numpy as np
from itertools import combinations, permutations
from scipy.spatial.transform import Rotation as R
from dtw import dtw
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from imitation_learning_base import ImitationLearningBase
from utils_learning import demo
from utils_learning import cal_local_demo, cal_local_tp, calculate_dist_index, radial_basis

class FraMoGen(ImitationLearningBase):
    '''
    Frame-weighted motion generation method
    '''
    def __init__(self, root_folder: str,
                 known_tp_pose: np.ndarray, local_tp_pose: np.ndarray,
                 training_demo_id: List[int], validation_demo_id: List[int],
                 num_frame: int = 2, num_rbf_component: int = 10, con_optim: bool = True,
                 sim_exp: bool = False):
        '''initialize the FraMoGen class

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
        num_rbf_component : int, optional
            the number of rbf components for frame weights approximation, by default 10
        con_optim : bool, optional
            during optimization, whether to add the constraint that the frame weight is between 0 and 1, by default True
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
        self.num_rbf_component = num_rbf_component
        self.num_frame = num_frame
        self.con_optim = con_optim
        self.sim_exp = sim_exp
        
        self.num_demo = 0
        self.num_static_demo = 0
        self.num_dynamic_demo = 0

        self.demo_file = []
        self.static_demo_file = []
        self.dynamic_demo_file = []

        self.demo_dataset = []
        self.fwbil_dataset = []
        self.rbf_weight = np.array([])

        self.sort_demo_file()
        self.generate_demo_set(no_repeat=True, smooth=False)
        self.optimize_rbf_weight()


    def reproduce_using_single_reference(self, ref_demo: demo, des_tp: np.ndarray, rbf_weight: np.ndarray = np.array([])) -> demo:
        '''Reproduce skills in a new situation by transforming a reference trajectory using frame weights

        Parameters
        ----------
        ref_demo : demo
            the reference trajectory
        des_tp : np.ndarray
            to describe two reference frames in the new situation, np.array([2])
        rbf_weight : np.ndarray, optional
            the component weight of each rbf function, np.array([num_rbf_component]), by default np.array([])

        Returns
        -------
        demo
            the reproduction is a demo object which does not contain orientation information
        '''
        # Step 1: project the demonstration and reference frames from the global coordinate system into the local frame {1}
        local_ref_demo = cal_local_demo(ref_demo)
        local_target_tp = cal_local_tp(des_tp[0], des_tp[1])
        if rbf_weight.size == 0:
            rbf_weight = self.rbf_weight
        demo_len = local_ref_demo.path_pos.shape[1]

        # Step 2: transformation based on the position difference of frame {2}
        ref_tp = local_ref_demo.tp_mat[:, 0]
        target_pos_diff = local_target_tp.b-ref_tp[1].b
        dist_index = calculate_dist_index(local_ref_demo.path_pos)
        point_weight = np.sum(radial_basis(dist_index, self.num_rbf_component)*rbf_weight, axis=1)
        repro_path = local_ref_demo.path_pos+np.tile(target_pos_diff, (1, demo_len))*point_weight

        # Step 3: transformation based on the orientation difference of frame {2}
        target_orien_diff = local_target_tp.A@np.linalg.inv(ref_tp[1].A)
        rot_vec = R.from_matrix(target_orien_diff).as_rotvec() # its norm represents the angle
        point_rot_vec = point_weight*np.tile(rot_vec.reshape((-1, 1)), (1, demo_len))
        point_rot_mat = R.from_rotvec(point_rot_vec.T).as_matrix()
        for i in range(repro_path.shape[1]):
            rot_center = local_target_tp.b.squeeze()
            repro_path[:, i] = point_rot_mat[i]@(repro_path[:, i]-rot_center)+rot_center
        
        # Step 4: transform the reproductions back to the global coordinate frame
        repro_path = des_tp[0].A@repro_path+des_tp[0].b # np.array([3, num_timesteps]), no orientation information

        return demo(np.tile(des_tp.reshape((-1, 1)), (1, demo_len)), repro_path)



    def calculate_similarity(self, rbf_weight: np.ndarray):
        '''
        calculate the model training error
        '''
        # generate demo pairs for mutual reproductions, permutations better than combinations
        demo_pair = list(permutations(self.training_demo_id, 2))
        dtw_similarity = 0
        for p in demo_pair:
            ref_demo = self.demo_dataset[p[0]]
            truth_demo = self.demo_dataset[p[1]]
            truth_tp = truth_demo.tp_mat[:, 0]
            repro = self.reproduce_using_single_reference(ref_demo, truth_tp, rbf_weight)
            # dtw need to have the same column
            dtw_similarity += dtw(repro.path_pos.T, truth_demo.path_pos.T).normalizedDistance
        return dtw_similarity

    def get_optim_con(self, soft_con=False):
        '''
        the function to get constraints for optimization
        '''
        index = np.linspace(0, 1, 100)
        # some of frame weights are between 0 and 1
        # the performance using soft constraints is proven to be worse
        if soft_con:
            def con_fun(rbf_weight: np.ndarray):
                component_value = radial_basis(index, self.num_rbf_component)
                component_sum = np.sum(component_value*rbf_weight, axis=1)
                right_num = np.sum(component_sum>0)+np.sum(component_sum<1)
                print(right_num)
                return right_num-0.9*2*index.shape[0]
            con = dict()
            con["type"] = "ineq"
            con["fun"] = con_fun
            return con
        # all the frame weights are between 0 and 1
        else:
            def con_fun_lower(rbf_weight: np.ndarray, i: int):
                component_value = radial_basis(np.array([i]), self.num_rbf_component)
                component_sum = float(np.sum(component_value*rbf_weight, axis=1).squeeze())
                return component_sum
            def con_fun_upper(rbf_weight: np.ndarray, i: int):
                component_value = radial_basis(np.array([i]), self.num_rbf_component)
                component_sum = float(np.sum(component_value*rbf_weight, axis=1).squeeze())
                return 1-component_sum
            con_list = []
            for i in index:
                c_lower = dict()
                c_lower["type"] = "ineq"
                c_lower["fun"] = con_fun_lower
                c_lower["args"] = [i]
                con_list.append(c_lower)
                c_upper = dict()
                c_upper["type"] = "ineq"
                c_upper["fun"] = con_fun_upper
                c_upper["args"] = [i]
                con_list.append(c_upper)
            return con_list


    def optimize_rbf_weight(self):
        '''
        get optimal frame weights through optimizing rbf weights
        '''
        # Using constraints seems better
        initial_rbf_weight = np.ones((self.num_rbf_component))*0.5
        result = minimize(self.calculate_similarity,
                            initial_rbf_weight, method='BFGS')
        if self.con_optim:
            initial_rbf_weight = result.x # use the solution without constraints as the initialization
            result = minimize(self.calculate_similarity, initial_rbf_weight,
                              constraints=self.get_optim_con(soft_con=False), method='COBYLA')
        print("Optimization finished.")
        self.rbf_weight = result.x
        return result.x


    def vis_frame_weight(self, rbf_weight=np.array([]), with_component: bool=True):
        '''
        visualize optimal frame weights
        '''
        if rbf_weight.size == 0:
            rbf_weight = self.rbf_weight
        _, ax = plt.subplots(figsize=[10, 10])
        index = np.linspace(0, 1, 100)
        component_value = radial_basis(index, self.num_rbf_component)
        component_sum = np.sum(component_value*rbf_weight, axis=1) # each column is a component
        ax.plot(index, component_sum, color=np.array([0, 166, 214])/255, linewidth=4, label="frame weight")
        if with_component:
            ax.plot(index, component_value, "--")
        ax.axis("equal")
        fs = 30
        # ax.set_title("Frame weight", fontsize=fs)
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=fs, color="black")
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=fs, color="black")
        plt.xlabel("relative distance indicator", fontsize=fs, color="black")
        plt.ylabel("frame relevance weight", fontsize=fs, color="black")
        ax.spines['top'].set_color("black")
        ax.spines['bottom'].set_color("black")
        ax.spines['left'].set_color("black")
        ax.spines['right'].set_color("black")
        ax.grid(False)
        # ax.legend(facecolor="white", fontsize=fs)
        ax.set_facecolor("white")
        plt.show()

    def reproduce_multiple_demos(self, des_tp_set=[], ref_demo_index=None, num_state=None, rbf_weight: np.ndarray = np.array([])):
        repro = []
        s = []
        if len(des_tp_set) == 0:
            des_tp_set = [self.demo_dataset[i].tp_mat[:, 0] for i in self.validation_demo_id]         
        for des_tp in des_tp_set:
            if ref_demo_index is not None:
                ref_demo_0 = self.demo_dataset[ref_demo_index]
            else:
                ref_demo_0 = self.demo_dataset[self.training_demo_id[0]]
            # ref_demo_1 = self.demo_dataset[self.training_demo_id[1]]
            # self.reproduce_using_multiple_references(des_tp)
            if rbf_weight.size != 0:
                repro.append(self.reproduce_using_single_reference(ref_demo_0, des_tp, rbf_weight))
            else:
                repro.append(self.reproduce_using_single_reference(ref_demo_0, des_tp))
            # repro.append(self.reproduce_using_single_reference(None, des_tp)) # using all differences
        return repro