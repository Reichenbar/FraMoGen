#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for Imitation Learning.
"""

from typing import List

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from dtw import dtw
from utils import get_current_pose
import pbdlib as pbd

class tp:
    def __init__(self, A, b):
        self.A = A
        self.b = b

class demo:
    def __init__(self, tp_mat: np.ndarray, path_pose: np.ndarray):
        self.tp_mat = tp_mat  # np.array([num_frame, demo_len], dtype=object)
        self.path_pos = path_pose[:3, :]  # 3*demo_len, [x, y, z]
        if path_pose.shape[0] > 3:
            self.path_orien = path_pose[3:, :]  # 4*demo_len, quaternions
        else:
            self.path_orien = None
        self.dtw_index = np.array(range(path_pose.shape[1]))
        self.sigma = None

class tpgmm_demo:
    def __init__(self, tp_mat, path_pos_f):
        self.tp_mat = tp_mat # np.array([num_frame, demo_len], dtype=object)
        self.path_pos_f = path_pos_f # np.array([num_frame, num_var, demo_len])
        self.path_pos_stack = np.vstack((path_pos_f[0, :1, :], path_pos_f[:, 1:, :].reshape((-1, path_pos_f.shape[2])))) # only one time dim

def cal_local_tp(ref_tp: tp, global_tp: tp):
    ref_homo_mat = np.vstack((np.hstack((ref_tp.A, ref_tp.b)), [0, 0, 0, 1]))
    global_homo_mat = np.vstack((np.hstack((global_tp.A, global_tp.b)), [0, 0, 0, 1]))
    local_homo_mat = np.linalg.inv(ref_homo_mat)@global_homo_mat
    return tp(local_homo_mat[:3, :3], local_homo_mat[:3, -1:])

def cal_local_demo(global_demo: demo):
    path_pos = global_demo.path_pos
    tp_mat = global_demo.tp_mat
    demo_len = path_pos.shape[1]
    num_frame = tp_mat.shape[0]
    tp_mat_local = np.empty(tp_mat.shape, object)
    data_local = np.empty(path_pos.shape)
    for i in range(demo_len):
        tp_mat_local[0, i] = tp(np.identity(3), np.zeros((3, 1)))
        for j in range(1, num_frame):
            tp_mat_local[j, i] = cal_local_tp(tp_mat[0, i], tp_mat[j, i])
        data_local[:, i] = np.squeeze(np.linalg.inv(tp_mat[0, i].A)@(path_pos[:, i:i+1]-tp_mat[0, i].b))
    return demo(tp_mat_local, data_local)

def cal_frame_pose(gripper_pose: np.ndarray, local_pose: np.ndarray):
    # gripper_pose: [x, y, z, quaternions]
    # local_pose: [x, y, z, fixed_angles]
    local_A = R.from_euler("xyz", local_pose[3:], True).as_matrix()
    local_homo = np.vstack((np.hstack((local_A, local_pose[:3, None])), [0, 0, 0, 1]))

    gripper_A = R.from_quat(gripper_pose[3:]).as_matrix()
    gripper_homo = np.vstack((np.hstack((gripper_A, gripper_pose[:3, None])), [0, 0, 0, 1]))

    global_homo = gripper_homo@local_homo

    return tp(global_homo[:3, :3], global_homo[:3, -1:])

def cal_gripper_pose(frame_pose: tp, local_pose: np.ndarray):
    # the inverse operation of cal_frame_pose
    # gripper_pose: tp
    # local_pose: [x, y, z, fixed_angles]
    local_A = R.from_euler("xyz", local_pose[3:], True).as_matrix()
    local_homo = np.vstack((np.hstack((local_A, local_pose[:3, None])), [0, 0, 0, 1]))

    frame_homo = np.vstack((np.hstack((frame_pose.A, frame_pose.b)), [0, 0, 0, 1]))

    gripper_homo = frame_homo@np.linalg.inv(local_homo)
    gripper_quat = R.from_matrix(gripper_homo[:3, :3]).as_quat()

    return np.concatenate((gripper_homo[:3, -1], gripper_quat))

def tp_to_pose(tp: tp):
    target_orien = R.from_matrix(tp.A).as_euler("xyz", degrees=True).tolist()
    target_pos = tp.b.squeeze().tolist()
    return np.array(target_pos+target_orien)

def radial_basis(x, num_component):
    '''
    Define the radial basis function
    '''
    center = np.linspace(0, 1, num_component)
    spread = np.ones(num_component)*5
    return np.exp(-spread*((x.reshape((-1, 1))-center)**2))

def remove_similar_points(demo_path, distance_threshold=4e-3):
    point_distance = np.sqrt(np.sum((demo_path[:, 1:]-demo_path[:, :-1])**2, axis=0))
    selected_index = np.append(True, point_distance > distance_threshold)  # add the first point
    return selected_index


def smooth_curve(demo_path, window_length_ratio=2):
    window_length = int(demo_path.shape[1]/window_length_ratio)
    smooth_demo_path = np.zeros(demo_path.shape)
    for i in range(demo_path.shape[0]):
        smooth_demo_path[i, :] = savgol_filter(demo_path[i, :], window_length, 3)
    return smooth_demo_path

def calculate_dist_index(demo_path):
    # start_pos = demo_path[:, 0].reshape((-1, 1))
    # target_pos = demo_path[:, -1].reshape((-1, 1))
    # dist_to_start = np.sqrt(np.sum((demo_path-start_pos)**2, axis=0))
    # dist_to_target = np.sqrt(np.sum((demo_path-target_pos)**2, axis=0))
    # return dist_to_start/(dist_to_start+dist_to_target)
    
    step_dist = np.cumsum(np.sqrt(np.sum((demo_path[:, 1:]-demo_path[:, :-1])**2, axis=0)))
    step_dist = np.hstack((0, step_dist))
    return step_dist/step_dist[-1]


def compare_dist_index(demo_path):
    # from utils_learning import compare_dist_index
    # for demo in model.demo_dataset:
    #     compare_dist_index(demo.path_pos)

    step_dist = np.cumsum(np.sqrt(np.sum((demo_path[:, 1:]-demo_path[:, :-1])**2, axis=0)))
    step_dist = np.hstack((0, step_dist))
    index = step_dist/step_dist[-1]


    start_pos = demo_path[:, 0].reshape((-1, 1))
    target_pos = demo_path[:, -1].reshape((-1, 1))
    dist_to_start = np.sqrt(np.sum((demo_path-start_pos)**2, axis=0))
    dist_to_target = np.sqrt(np.sum((demo_path-target_pos)**2, axis=0))
    i =  dist_to_start/(dist_to_start+dist_to_target)

    import matplotlib.pyplot as plt
    plt.plot(index, "r", label="cumulative")
    plt.plot(i, "b", label="direct")
    plt.legend()
    plt.show()



def interpolate_orien(start_orien: np.ndarray, end_orien: np.ndarray,
                              path_len: int) -> np.ndarray:
    '''
    start_orien : [quaternions]
    end_orien : [quaternions]
    '''
    gripper_orien = np.empty((4, path_len))
    num_orien_change_step = int(path_len/5)
    start_orien = start_orien.reshape((-1, 1))
    end_orien = end_orien.reshape((-1, 1))

    key_quats = np.concatenate((start_orien.reshape((1, -1)), end_orien.reshape((1, -1))))
    key_rots = R.from_quat(key_quats)
    rot_slerp = Slerp(range(key_quats.shape[0]), key_rots)
    step_orientation_rot = rot_slerp(np.linspace(0, key_quats.shape[0]-1, num_orien_change_step))
    step_orientation_quat = step_orientation_rot.as_quat().T

    step_start = int(path_len/5*2)
    gripper_orien[:, :step_start] = start_orien
    gripper_orien[:, step_start:step_start+num_orien_change_step] = step_orientation_quat
    gripper_orien[:, step_start+num_orien_change_step:] = end_orien

    return gripper_orien

def add_repro_orien(repro_demo: demo, T_left_to_right: np.ndarray):
    path_len = repro_demo.path_pos.shape[1]
    left_orien_quat = np.array(get_current_pose(right_robot=False)[3:])
    left_orien_mat = R.from_quat(left_orien_quat).as_matrix()
    right_orien_mat = left_orien_mat@T_left_to_right
    desired_right_orien_quat = R.from_matrix(right_orien_mat).as_quat()
    right_orien_quat = np.array(get_current_pose(right_robot=True)[3:])
    repro_demo.path_orien = interpolate_orien(right_orien_quat, desired_right_orien_quat, path_len)

def dtw_align_multi(demo_dataset: List[demo]):
    for demo_r in demo_dataset[1:]:
        demo_l = demo_dataset[0]
        dtw_obj = dtw(demo_l.path_pos[:, demo_l.dtw_index].T,
                      demo_r.path_pos[:, demo_r.dtw_index].T)
        demo_l.dtw_index = demo_l.dtw_index[dtw_obj.index1]
        demo_r.dtw_index = demo_r.dtw_index[dtw_obj.index2]
        for demo_pre in demo_dataset[1:demo_dataset.index(demo_r)]:
            demo_pre.dtw_index = demo_pre.dtw_index[dtw_obj.index1]

def add_time_dim(tp_no_t):
    num_dim = tp_no_t.A.shape[0]
    A_with_t = np.identity(num_dim+1)
    b_with_t = np.zeros((num_dim+1, 1))
    A_with_t[1:, 1:] = tp_no_t.A
    b_with_t[1:, :] = tp_no_t.b
    return tp(A_with_t, b_with_t)

def remove_time_dim(tp_with_t):
    A_no_t = tp_with_t.A[1:, 1:]
    b_no_t = tp_with_t.b[1:, :]
    return tp(A_no_t, b_no_t)


def tpgmm_em(local_model, data_stack, num_max_step=100, num_min_step=5, max_diff_ll=1e-5):
    num_state = local_model[0].nb_states
    num_frame, num_var, num_sample = data_stack.shape
    priors = local_model[0].priors
    LL = np.zeros((num_max_step))
    for it in range(num_max_step):
        # E-step
        L = np.ones((num_state, num_sample))
        GAMMA0 = np.zeros((num_state, num_frame, num_sample))
        for i in range(num_state):
            for j in range(num_frame):
                data_f = data_stack[j, :, :] # num_var, num_sample
                GAMMA0[i, j, :] = pbd.multi_variate_normal(data_f.T, local_model[j].mu[i], local_model[j].sigma[i], log=False)
                # print(GAMMA0[i, j, :])
                L[i, :] = L[i, :]*GAMMA0[i, j, :]
                # print(L[i, :10])
            L[i, :] = L[i, :]*priors[i]
        # input("$$$$$$$$$$$$$$$$$$$")
        # print("--------------")
        GAMMA = L / np.sum(L, axis=0)
        # print(GAMMA[:, :10])
        # print("--------------")
        GAMMA2 = GAMMA / np.sum(GAMMA, axis=1)[:, np.newaxis]
        # print(GAMMA2[:, :10])
        # print("--------------")
        # M-step
        priors = np.mean(GAMMA, axis=1)
        for j in range(num_frame):
            data_f = data_stack[j, :, :] # num_var, num_sample
            local_model[j].mu = np.einsum('ac,ic->ai', GAMMA2, data_f)  # a states, c sample, i dim
            dx = data_f[None, :]-local_model[j].mu[:, :, None]  # num_state, num_var, num_sample
            local_model[j].sigma = np.einsum('acj,aic->aij', np.einsum('aic,ac->aci', dx, GAMMA2), dx)  # a states, c sample, i-j dim
            local_model[j].sigma += local_model[j].reg

        # print(L[:, :10])
        LL[it] = np.mean(np.log(np.sum(L, axis=0)))
        # a = np.sum(L, axis=0)
        # print(a[:10])
        # print(np.sum(L<0))
        # b = np.log(a)
        # for i in range(50):
        #     print(b[i*100:(i+1)*100])  
        # print(LL[it])
        # input("&&&")
        # Check for convergence
        if it > num_min_step:
            if LL[it]-LL[it - 1] < max_diff_ll:
                for j in range(num_frame):
                    local_model[j].priors = priors
                print(f'EM Converged after {it+1} iterations.')
                return GAMMA
        # import sys
        # sys.exit(0)
    print("GMM did not converge before reaching max iteration. Consider augmenting the number of max iterations.")
    return GAMMA
