#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for robot experiments.
"""

import time
import rospy
import numpy as np
import dynamic_reconfigure.client
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from franka_gripper.msg import MoveActionGoal

def control_gripper(gripper_publisher, width, speed=0.04):
    # initialize the message  
    msg = MoveActionGoal()

    msg.header.seq = 0
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = ""

    msg.goal_id.stamp = rospy.Time.now()
    msg.goal_id.id = ""

    msg.goal.width = width
    msg.goal.speed = speed

    time.sleep(2) # to wait for the connections of topics to publishers and subscribers
    gripper_publisher.publish(msg)

def control_orientation(z_rot_angle_left, right_robot = True):
    '''
    xyz: extrinsic; XYZ: intrinsic
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    '''
    if right_robot:
        return R.from_euler("xyz", [0, 180, -90], degrees = True).as_quat()
    else:
        return R.from_euler("xyz", [180, 0, z_rot_angle_left], degrees = True).as_quat() # [180, 0, 0~90]


def pose_to_list(pose_msg, return_time_stamp=False):
    '''transform the pose to list

    Parameters
    ----------
    pose_msg : PoseStamped()
    return_time_stamp : bool, optional
        whether to return the time stamp, by default False

    Returns
    -------
    list
        [3D Cartesian coordinates, quaternions], length 7
        [3D Cartesian coordinates, quaternions, time stamp], length 8
    '''
    position = pose_msg.pose.position
    orientation = pose_msg.pose.orientation
    time_stamp = pose_msg.header.stamp.to_sec()
    pose_list = [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w]
    if return_time_stamp: 
        pose_list.append(time_stamp)
    return pose_list

def update_equilibrium_pose(pose_publisher, right_robot = True):
    '''update equilibrium pose using the current pose after manually moving robots.
    Use this before the impedance changes. Otherwise the robot will suddenly move to 
    the previous equilibrium pose.

    Parameters
    ----------
    pose_publisher : rospy.Publisher()
    
    right_robot : bool, optional
        for which robot, left or right, by default True
    '''
    if right_robot:
        current_msg = rospy.wait_for_message("/cartesian_pose_right", PoseStamped)
    else:
        current_msg = rospy.wait_for_message("/cartesian_pose_left", PoseStamped)
    current_msg.header.stamp = rospy.Time.now()
    # while not rospy.is_shutdown():
    #     print(f"The number of subscribers to this topic is: {pose_publisher.get_num_connections()}")
    #     time.sleep(1)
    #     # The number of subscribers to this topic is: 2
    #     # No need to use time.sleep() here
    time.sleep(2) # not necessary, just for safety
    pose_publisher.publish(current_msg)



def record_key_point(right_robot = True):
    '''collect key points of robot movement

    Parameters
    ----------
    translation_stiffness : list, length 3
        [x, y, z] to define the translation stiffness along three axes
    rotation_stiffness : list, length 3
        [x, y, z] to define the rotation stiffness around three axes
    right_robot : bool, optional
        for which robot, right (True) or left (False), by default True

    Returns
    -------
    array_like, 3*N
        N 3D Cartesian coordinates
    '''
    # get the key points
    num_key_point = input("Enter the number of key points: ")
    key_points = []
    for i in range(int(num_key_point)):
        input(f"Ready to record the {i+1}-th position. Press <ENTER> to continue.")
        key_points.append(get_current_pose(right_robot = right_robot))
    return np.array(key_points)[:, :3].T

def set_impedance(translation_stiffness, rotation_stiffness, right_robot = True):
    '''set the translation and rotation stiffness for the robot

    Parameters
    ----------
    translation_stiffness : list, length 3
        [x, y, z] to define the translation stiffness along three axes
    rotation_stiffness : list, length 3
        [x, y, z] to define the rotation stiffness around three axes
    right_robot : bool, optional
        for which robot, right (True) or left (False), by default True
    '''
    if right_robot:
        robot_object = "right"
    else:
        robot_object = "left"
    try:
        client = dynamic_reconfigure.client.Client("/dynamic_reconfigure_compliance_param_node")
        params = {robot_object+"_translational_stiffness_X": str(translation_stiffness[0]),
                  robot_object+"_translational_stiffness_Y": str(translation_stiffness[1]),
                  robot_object+"_translational_stiffness_Z": str(translation_stiffness[2]),
                  robot_object+"_rotational_stiffness_X": str(rotation_stiffness[0]),
                  robot_object+"_rotational_stiffness_Y": str(rotation_stiffness[1]),
                  robot_object+"_rotational_stiffness_Z": str(rotation_stiffness[2])}
        client.update_configuration(params)
    except rospy.ROSInterruptException:
        pass

def get_current_pose(right_robot = True):
    if right_robot:
        topic_sub = "/cartesian_pose_right"
    else:
        topic_sub = "/cartesian_pose_left"
    current_msg = rospy.wait_for_message(topic_sub, PoseStamped)
    return pose_to_list(current_msg)


def generate_target_path(key_points, safe_distance = 0.01, return_point_index = False):
    '''generate linearly interpolated target movement paths

    Parameters
    ----------
    key_points : numpy.ndarray
        (3, N) or (2, N) target key points
    safe_distance : float, optional
        safe end-effector moving distance, by default 0.015
    return_point_index : bool, optional
        whether to return key point index, by default False

    Returns
    -------
    numpy.ndarray, (list)
        (3, N) or (2, N) linearly interpolated target movement path, (target key movement point index)
    '''
    target_path = key_points[:, :1]
    point_distance = np.sqrt(np.sum((key_points[:, 1:]-key_points[:, :-1])**2, axis=0))
    key_point_index = [0]
    for i in range(key_points.shape[1]-1):
        num_step = int(np.ceil(point_distance[i]/safe_distance))+1
        step_position = np.linspace(key_points[:, i], key_points[:, i+1], num_step, axis=1)
        target_path = np.append(target_path, step_position[:, 1:], axis=1)
        key_point_index.append(key_point_index[i]+num_step-1)
    if return_point_index:
        return target_path, key_point_index
    else:
        return target_path


def move_to_pose(pose_publisher, desired_pose, pub_rate,
                 right_robot=True, safe_distance=0.01, safe_quat_diff=0.01):
    '''move to the desired pose, both position and orientation
    Parameters
    ----------
    pose_publisher : rospy.Publisher()
        choose the left or right robot publisher
    desired_pose : list, length 6
        [3D Cartesian coordinates, Fixed angles]
    pub_rate : rospy.Rate()
        control the frequency of publishing messages
    right_robot : bool, optional
        for which robot, by default True
    safe_distance : float, optional
        safe end-effector moving distance, by default 0.03
    safe_quat_diff : int, optional
        safe quaternion change amount, by default 0.01
    '''
    current_pose = np.array(get_current_pose(right_robot=right_robot))
    desired_pose = np.array(desired_pose)
    # print(current_pose)
    # print(desired_pose)

    # calculate the number of steps
    position_diff = np.sqrt(np.sum((desired_pose[:3]-current_pose[:3])**2))
    num_step_by_position = np.ceil(position_diff/safe_distance).astype(int)

    current_orientation_quat = current_pose[3:]
    desired_orientation_quat = R.from_euler("xyz", desired_pose[3:], degrees=True).as_quat()
    # quat and negative quat have the same euler pose
    quat_diff = np.abs(desired_orientation_quat-current_orientation_quat)
    num_step = np.max((np.ceil(quat_diff/safe_quat_diff)).astype(int))
    quat_diff_neg = np.abs(-desired_orientation_quat-current_orientation_quat)
    num_step_neg = np.max((np.ceil(quat_diff_neg/safe_quat_diff)).astype(int))
    num_step_by_orientation = min(num_step, num_step_neg)

    num_step = max(num_step_by_position, num_step_by_orientation)

    num_step += 1

    # for position, linear interpolation
    step_position = np.linspace(current_pose[:3], desired_pose[:3], num_step, axis=1)
    # for orientation, 
    # linear interpolation (Lerp) ******************
    # step_orientation_quat = np.linspace(current_orientation_quat, desired_orientation_quat, num_step, axis=1)
    # spherical linear interpolation (Slerp) *********************
    key_quats = np.concatenate((current_orientation_quat.reshape((1, -1)), desired_orientation_quat.reshape((1, -1))))
    key_rots = R.from_quat(key_quats)
    rot_slerp = Slerp(range(key_quats.shape[0]), key_rots)
    step_orientation_rot = rot_slerp(np.linspace(0, key_quats.shape[0]-1, num_step))
    step_orientation_quat = step_orientation_rot.as_quat().T
    # constant *********************
    # step_orientation_quat = np.tile(current_pose[3:].reshape((-1, 1)), num_step)

    step_pose = np.concatenate((step_position, step_orientation_quat), axis=0)

    # visualize the interpolated orientations
    # print(f"num_step_by_position is: {num_step_by_position}.")
    # print(f"num_step_by_orientation is: {num_step_by_orientation}.")
    # visualize_changing_pose(step_pose)

    # initialize the message  
    rate = rospy.Rate(pub_rate)
    goal = PoseStamped()
    goal.header.frame_id = ""

    # publish the continuously changing targets
    for i in range(num_step):
        goal.header.seq = i+1
        goal.header.stamp = rospy.Time.now()
        goal.pose.position = Point(step_pose[0, i], step_pose[1, i], step_pose[2, i])
        goal.pose.orientation = Quaternion(step_pose[3, i], step_pose[4, i], step_pose[5, i], step_pose[6, i])
        pose_publisher.publish(goal)
        rate.sleep()

def visualize_changing_pose(step_pose):
    fig = plt.figure(figsize=[30, 15])
    ax_position = fig.add_subplot(3, 1, 1)
    ax_quat = fig.add_subplot(3, 1, 2)
    ax_fixed = fig.add_subplot(3, 1, 3)

    ax_position.plot(step_pose[0, :], "r", label="x")
    ax_position.plot(step_pose[1, :], "g", label="y")
    ax_position.plot(step_pose[2, :], "b", label="z")
    ax_position.legend()
    ax_position.set_title("position")

    ax_quat.plot(step_pose[3, :], "r", label="x")
    ax_quat.plot(step_pose[4, :], "g", label="y")
    ax_quat.plot(step_pose[5, :], "b", label="z")
    ax_quat.plot(step_pose[6, :], "c", label="w")
    ax_quat.legend()
    ax_quat.set_title("orientation in quaternions")

    step_orientation_fixed = R.from_quat(step_pose[3:, :].T).as_euler("xyz", degrees=True).T
    ax_fixed.plot(step_orientation_fixed[0, :], "r", label="x")
    ax_fixed.plot(step_orientation_fixed[1, :], "g", label="y")
    ax_fixed.plot(step_orientation_fixed[2, :], "b", label="z")
    ax_fixed.legend()
    ax_fixed.set_title("orientation in fixed angles")

    plt.show()

def move_along_path(target_path, fixed_orien, publisher, pub_rate):
    # initialize the message
    rate = rospy.Rate(pub_rate)
    goal = PoseStamped()
    goal.header.frame_id = ""
    goal.pose.orientation = Quaternion(fixed_orien[0], fixed_orien[1], fixed_orien[2], fixed_orien[3])

    # publish the continuously changing targets
    for i in range(target_path.shape[1]):
        goal.header.seq = i+1
        goal.header.stamp = rospy.Time.now()
        goal.pose.position = Point(target_path[0, i], target_path[1, i], target_path[2, i])
        publisher.publish(goal)
        rate.sleep()
    print("The target completes moving.")

def move_along_pose(target_pose, publisher, pub_rate):
    # initialize the message
    rate = rospy.Rate(pub_rate)
    goal = PoseStamped()
    goal.header.frame_id = ""

    # publish the continuously changing targets
    for i in range(target_pose.shape[1]):
        goal.header.seq = i+1
        goal.header.stamp = rospy.Time.now()
        goal.pose.position = Point(target_pose[0, i], target_pose[1, i], target_pose[2, i])
        goal.pose.orientation = Quaternion(target_pose[3, i], target_pose[4, i], target_pose[5, i], target_pose[6, i])
        publisher.publish(goal)
        rate.sleep()
    print("The target completes moving.")

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
