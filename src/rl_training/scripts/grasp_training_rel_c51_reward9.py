#!/usr/bin/env python3
import sys
import os

from tensorflow.python.ops.math_ops import truediv
from yaml.loader import Loader
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
import numpy as np
import math
import pickle
import random 
import tensorflow as tf
import time
import datetime

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

solve_cudnn_error()

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from rl_training.msg import AngleAxis_rotation_msg
from rl_training.srv import loadPointCloud
from rl_training.srv import rl_is_success

import time

import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

from std_msgs.msg import Int64, Float64
from cv_bridge import CvBridge, CvBridgeError

import abc

import tensorboardX
import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent

from tf_agents.networks import q_network
from tf_agents.policies import policy_saver

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils as env_utils
from tf_agents.environments import wrappers
from tf_agents.environments import random_py_environment

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

from tf_agents.networks import network
from tf_agents.networks import encoding_network
from tf_agents.networks import utils
from tf_agents.networks import categorical_q_network

from tf_agents.utils import common
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

from tf_agents.trajectories import time_step as ts
from rl_training.srv import rl_is_success


from grasp_Env_RelAction_reward9 import GraspEnv

tf.compat.v1.enable_v2_behavior()

xyz = np.array([[0,0,0]])
rgb = np.array([[0,0,0]])

def rotate_grasp():
    rotation = AngleAxis_rotation_msg()
    rotation_angle = math.pi/2
    interval = 50

    for i in range(interval):
        rotation.x = 0
        rotation.y = -1*rotation_angle/interval*i
        rotation.z = 0
        pub_AngleAxisRotation.publish(rotation)
        time.sleep(0.1)

def grab_pointClouds_callback(ros_point_cloud):
    global xyz, rgb
    #self.lock.acquire()
    gen = pc2.read_points(ros_point_cloud, skip_nans=True)
    int_data = list(gen)

    xyz = np.array([[0,0,0]])
    rgb = np.array([[0,0,0]])
    
    for x in int_data:
        test = x[3] 
        # cast float32 to int so that bitwise operations are possible
        s = struct.pack('>f' ,test)
        i = struct.unpack('>l',s)[0]
        # you can get back the float value by the inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000)>> 16
        g = (pack & 0x0000FF00)>> 8
        b = (pack & 0x000000FF)
        # prints r,g,b values in the 0-255 range
                    # x,y,z can be retrieved from the x[0],x[1],x[2]
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
        rgb = np.append(rgb,[[r,g,b]], axis = 0)

    # rospy.loginfo('Done grab_pointClouds_callback')

def do_loadPointCloud(req):
    rospy.wait_for_service('/load_pointcloud')
    try:
        resp = handle_loadPointCloud(req)
        return resp

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    time_start = time.time()
    num_env = 50
    total_success = 0
    for _ in range(num_env):
        do_loadPointCloud(1)
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                grasp_result = check_rl_is_success(1)
                episode_return += time_step.reward
            total_success += grasp_result.is_success
            total_return += episode_return

    print("avg execute time ", (time.time()-time_start)/num_episodes/num_env)

    avg_return = total_return / num_episodes / num_env
    success_rate = total_success / num_episodes / num_env

    return avg_return.numpy()[0], success_rate

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    # print("next_time_step.reward ", next_time_step.reward)
    traj = tf_agents.trajectories.trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)

def save_agent(save_path, agent_name, agent_policy):
    policy_dir = os.path.join(save_path, agent_name)
    tf_policy_saver = policy_saver.PolicySaver(agent_policy)
    tf_policy_saver.save(policy_dir)

def check_rl_is_success(req):
    rospy.wait_for_service('/rl_grasp_is_success')
    try:
        resp  = handle_rl_is_success(req)
        return resp
    except rospy.ServiceException as e:
        print("Service call rl_si_success failed: %s"%e)

num_iterations = 15000

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_capacity = 100000

fc_layer_params = (100,)

batch_size = 64
learning_rate = 1e-3
gamma = 0.99
log_interval = 200

num_atoms = 51
min_q_value = -20
max_q_value = 20
n_step_update = 1 

num_eval_episodes = 10
eval_interval = 1000

if __name__ == '__main__':

    # print("os.path.dirname(__file__)  ", os.path.dirname(__file__))

    file_path = os.path.dirname(__file__)

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')

    description = rospy.get_param('description')

    tb = tensorboardX.SummaryWriter(file_path + "/trained-model/C51/" + "C51_" + str(dt) + "_" + str(description) + "/")

    _step_lengtn = 5
    
    #init ros
    rospy.init_node('Reinforcement_Learning_Training', anonymous=True)

    handle_loadPointCloud = rospy.ServiceProxy('/load_pointcloud', loadPointCloud)
    handle_rl_is_success = rospy.ServiceProxy('/rl_grasp_is_success', rl_is_success)

    do_loadPointCloud(1)

    environment = GraspEnv([120, 160], "training", step_lengtn=_step_lengtn)

    time.sleep(1)

    env_utils.validate_py_environment(environment, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(environment)
    
    preprocessing_layers = {
    'depth_grab': tf.keras.models.Sequential([ 
        tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten()])
                                        }

    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)


    categorical_q_net = categorical_q_network.CategoricalQNetwork(
                    tf_env.observation_spec(), 
                    tf_env.action_spec(),
                    preprocessing_layers=preprocessing_layers,
                    num_atoms=num_atoms,
                    fc_layer_params=(100, 50),
                    activation_fn=tf.keras.activations.tanh,
                    name='C51_QNetwork')

    learning_rate = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    start_epsilon = 0.1
    n_of_steps = 50000
    end_epsilon = 0.0001
    epsilon = tf.compat.v1.train.polynomial_decay(
        start_epsilon,
        global_step,
        n_of_steps,
        end_learning_rate=end_epsilon)
    n_TD_step_update = 1

    agent = categorical_dqn_agent.CategoricalDqnAgent(
        tf_env.time_step_spec(), 
        tf_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        epsilon_greedy=epsilon,
        gamma=gamma,
        train_step_counter=global_step)

    agent.initialize()

    print("categorical_q_net.summary(): ", categorical_q_net.summary())

    replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                                                            batch_size=tf_env.batch_size,
                                                                                            max_length=64*100)

    
    avg_return, success_rate = compute_avg_return(tf_env, agent.policy, 1)
    print('step = {0}: Average Return = {1}'.format(0, avg_return))
    print('step = {0}: Success Rate = {1}'.format(0, success_rate))


    returns = [avg_return]

    collect_steps_per_iteration = 1
    batch_size = 64
    dataset = replay_buffer.as_dataset(num_parallel_calls=3, 
                                        sample_batch_size=batch_size, 
                                        num_steps=(n_TD_step_update+1)).prefetch(3)
    iterator = iter(dataset)
    num_iterations = 10000

    time_step = tf_env.reset()

    for _ in range(batch_size):
        collect_step(tf_env, agent.collect_policy, replay_buffer)

    while not rospy.is_shutdown():

        for _ in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                collect_step(tf_env, agent.collect_policy, replay_buffer)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % batch_size == 0:
                do_loadPointCloud(1)

            # Print loss every 200 steps.
            if step % 200 == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

                tb.add_scalar("/train/loss", train_loss.numpy(), step)

            # Evaluate agent's performance every 1000 steps.
            if step % 1000 == 0:
                avg_return, success_rate = compute_avg_return(tf_env, agent.policy, 1)

                save_agent(file_path + "/trained-model/C51/" + "C51_" + str(dt) + "_" + str(description) + \
                            "/Model/",'C51_policy_' + str(step/1000) + "_avg_return_" + str(avg_return) + "_success_rate_" + str(success_rate), agent.policy)

                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                print('step = {0}: Success Rate = {1}'.format(step, success_rate))

                tb.add_scalar("/train/reward", avg_return, step)
                tb.add_scalar("/train/success_rate", success_rate, step)
