#!/usr/bin/env python3
import sys
import cv2
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
from rl_grasp.srv import get_Agent_action

from rl_grasp.srv import loadPointCloud
import time

import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct

from std_msgs.msg import Int64, Float64
from cv_bridge import CvBridge, CvBridgeError

import abc
import tf_agents
from tf_agents.agents.dqn import dqn_agent
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

from tf_agents.utils import common
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

from grasp_Env_RelAction_reward8 import GraspEnv


if __name__ == '__main__':

    rospy.init_node('Reload_Reinforcement_Learning_Agent', anonymous=True)

    step_length = 5
    
    environment = GraspEnv([120, 160], "inference", step_length)

    time.sleep(1)

    # env_utils.validate_py_environment(environment, episodes=5)

    tf_env = tf_py_environment.TFPyEnvironment(environment)

    file_path = os.path.dirname(__file__)

    # C51
    # policy_dir = os.path.join(file_path + "/trained-model" + "/C51/C51_220428_1751_step_length_1_reward_v2/Model/C51_policy_830.0_5.4259133")

    # 20220503
    # policy_dir = os.path.join(file_path + "/trained-model" + "/DQN/DQN_220502_1726_dqn_reward2/Model/DQN_policy_327.0_5.9956946")
    # policy_dir = os.path.join(file_path + "/trained-model" + "/DQN/DQN_220502_1032_service2_reward3_only_y_action/Model/DQN_policy_120.0_-3.2558465")

    # 20220504 new dataset
    # policy_dir = os.path.join(file_path + "/trained-model" + "/DQN/DQN_220503_1423_service2_reward2_newdataset/Model/DQN_policy_28.0_-1.3549198")

    # 20220505 new dataset
    # not bad!
    # policy_dir = os.path.join(file_path + "/trained-model" + "/DQN/DQN_220504_1054_service2_reward4_newdataset/Model/DQN_policy_80.0_5.261268")

    # 20220506 newdataset
    # reward 5
    # # ---bad 
    # policy_dir = os.path.join(file_path + "/trained-model" + "/DQN/DQN_220505_1144_dqn_service2_reward5_steplength5_onlynormaldepth/Model/DQN_policy_37.0_1.9077859")
    # # ---great !
    # policy_dir = os.path.join(file_path + "/trained-model" + "/C51/C51_220505_1145_c51_service2_reward5_steplength5_onlynormaldepth/Model/C51_policy_143.0_4.103627")

    # reward 6
    # # ---bad
    # policy_dir = os.path.join(file_path + "/trained-model" + "/DQN/DQN_220505_1229_dqn_service2_reward6_steplength5_onlynormaldepth_xyaction/Model/DQN_policy_124.0_-2.7130487")

    # ---not bad
    # policy_dir = os.path.join(file_path + "/trained-model" + "/C51/C51_220505_1232_c51_service2_reward6_steplength5_onlynormaldepth_xyaction/Model/C51_policy_77.0_0.99006736")

    # 20220507 newdataset
    # reward 7
    # --not bad
    # policy_dir = os.path.join(file_path + "/trained-model" + "/C51/C51_220506_1547_service2_reward7_steplength5_onlynormaldepth/Model/C51_policy_242.0_2.973694")
    # reward 8
    policy_dir = os.path.join(file_path + "/trained-model" + "/C51/C51_220506_1603_c51_service2_reward8_steplength5_input2ch/Model/C51_policy_188.0_2.8839202")


    saved_policy = tf.saved_model.load(policy_dir)


    def handle_get_agent_action(req):
        time_step = tf_env.reset()
        total_reward = 0
        time1 = time.time()
        for i in range(step_length):
            action_step = saved_policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            total_reward += time_step.reward.numpy()
            print("time_step reward: ", time_step.reward.numpy())
        print("total_reward: ", total_reward)
        print("rl_grasp run time ", time.time() - time1)
        return total_reward

    s = rospy.Service('/get_agent_action', get_Agent_action, handle_get_agent_action)

    print("[agent ready]")

    rospy.spin()

    # while 1:
    #   time_step = tf_env.reset()
    #   total_reward = 0
    # #   while not time_step.is_last():
    #   for i in range(2):
    #     action_step = saved_policy.action(time_step)
    #     time_step = tf_env.step(action_step.action)
    #     total_reward += time_step.reward.numpy()
    #     print("time_step reward: ", time_step.reward.numpy())
    #   print("total_reward: ", total_reward)
