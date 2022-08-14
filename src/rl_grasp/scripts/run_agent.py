#!/usr/bin/env python3
import os
import time
import rospy
import tensorflow as tf
from rl_grasp.srv import get_Agent_action
from tf_agents.environments import tf_py_environment
from grasp_Env_RelAction_reward12 import GraspEnv

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

tf.compat.v1.enable_v2_behavior()

if __name__ == '__main__':

    rospy.init_node('Reload_Reinforcement_Learning_Agent', anonymous=True)

    step_length = 10
    
    environment = GraspEnv([120, 160], "inference", step_length)

    time.sleep(1)

    tf_env = tf_py_environment.TFPyEnvironment(environment)

    file_path = os.path.dirname(__file__)

    # 20220513
    # ---good
    # policy_dir = os.path.join(file_path + "/trained-model" + "/C51/C51_220512_1750_service4_reward12_steplength4_100objects/Model/C51_policy_210.0_avg_return_3.6619248_success_rate_0.98")

    # 20220516
    # ---good
    policy_dir = os.path.join(file_path + "/trained-model" + "/C51/C51_220515_1506_service4_reward12_steplength10_100objects/Model/C51_policy_230.0_avg_return_4.3904934_success_rate_1.0")
    
    saved_policy = tf.saved_model.load(policy_dir)

    def handle_get_agent_action(req):
        time_step = tf_env.reset()
        total_reward = 0
        time1 = time.time()
        counter = 0
        while total_reward <= 0:
            for i in range(step_length):
                action_step = saved_policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                total_reward += time_step.reward.numpy()
                if time_step.is_last():
                    break
                print("time_step reward: ", time_step.reward.numpy())
            counter += 1
            if counter > 3:
                # time_step = tf_env.reset()
                break

        print("total_reward: ", total_reward)
        print("rl_grasp run time ", time.time() - time1)
        return total_reward

    s = rospy.Service('/get_agent_action', get_Agent_action, handle_get_agent_action)

    print("[agent ready]")

    rospy.spin()