import gym
from gym import spaces
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, Path
from nav2_msgs.msg import Costmap

from vehicle_model import EkfVehicleModel

import cv2 as cv

# This gym environment is directly set up to interface with the Nav2 autonomy stack Transformer-Dreamer Planner

# TODO look into using a learned or dynamically modeled vehicle model in control


class Nav2Gz(gym.Env):

    def __init__(self, transdreamer_planner, cost_map: Costmap, initial_state = None, mse_threshold=0.1, delta_theta_max=np.pi/16, theta_max=np.pi/4, control_rate_hz=10):
        super(Nav2Gz, self).__init__()
        # need access to gazebo sim
        # need access to nav2 autonomy stack
        # need access to transformer dreamer planner

        # design gym environment to be flexible with any state space and action space for future 

        self.control_rate_hz = control_rate_hz
        self.dt = 1.0/self.control_rate_hz
        self.planner = transdreamer_planner
        self.vehicle_model = EkfVehicleModel(initial_state)
        # TODO: possibly get steering change and max steering_angle from vehicle model class

        self.threshold = mse_threshold
        self.delta_theta_max = delta_theta_max
        self.theta_max = theta_max

        # robot state = [pos_x, pos_y, heading_theta, velocity, speed_limit, target_pos_x, target_pos_y]
        # - heading_theta: defined from x-axis
        # - velocity: velocity straight ahead (no backing up)
        if (initial_state):
            self.state = initial_state
        else:
            self.state = np.array(cost_map.metadata.size_x/2, cost_map.metadata.size_y/2, 0.0, 0.0, 5.0, cost_map.metadata.size_x/2, cost_map.metadata.size_y/2)

        # action space for bicycle model
        # - [x, :] -> forward velocity
        # - [:, x] -> steering angle
        self.action_space = spaces.Box(
            low=np.array([0.0, -theta_max]),
            high=np.array([1.0, theta_max]),
            dtype=np.float32
        )

        # observation space
        # - robot state
        # - world model (cost map)
        # - image sensor (if 3D)
        # TODO: check space of image and cost map
        self.observation_space = spaces.Dict({
            "state_vector": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            "cost_map": spaces.Box(low=0, high=255, shape=(100, 100), dtype=np.uint8)
            # "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })

        self.previous_predicted_path = Path
        self.predicted_path = Path

        # set up communication with ROS2 node
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.latest_odom = None


    @property
    def observation_space(self):
        return self.observation_space
        # shape = (1 if self._grayscale else 3,) + self._size
        # space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        # return gym.spaces.Dict({'image': space})
        # # return gym.spaces.Dict({
        # #     'image': self._env.observation_space,
        # # })

    @property
    def action_space(self):
        return self.action_space
    
    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def step(self, action, predicted_path):
        self.predicted_path = predicted_path

        # scale velocity (scale * speed limit)
        scaled_vel_cmd = action[0] * self.state[4]

        heading_diff = min(abs(action[1] - self.state[2]),2*self.theta_max) * (action[1] - self.state[2]) / abs(action[1] - self.state[2])
        angular_velocity = heading_diff/self.dt
 
        # set up twist msg
        msg = Twist()
        msg.linear.x = scaled_vel_cmd
        msg.angular.z = angular_velocity

        # take the action
        next_state = self.take_action(msg)

        # update observation
        obs = self.update_obs()

        reward = self.compute_reward(self.state, msg, obs)

        # update state information
        self.state = next_state
        self.previous_predicted_path = self.predicted_path

        done = self.is_done(obs)

        return obs, reward, done, {}

    def reset(self):
        # start a new gazebo environment
            # needs multiple gazebo environment
            # needs random generations
        # find new start and end positions
            # needs random start and goal positions
        return
    
    def set(self, initial_state):
        self.reset()
        state = initial_state

    
    def render(self, mode):
        return self._env.render(mode)
    
    def take_action(self, msg: Twist):
        next_state = self.vehicle_model.predict(self.state, msg, self.dt)
        return next_state
    
    def update_obs(self):
        self.state
        obs = False
        return obs
    
    def compute_reward(self, previous_state, cmd_msg: Twist, new_obs):
        '''
        Reward structure:
        - Prediction reward
        - Control command reward
        - Energy penalty
        - Cost map penalty
        '''
        prediction_parameter = 1
        control_cmd_parameter = 2
        energy_parameter = 0.3
        cost_map_parameter = 1

        cost_map_index = self.state_to_costmap_index(new_obs["state"], new_obs["cost_map"])

        reward = prediction_parameter*self.path_mse_error() + \
                 control_cmd_parameter*self.cmd_error(new_obs["state"], previous_state, self.previous_predicted_path.poses[1]) + \
                 energy_parameter*self.l2_norm(0.4*cmd_msg.linear.x, 0.6*cmd_msg.angular.z) + \
                 cost_map_parameter*new_obs["cost_map"].data[cost_map_index]
        
        max_recommended_angular_velocity = self.delta_theta_max/self.dt
        if (abs(cmd_msg.angular.z) > max_recommended_angular_velocity):
            reward -= (abs(cmd_msg.angular.z) - max_recommended_angular_velocity)

        return reward

    def is_done(self):
        if (self.state[0] == self.state[5] and self.state[1] == self.state[6]) or (self.is_obstacle(self.state[0], self.state[1])):
            return True
    
    def is_obstacle(self, pos_x, pos_y):
        return self.observation_space["cost_map"][pos_x][pos_y] == 0

    def l2_norm(self, x, y):
        return np.sqrt(x**2 + y**2)
    
    def state_to_costmap_index(self, state, cost_map):
        cost_map_x = state[0] * cost_map.info.resolution
        cost_map_y = state[1] * cost_map.info.resolution
        cost_map_index = cost_map_y * cost_map.info.width + cost_map_x
        return cost_map_index

    def path_mse_error(self):
        pass

    def cmd_error(self, current_state, previous_state, previous_predicted_next_state):
        '''
        cmd was taken to move from previous_state to state
        check similarity between predicted_next_state to state
            - this will measure how well cmd moved predicted_next_state to state

        -----------
        parameters:
        state = new_obs["state_vector"]
        previous_state -> input
        predicted_next_state = previous_action.predicted_path[1].to_state()

        ----------
        time step update
        i.e.:
        t-1:
        - previous_state
        - previous_predicted_next_state

        t:
        - current_state

        ----------
        Returns:
        MSE error of the difference between the two states
        '''

        return np.mean((current_state - previous_predicted_next_state)**2)