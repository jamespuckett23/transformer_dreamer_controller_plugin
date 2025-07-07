# Goal
- Implement Transformer Dreamer as a controller/planner combo
- The transformer dreamer paper has two components: the specific action and the prediction of actions over a period of time
    - Dreamer trains the world model to accurately predict the transitions over a certain number of future steps. This algorithm uses A* to train the world model predictions
- This implementation looks to train both the prediction and the action phases.
    - Planning/prediciton component looks to immitate the A* algorithm
    - Action component looks to follow this predicted path smoothly and efficiently
- The Nav2 stack has not used any RL algorithms yet at the time of writing. This plugin will be a simple example setting up the training scripts and requirements to train in simulation for the Nav2 stack.



# To Do
- Transformer Dreamer - done
- Training environment
    - Gym custom vehicle model - done
    - Gazebo sim - not started
        - design world
        - set up ros2_gz bridge
        - launch files
    - Gym/Gazebo environment - started
        - Interface with gym - done
        - Interface with gazebo - not started
    - Training loop - started
        - Curriculum loop design - done
        - Interface with transformerdreamer - started
    - Cloud training script - not started
- Training algorithm
    - Design - done
    - Implementation - started
- Nav2 Controller plugin
    - Design - done
    - Implementation - started
- Finetuning in Nav2 stack - not started



# Repo Requirements
- TransDreamer github -> inside models
- gym
- gazebo
- ROS2 Humble
- Turtlebot3



# Usage
- Nav2: https://docs.nav2.org/
- Example:
    - Train model locally with train.py or in the cloud with train_google_colabs.ipynb
        - This trains the model with a custom vehicle model in a gazebo simulation
    - Once finished training, set up the nav2 autonomy stack and finetune for any differences (hz of control loop, vehicle model dynamics, etc)
        - start gz
        - start nav2 with transformer_dreamer_controller_plugin that has access to the transformer_dreamer_model (train tag set to true)
        - save model once performance improves enough
    - Run nav2 with transformer_dreamer_controller_plugin 


# Contact
- James Puckett: jcpuckett2001 at gmail dot com


Further information can be found in the documentation folder