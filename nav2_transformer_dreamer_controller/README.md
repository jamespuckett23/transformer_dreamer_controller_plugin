# Goal
- Implement Transformer Dreamer as a controller/planner combo


# TODO
- TransDreamer was trained on recorded transitions (off-policy training). We want to train in a simulation environment, which means we want to train on-policy
    - On-policy training means step() -> reward, next_state, done, info
    - Directly interacts with Nav2 simulation environment
- Nav2 Gym environment
    - Implement as a controller that takes in a path, but actually ignores and plans/controls the vehicle itself




# Repo Requirements
- TransDreamer github -> inside models
- gym
- gazebo
- ROS2 Humble


# Usage
- Example
- Helpful Nav2 links


# Contact
- James Puckett: jcpuckett2001 at gmail dot com

Further information in the documentation folder