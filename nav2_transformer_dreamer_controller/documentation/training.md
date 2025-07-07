# Training Overview
- Initially train model in gazebo sim using custom gym environment designed with gym vehicle model
- Switch to Nav2 environment to test/train further in Nav2 environment as necessary



# Training:
- Using an RL teacher (curriculum learning) that has the optimal path from A* to improve long horizon planning and short-term control
- Curriculum levels:
    - 3: no help
    - 2: teacher provides next best step from A* path
    - 1: teacher provides complete A* path
- Adjusting the curriculum level:
    - Measure the average success over the last N episodes. If less than a certain score, decrease the curriculum level