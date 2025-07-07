
# State description
- current location
- goal location
- STVL: spatial temporal voxel layer (https://docs.nav2.org/tutorials/docs/navigation2_with_stvl.html)
- World Model (dynamic objects w/ predictions)


# Model
- Action space
    - Predictive capabilities:
        z = encoder(obs)
        latents = dreamer.rollout(z, horizon=10)
        predicted_positions = position_decoder(latents)  # MLP → (x, y)
        loss = MSE(predicted_positions, ground_truth_future_path)
    - Predicted path
    - Control actions (speed and direction of vehicle)
- Observation space
    - Current pose
    - Goal pose
    - Predicted path
    - STVL/cost map




# Loss Function
- This model is designed to predict the best plan and command control actions based on that predicted plan. This is down with curriculum learning found in the training.md document.
- Based on which curriculum level:
    - 0: A* path is provided, 
        - control_action_loss = accuracy_measured_by_true_path() + vehicle_hits_obstacle()
        - planning_loss = difference_between_predicted_path_and_true_path()
    - 1: next step in the A* path is provided
        - control_action_loss = accuracy_measured_by_true_path() + vehicle_hits_obstacle()
        - planning_loss = predicted_path_cost() - path_reaches_goal() {reward} + path_crosses_obstacle()
    - 2: no information provided
        - control_action_loss = accuracy_measured_against_predicted_path() + vehicle_hits_obstacle()
        - planning_loss = difference_between_predicted_path_and_true_path() - path_reaches_goal() {reward}