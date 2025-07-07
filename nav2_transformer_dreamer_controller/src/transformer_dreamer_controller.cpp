#include "../include/nav2_dt_planner/transformer_dreamer_planner.h"

void TransDreamerPlanner::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros, 
    std::string path_to_model = "") {


    if (path_to_model.empty()) {
        // you must include a path to the pre-trained model
        throw;
    }

    // load model from path
    try {
        agent = torch::jit::load("transformer_dreamer.pt");
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    max_speed = 5.0

    /* example cmd for agent
    // Create input tensor
    torch::Tensor input = torch::ones({1, 4});

    // Execute the model and print the output
    at::Tensor output = agent.forward({input}).toTensor();
    std::cout << output << std::endl;
    */
}

TransDreamerPlanner::activate() {
    
}

TransDreamerPlanner::deactivate() {
    
}

TransDreamerPlanner::cleanup() {
    
}

void TransDreamerPlanner::setPlan(const nav_msgs::msg::Path & path) {
    // load start and goal for transdreamer
    
}

geometry_msgs::msg::TwistStamped TransDreamerPlanner::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * /*goal_checker*/) {
    
    if (!agent) {
        RCLCPP_INFO(this->get_logger(), "Need to load trained RL model");
    }
    // take an action, given: current state (current pose/velocity, start, goal, and map)
    // return action based off transdreamer policy as a twist msg

    // auto action = agent.select_action(state);
    // geometry_msgs::msg::TwistStamped cmd_vel;
    // cmd_vel.header.frame_id = pose.header.frame_id;
    // cmd_vel.header.stamp = clock_->now();
    // cmd_vel.twist.linear.x = min(linear_vel, max_speed);
    // cmd_vel.twist.angular.z = max(
    //   -1.0 * abs(max_angular_vel_), min(
    //     angular_vel, abs(
    //       max_angular_vel_)));
  
    return cmd_vel;
}

void TransDreamerPlanner::state_update() {
    // used in training to return the current state to python gym environment

    agent.update_state();
}

void TransDreamerPlanner::setSpeedLimit(double max_speed) {
    // manually set the speed limit
    max_speed = max_speed;
}