#ifndef NAV2_TRANS_DREAMER_PLANNER__TRANSFORMER_DREAMER_PLANNER_HPP_
#define NAV2_TRANS_DREAMER_PLANNER__TRANSFORMER_DREAMER_PLANNER_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

#include "nav2_core/global_planner.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_util/lifecycle_node.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

// #include <pybind11/embed.h>
#include <torch/script.h>
// namespace py = pybind11;


class TransDreamerPlanner {
    public:
        TransDreamerPlanner();

        void configure(
            const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
            std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
            std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros);
        
        void activate();
        
        void deactivate():
        
        void cleanup();
        
        void setPlan(const nav_msgs::msg::Path & path);
        
        geometry_msgs::msg::TwistStamped computeVelocityCommands(
            const geometry_msgs::msg::PoseStamped & pose,
            const geometry_msgs::msg::Twist & velocity,
            nav2_core::GoalChecker * /*goal_checker*/);
        
        void state_update();
        
        void setSpeedLimit(double max_speed);

    private:
        torch::jit::script::Module agent; // used to load in the trained agent
        double max_speed;

}


#endif // NAV2_TRANS_DREAMER_PLANNER__TRANSFORMER_DREAMER_PLANNER_HPP_