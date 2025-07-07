cmake_minimum_required(VERSION 3.5)
project(nav2_transformer_dreamer_controller)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

nav2_package()

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_FMA)
  add_compile_options(-mfma)
endif()

# handle c++ to python bindings using pybind11
# add_subdirectory(pybind11)
# pybind11_add_module(transformer_dramer_controller trans_dreamer_binding.cpp)

# install python scripts
install(PROGRAMS
  scripts/train.py
  DESTINATION lib/${PROJECT_NAME}
)

# add_library(transformer_dramer_controller SHARED
#   src/trans_dreamer_binding.cpp
#   src/trans_dreamer_planner.cpp
# )
target_compile_options(transformer_dramer_controller PUBLIC -O3)
target_include_directories(transformer_dramer_controller
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
target_link_libraries(transformer_dramer_controller PUBLIC
  ${geometry_msgs_TARGETS}
  nav2_core::nav2_core
  nav2_costmap_2d::layers
  nav2_costmap_2d::nav2_costmap_2d_core
  ${nav_msgs_TARGETS}
  pluginlib::pluginlib
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${std_msgs_TARGETS}
  tf2::tf2
  tf2_geometry_msgs::tf2_geometry_msgs
  tf2_ros::tf2_ros
  ${visualization_msgs_TARGETS}
)

install(TARGETS transformer_dramer_controller
  EXPORT nav2_transformer_dramer_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_find_gtest()

  add_subdirectory(test)
  add_subdirectory(benchmark)
endif()

ament_export_libraries(${libraries})
ament_export_dependencies(
  geometry_msgs
  nav2_core
  nav2_costmap_2d
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  std_msgs
  visualization_msgs
  Eigen3
)
ament_export_include_directories(include/${PROJECT_NAME})
ament_export_targets(nav2_transformer_dramer_controller)

pluginlib_export_plugin_description_file(nav2_core mppic.xml)
pluginlib_export_plugin_description_file(nav2_transformer_dramer_controller critics.xml)

ament_package()
