cmake_minimum_required(VERSION 3.8)
project(guardian_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MetricSample.msg"
  "msg/NodeHealth.msg"
  "msg/SystemAlert.msg"
  "msg/HealingAction.msg"
  "srv/TriggerHealing.srv"
  "srv/GetHealthReport.srv"
  "srv/GetHealthReport.srv"
  "action/ExecuteRecovery.action"
  DEPENDENCIES
    builtin_interfaces
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
