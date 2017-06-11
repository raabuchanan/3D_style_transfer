
cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
PROJECT(vision_project)

#set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )


find_package(PCL 1.7 REQUIRED)

find_package( Eigen3 REQUIRED )
include_directories( EIGEN3_INCLUDE_DIR )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library (pcl_help src/pcl_filters.cpp src/pcl_simple_viewer.cpp)

add_executable (filter_point_cloud src/filter_point_cloud.cpp)
target_link_libraries (filter_point_cloud ${PCL_LIBRARIES} pcl_help)

add_executable (view_point_cloud src/view_point_cloud.cpp)
target_link_libraries (view_point_cloud ${PCL_LIBRARIES} pcl_help )
set_target_properties(view_point_cloud PROPERTIES COMPILE_FLAGS "-std=c++11")

add_executable (cloud_painting src/cloud_painting.cpp src/data_txt_reader.cpp)
target_link_libraries (cloud_painting ${PCL_LIBRARIES} ${OpenCV_LIBS} pcl_help )
set_target_properties(cloud_painting PROPERTIES COMPILE_FLAGS "-std=c++11")