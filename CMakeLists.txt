#  minimum required cmake version: 3.17
cmake_minimum_required(VERSION 3.17)

project(LaneDetection)

set( PROJECT_SOURCE_DIR "D:/CODE/PRIVATE/20201111_lane_detection_semantic_segmentation")

# Compile OpenCV library
find_package( OpenCV REQUIRED HINTS "D:/Libraries/opencv/build" )
include_directories(${OpenCV_INCLUDE_DIR})
include_directories( "D:/Libraries/opencv/build/x64/vc15/bin" )
link_directories( "D:/Libraries/opencv/build/x64/vc15/bin" )
set( ALL_LIBS ${ALL_LIBS} ${OpenCV_LIBS} )
set( DEPENDENCIES ${OpenCV_LIBS} )


# Compile Eigen library
set( EIGEN3_INCLUDE_DIR "D:/Libraries/eigen-3.3.7" )
include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( "D:/Libraries/openCV/Build/install/x64/vc16/bin" )



# Compile RealSense SDK
set( REALSENSE_SDK "C:/Program Files (x86)/Intel RealSense SDK 2.0" )     # Same directory in both AlienWare and Acer

include_directories( "${REALSENSE_SDK}/include" )
include_directories( "${REALSENSE_SDK}/samples" )
include_directories( "${REALSENSE_SDK}/include/librealsense2" )
include_directories( "${REALSENSE_SDK}/third-party" )
include_directories( "${REALSENSE_SDK}/bin/x64" )

link_directories( "${REALSENSE_SDK}/lib/x64")
link_directories( "${REALSENSE_SDK}/samples/x64/Debug" )
link_directories( "${REALSENSE_SDK}/bin/x64" )

# Combine all libs, headers & source codes files
include_directories(${PROJECT_SOURCE_DIR}/src)
set( ALL_LIBS ${ALL_LIBS} realsense2 )
FILE( GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp )
FILE( GLOB HEADERS ${PROJECT_SOURCE_DIR}/src/*.hpp )


# Build executable files
add_executable( ${PROJECT_NAME} main.cpp ${SOURCES} ${HEADERS} )

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

target_link_libraries(${PROJECT_NAME} ${ALL_LIBS} )
