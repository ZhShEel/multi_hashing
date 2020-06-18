cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
PROJECT(uniform_try)

#----------
# C++ version setting
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -m64 -Wall -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -m64 -O3 -Wall")

#----------
# Project variable configurations
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#----------
# Find packages
FIND_PACKAGE(PkgConfig)
LIST(APPEND
        CMAKE_MODULE_PATH
        ${PROJECT_SOURCE_DIR}/cmake_modules)

# PCL dependent
find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# Graphics dependent
FIND_PACKAGE(GLFW3  REQUIRED)
FIND_PACKAGE(GLEW   REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
# CV utilities
FIND_PACKAGE(OpenCV 3.4.0 REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
# Parallel computation
FIND_PACKAGE(CUDA REQUIRED)

# Log utilities
FIND_PACKAGE(Glog REQUIRED)

find_package(Pangolin REQUIRED)

###PCL+CUDA
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()
foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

###

#----------
## Found Headers
GET_PROPERTY(included_dirs
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTY INCLUDE_DIRECTORIES)
MESSAGE(STATUS "Found include paths:")
foreach(included_dir ${included_dirs})
    MESSAGE(STATUS ${included_dir})
endforeach()

#----------
## Found Libs
SET(OPENGL_DEPENDENCIES "")
LIST(APPEND
        OPENGL_DEPENDENCIES
        ${OPENGL_LIBRARY}
        ${GLEW_LIBRARY})
if (APPLE)
    LIST(APPEND OPENGL_DEPENDENCIES
            ${GLFW3_LIBRARIES})
else ()
    LIST(APPEND OPENGL_DEPENDENCIES
            ${GLFW3_STATIC_LIBRARIES})
endif()

SET(CUDA_DEPENDENCIES "")
LIST(APPEND
        CUDA_DEPENDENCIES
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUDA_LIBRARY})

MESSAGE(STATUS "Found libraries:")
foreach(library ${OPENGL_DEPENDENCIES})
    MESSAGE(STATUS ${library})
endforeach()
foreach(library ${CUDA_DEPENDENCIES})
    MESSAGE(STATUS ${library})
endforeach()
foreach(library ${OTHER_DEPENDENCIES})
    MESSAGE(STATUS ${library})
endforeach()
#---------

#----------
SET(VH ${PROJECT_SOURCE_DIR}/src)
SET(GL_WRAPPER ${VH}/extern/opengl-wrapper)

INCLUDE_DIRECTORIES(
        ${VH}
        ${VH}/extern/cuda
        ${GL_WRAPPER}/include
        ${EIGEN3_INCLUDE_DIR}
        ${GLOG_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_INCLUDE}
        ${Pangolin_INCLUDE_DIRS})

#----------
## Building Libraries
### 1. OpenGL util
ADD_LIBRARY(gl-util SHARED
        ${GL_WRAPPER}/src/core/args.cc
        ${GL_WRAPPER}/src/core/camera.cc
        ${GL_WRAPPER}/src/core/framebuffer.cc
        ${GL_WRAPPER}/src/core/model.cc
        ${GL_WRAPPER}/src/core/program.cc
        ${GL_WRAPPER}/src/core/texture.cc
        ${GL_WRAPPER}/src/core/uniforms.cc
        ${GL_WRAPPER}/src/core/window.cc)
SET_TARGET_PROPERTIES(gl-util
        PROPERTIES
        COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(gl-util
        ${OpenCV_LIBRARIES}
        ${OPENGL_DEPENDENCIES}
        ${GLOG_LIBRARIES}
        ${CUDA_DEPENDENCIES}
        ${PCL_LIBRARIES}
        )

### 2. CUDA
# Don't know exactly how it should be configured
LIST(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' --shared)
CUDA_ADD_LIBRARY(mesh-hashing-cuda SHARED
        ${VH}/core/PointCloud.cu
        ${VH}/core/hash_table.cu
        ${VH}/core/entry_array.cu
        ${VH}/core/block_array.cu
        ${VH}/core/mesh.cu
        ${VH}/core/collect_block_array.cu
        ${VH}/localizing/point_to_psdf.cu
        ${VH}/localizing/icp.cu
        ${VH}/mapping/allocate.cu
        ${VH}/mapping/update_simple.cu
        ${VH}/mapping/update_psdf.cu
        # ${VH}/mapping/update_bayesian.cu
        ${VH}/mapping/recycle.cu

        ${VH}/optimize/linear_equations.cu
        ${VH}/optimize/primal_dual.cu

        ${VH}/meshing/marching_cubes.cu
        ${VH}/meshing/remeshing.cu

        ${VH}/visualization/colorize.cu
        ${VH}/visualization/compact_mesh.cu
        ${VH}/visualization/bounding_box.cu
        ${VH}/visualization/trajectory.cu
        ${VH}/visualization/compress_mesh.cu
        ${VH}/visualization/extract_bounding_box.cu
        ${VH}/visualization/ray_caster.cu
        ${VH}/sensor/rgbd_sensor.cu
        ${VH}/sensor/preprocess.cu
        ${VH}/geometry/knncuda.cu
        )

TARGET_LINK_LIBRARIES(mesh-hashing-cuda
        ${CUDA_DEPENDENCIES}
        ${GLOG_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PCL_LIBRARIES})

### 3. C++
ADD_LIBRARY(mesh-hashing SHARED
        ${VH}/engine/main_engine.cc
        ${VH}/engine/mapping_engine.cc
        ${VH}/engine/logging_engine.cc
        ${VH}/engine/visualizing_engine.cc

        ${VH}/io/config_manager.cc
        ${VH}/io/mesh_writer.cc
        ${VH}/sensor/point_cloud_provider.cc

        ${VH}/sensor/rgbd_data_provider.cc)
#${VH}/tool/cpp/debugger.cc)
SET_TARGET_PROPERTIES(mesh-hashing
        PROPERTIES
        COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(mesh-hashing
        mesh-hashing-cuda
        gl-util)


ADD_LIBRARY(psdf  SHARED
        src/app/psdf.cc src/app/psdf.h)
SET_TARGET_PROPERTIES(psdf
        PROPERTIES
        COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(psdf
        mesh-hashing
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${PCL_LIBRARIES})


#----------
### Loop over
MESSAGE(STATUS ${GLOG_LIBRARIES})
ADD_EXECUTABLE(reconstruction src/app/reconstruction.cc src/app/psdf.cc src/app/psdf.h)
SET_TARGET_PROPERTIES(reconstruction
        PROPERTIES
        COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(reconstruction
        mesh-hashing
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Pangolin_LIBRARIES})

#ADD_EXECUTABLE(slam src/app/slam.cc)
#SET_TARGET_PROPERTIES(slam
#        PROPERTIES
#        COMPILE_DEFINITIONS USE_CUDA_GL)
#TARGET_LINK_LIBRARIES(slam
#        mesh-hashing
#        ${OpenCV_LIBRARIES}
#        ${GLOG_LIBRARIES})

ADD_EXECUTABLE(block_analysis src/app/block_analysis.cc)
SET_TARGET_PROPERTIES(block_analysis
        PROPERTIES
        COMPILE_DEFINITIONS USE_CUDA_GL)
TARGET_LINK_LIBRARIES(block_analysis
        gl-util
        mesh-hashing
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${PCL_LIBRARIES})

#### An ORB app
#find_package(freenect2 REQUIRED)
#include_directories(${freenect2_INCLUDE_DIR})
#OPTION(WITH_ORBSLAM2 "Build with orb slam" ON)
#if (WITH_ORBSLAM2)
#    SET(ORB_SLAM2_PATH ${VH}/extern/orb_slam2)
#    MESSAGE(STATUS "Build with ORB-SLAM2")
#
#    FIND_PACKAGE(Pangolin REQUIRED)
#    FIND_PACKAGE(Eigen3 REQUIRED)
#    INCLUDE_DIRECTORIES(
#            ${ORB_SLAM2_PATH}
#            ${ORB_SLAM2_PATH}/include
#            ${EIGEN3_INCLUDE_DIR})
#    LINK_DIRECTORIES(${ORB_SLAM2_PATH}/lib)
#
#    ADD_EXECUTABLE(orb_slam2 src/app/orb_slam2.cc)
#    TARGET_LINK_LIBRARIES(orb_slam2
#            mesh-hashing
#            ORB_SLAM2
#            ${OpenCV_LIBRARIES}
#            ${Pangolin_LIBRARIES}
#            ${freenect2_LIBRARY}
#            ${GLOG_LIBRARIES})
#
#    SET_TARGET_PROPERTIES(orb_slam2
#            PROPERTIES
#            COMPILE_DEFINITIONS USE_CUDA_GL)
#endif(WITH_ORBSLAM2)
