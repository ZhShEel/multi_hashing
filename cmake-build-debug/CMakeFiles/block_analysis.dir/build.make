# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/zhangsheng/Downloads/clion-2017.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/zhangsheng/Downloads/clion-2017.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhangsheng/Research/recon/recon/psdf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhangsheng/Research/recon/recon/psdf/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/block_analysis.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/block_analysis.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/block_analysis.dir/flags.make

CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o: CMakeFiles/block_analysis.dir/flags.make
CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o: ../src/app/block_analysis.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhangsheng/Research/recon/recon/psdf/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o -c /home/zhangsheng/Research/recon/recon/psdf/src/app/block_analysis.cc

CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhangsheng/Research/recon/recon/psdf/src/app/block_analysis.cc > CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.i

CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhangsheng/Research/recon/recon/psdf/src/app/block_analysis.cc -o CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.s

CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.requires:

.PHONY : CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.requires

CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.provides: CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.requires
	$(MAKE) -f CMakeFiles/block_analysis.dir/build.make CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.provides.build
.PHONY : CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.provides

CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.provides.build: CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o


# Object files for target block_analysis
block_analysis_OBJECTS = \
"CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o"

# External object files for target block_analysis
block_analysis_EXTERNAL_OBJECTS =

../bin/block_analysis: CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o
../bin/block_analysis: CMakeFiles/block_analysis.dir/build.make
../bin/block_analysis: ../lib/libmesh-hashing.so
../bin/block_analysis: /usr/lib/x86_64-linux-gnu/libglog.so
../bin/block_analysis: ../lib/libgl-util.so
../bin/block_analysis: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/block_analysis: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/block_analysis: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/block_analysis: ../lib/libmesh-hashing-cuda.so
../bin/block_analysis: /usr/local/lib/libopencv_cudabgsegm.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudastereo.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_dnn.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_ml.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_shape.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_stitching.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_superres.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudacodec.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_videostab.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudaoptflow.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudalegacy.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_calib3d.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudawarping.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_features2d.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_flann.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_highgui.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_objdetect.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_photo.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudaimgproc.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudafilters.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudaarithm.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_video.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_videoio.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_imgproc.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_core.so.3.4.0
../bin/block_analysis: /usr/local/lib/libopencv_cudev.so.3.4.0
../bin/block_analysis: /usr/local/cuda/lib64/libcudart_static.a
../bin/block_analysis: /usr/lib/x86_64-linux-gnu/librt.so
../bin/block_analysis: /usr/lib/x86_64-linux-gnu/libglog.so
../bin/block_analysis: /usr/local/cuda/lib64/libcudart.so
../bin/block_analysis: /usr/lib/x86_64-linux-gnu/libcuda.so
../bin/block_analysis: CMakeFiles/block_analysis.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhangsheng/Research/recon/recon/psdf/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/block_analysis"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/block_analysis.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/block_analysis.dir/build: ../bin/block_analysis

.PHONY : CMakeFiles/block_analysis.dir/build

CMakeFiles/block_analysis.dir/requires: CMakeFiles/block_analysis.dir/src/app/block_analysis.cc.o.requires

.PHONY : CMakeFiles/block_analysis.dir/requires

CMakeFiles/block_analysis.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/block_analysis.dir/cmake_clean.cmake
.PHONY : CMakeFiles/block_analysis.dir/clean

CMakeFiles/block_analysis.dir/depend:
	cd /home/zhangsheng/Research/recon/recon/psdf/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhangsheng/Research/recon/recon/psdf /home/zhangsheng/Research/recon/recon/psdf /home/zhangsheng/Research/recon/recon/psdf/cmake-build-debug /home/zhangsheng/Research/recon/recon/psdf/cmake-build-debug /home/zhangsheng/Research/recon/recon/psdf/cmake-build-debug/CMakeFiles/block_analysis.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/block_analysis.dir/depend

