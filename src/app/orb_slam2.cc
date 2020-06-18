/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <opencv2/core/core.hpp>

#include <System.h>
#include <glog/logging.h>
#include <sensor/rgbd_data_provider.h>

#include "core/params.h"
#include "io/config_manager.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "visualization/ray_caster.h"
#include "glwrapper.h"

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

const unsigned int kDepthWidth  = 512;
const unsigned int kDepthHeight = 424;
const unsigned int kRgbWidth    = 1920;
const unsigned int kRgbHeight   = 1080;
const unsigned int kSaveWidth   = 640;
const unsigned int kSaveHeight  = 360;

int wait_time[2] = {-1, 10};
int wait_time_idx = 0;

const std::string orb_configs[] = {
    "../config/ORB/ICL.yaml",
    "../config/ORB/TUM1.yaml",
    "../config/ORB/TUM2.yaml",
    "../config/ORB/TUM3.yaml",
    "../config/ORB/SUN3D.yaml",
    "../config/ORB/SUN3D_ORIGINAL.yaml",
    "../config/ORB/PKU.yaml",
};

std::string path_to_vocabulary = "../src/extern/orb_slam2/Vocabulary/ORBvoc.bin";

float4x4 MatTofloat4x4(cv::Mat m) {
  float4x4 T;
  T.setIdentity();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      T.entries2[i][j] = (float) m.at<float>(i, j);
  return T;
}

int main(int argc, char **argv) {
  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev = 0;
  libfreenect2::PacketPipeline *pipeline = 0;

  // Enumerate USB devices
  if (freenect2.enumerateDevices() == 0) {
    LOG(ERROR) << "No device connected!";
    return -1;
  }

  // Open device at the enumerated valid serial
  std::string serial = freenect2.getDefaultDeviceSerialNumber();
  pipeline = new libfreenect2::OpenGLPacketPipeline();
  if (pipeline) {
    dev = freenect2.openDevice(serial, pipeline);
  }
  if (dev == 0) {
    LOG(ERROR) << "Failure opening device!";
    return -1;
  }

  // Sync data reader
  libfreenect2::SyncMultiFrameListener listener(
      libfreenect2::Frame::Color
      | libfreenect2::Frame::Ir
      | libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;
  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);

  // Start
  LOG(INFO) << "device serial: " << dev->getSerialNumber();
  LOG(INFO) << "device firmware: " << dev->getFirmwareVersion();
  dev->start();

  // Register Depth and Color images
  libfreenect2::Registration* registration
      = new libfreenect2::Registration(dev->getIrCameraParams(),
                                       dev->getColorCameraParams());
  libfreenect2::Frame
      undistorted_depth(kDepthWidth, kDepthHeight,   4),
      registered_rgb   (kDepthWidth, kDepthHeight,   4), // rgb -> depth
      registered_depth (kRgbWidth,   kRgbHeight + 2, 4); // depth -> rgb

  cv::Mat rgb_mat, depth_mat_raw, depth_mat;

  /// Use this to substitute tedious argv parsing
  RuntimeParams args;
  LoadRuntimeParams("../config/args.yml", args);

  ConfigManager config;
  RGBDDataProvider rgbd_local_sequence;

  DatasetType dataset_type = DatasetType(args.dataset_type);
  config.LoadConfig(dataset_type);
  rgbd_local_sequence.LoadDataset(dataset_type);
  Sensor sensor(config.sensor_params);
  MainEngine main_engine(
      config.hash_params,
      config.sdf_params,
      config.mesh_params,
      config.sensor_params,
      config.ray_caster_params
  );
  main_engine.ConfigMappingEngine(
      true
  );

  gl::Lighting light;
  light.Load("../config/lights.yaml");
  main_engine.ConfigVisualizingEngine(
      light,
      args.enable_navigation,
      args.enable_global_mesh,
      args.enable_bounding_box,
      args.enable_trajectory,
      args.enable_polygon_mode,
      args.enable_meshing,
      args.enable_ray_casting,
      args.enable_color
  );

  main_engine.ConfigLoggingEngine(
      ".",
      args.enable_video_recording,
      args.enable_ply_saving
  );
  main_engine.enable_sdf_gradient() = args.enable_sdf_gradient;

  ORB_SLAM2::System orb_slam_engine(
      path_to_vocabulary,
      orb_configs[dataset_type],
      ORB_SLAM2::System::RGBD,
      true);

  double tframe;
  cv::Mat color, depth;
  float4x4 wTc, cTw;
  int frame_count = 0;
  bool begin = true;
  //std::cin >> begin;
  while (true) {
    if (main_engine.vis_engine_.window_.get_key(GLFW_KEY_ESCAPE) == GLFW_PRESS) {
      break;
    }

    listener.waitForNewFrame(frames);
    libfreenect2::Frame *rgb   = frames[libfreenect2::Frame::Color];
    // libfreenect2::Frame *ir    = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    // 2. Register frames
    registration->apply(rgb, depth, &undistorted_depth, &registered_rgb,
                        true, &registered_depth);
    rgb_mat       = cv::Mat(kRgbHeight,     kRgbWidth, CV_8UC4,
                            rgb->data);
    depth_mat_raw = cv::Mat(kRgbHeight + 2, kRgbWidth, CV_32F,
                            registered_depth.data);
    depth_mat     = depth_mat_raw(cv::Rect(0, 1, kRgbWidth, kRgbHeight));

    // 3. Collect desired depth and color frames
    cv::resize(rgb_mat,   rgb_mat,   cv::Size(kSaveWidth, kSaveHeight));
    cv::resize(depth_mat, depth_mat, cv::Size(kSaveWidth, kSaveHeight),
               0, 0, cv::INTER_NEAREST);
    cv::flip(rgb_mat,   rgb_mat,   1);
    cv::flip(depth_mat, depth_mat, 1);

    frame_count++;

    cv::Mat color_slam = rgb_mat.clone();
    cv::Mat depth_slam = depth_mat.clone();
    cv::Mat cTw_orb = orb_slam_engine.TrackRGBD(color_slam, depth_slam, tframe);
    listener.release(frames);

    if (cTw_orb.empty()) continue;
    cTw = MatTofloat4x4(cTw_orb);
    wTc = cTw.getInverse();

    sensor.Process(depth_slam, color_slam, true); // abandon wTc
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

    main_engine.Mapping(sensor);
    main_engine.Meshing();
    if (main_engine.Visualize(cTw))
      break;

    main_engine.Log();
    main_engine.Recycle();
    int key = cv::waitKey(wait_time[wait_time_idx]);
    if (key == 27) return 0;
    else if (key == 13) wait_time_idx = 1;
    else if (key == 32) wait_time_idx = 1 - wait_time_idx; /* Toggle */
  }

  dev->stop();
  dev->close();
  delete registration;

  main_engine.FinalLog();

  orb_slam_engine.Shutdown();
  return 0;
}