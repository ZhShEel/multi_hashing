//
// Created by Yan on 19-1-13.
//
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

#include <helper_cuda.h>
#include <chrono>

#include <string>
#include <cuda_runtime.h>
#include <glog/logging.h>


#include "util/timer.h"
#include <queue>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <engine/visualizing_engine.h>
#include <io/mesh_writer.h>
#include <meshing/marching_cubes.h>
#include <visualization/compress_mesh.h>

#include "sensor/point_cloud_provider.h"
#include "sensor/rgbd_sensor.h"
#include "visualization/ray_caster.h"

#include "io/config_manager.h"
#include "core/PointCloud.h"
#include "core/collect_block_array.h"
#include "glwrapper.h"

#ifndef ECCV18_PSDF_H
#define ECCV18_PSDF_H

class psdf {
public:

    psdf(const std::string& config_path);
  ~psdf();

  //MainEngine main_engine;
  RuntimeParams args;
  ConfigManager config;
  DatasetType dataset_type;
  Sensor sensor;
  MainEngine main_engine;
  gl::Lighting lighting;
  void normalizeDepthImage(cv::Mat& depth, cv::Mat& disp);
  // int reconstruction(cv::Mat img, cv::Mat dpt, float4x4 wTc);
  int reconstruction(pcl::PointCloud<pcl::PointXYZRGB>,
                     pcl::gpu::DeviceArray< pcl::PointXYZRGB > pc,
                     pcl::gpu::DeviceArray< pcl::Normal >,
                     pcl::gpu::DeviceArray< pcl::PrincipalCurvatures >);
  int reconstruction( PointCloud);

private:
  PointCloud pc_last;
  float4x4 cTw_last;
};

#endif //ECCV18_PSDF_H
