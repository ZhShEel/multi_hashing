//
// Created by wei on 17-4-30.
//

#ifndef VH_DATA_MANAGER
#define VH_DATA_MANAGER

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "matrix.h"
#include "../core/params.h"

enum DatasetType {
  ICL = 0,
  TUM1 = 1,
  TUM2 = 2,
  TUM3 = 3,
  ZHOU = 4,
  SUN3D = 5,
  PKU = 6,
  BOE = 7
};

// Deprecated:
struct Dataset {
  DatasetType type;
  std::string path;
};

void LoadRuntimeParams(std::string path, RuntimeParams &params);
void LoadScaleTableParams(std::string path, ScaleParams &params);
void LoadHashParams(std::string path, HashParams& params);
void LoadMeshParams(std::string path, MeshParams &params);
void LoadVolumeParams(std::string path, VolumeParams& params);
void LoadSensorParams(std::string path, SensorParams& params);
void LoadRayCasterParams(std::string path, RayCasterParams& params);

/// 1-1-1 correspondences
void LoadICL(
        std::string dataset_path,
        std::vector<std::string> &depth_image_list,
        std::vector<std::string> &color_image_list,
        std::vector<float4x4>&   wTcs);
void LoadZHOU(
    std::string dataset_path,
    std::vector<std::string> &depth_img_list,
    std::vector<std::string> &color_img_list,
    std::vector<float4x4> &wTcs);
void LoadSUN3D(
    std::string dataset_path,
    std::vector<std::string> &depth_img_list,
    std::vector<std::string> &color_img_list,
    std::vector<float4x4> &wTcs);
void Load3DVCR(
        std::string dataset_path,
        std::vector<std::string> &depth_image_list,
        std::vector<std::string> &color_image_list,
        std::vector<float4x4>    &wTcs);

/// no 1-1-1 correspondences
void LoadTUM(
        std::string dataset_path,
        std::vector<std::string> &depth_image_list,
        std::vector<std::string> &color_image_list,
        std::vector<float4x4>    &wTcs);

struct ConfigManager {
  ScaleParams     scale_params;
  HashParams      hash_params;
  MeshParams      mesh_params;
  VolumeParams       sdf_params;
  SensorParams    sensor_params;
  RayCasterParams ray_caster_params;
  std::string     data_path;

  void LoadConfig(DatasetType dataset_type);
  void LoadConfig(std::string config_path);
};



#endif //VH_DATA_MANAGER
