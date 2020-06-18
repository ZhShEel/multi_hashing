//
// Created by wei on 17-4-5.
//
// MainEngine: managing HashTable<Block> and might be other structs later

#ifndef ENGINE_MAIN_ENGINE_H
#define ENGINE_MAIN_ENGINE_H

#include <pcl/common/common.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/gpu/containers/device_array.h>

#include <cuda_runtime.h>

#include "core/PointCloud.h"
#include "core/hash_table.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "core/mesh.h"

#include "engine/visualizing_engine.h"
#include "engine/logging_engine.h"
#include "visualization/compact_mesh.h"
#include "visualization/bounding_box.h"
#include "localizing/icp.h"
#include "sensor/rgbd_sensor.h"
#include "meshing/remeshing.h"
#include "mapping_engine.h"
#include "geometry/geometry_helper.h"

class MainEngine {
public:
  // configure main data
  MainEngine(){};

  long init(
      const ScaleParams &scale_params,
      const HashParams &hash_params,
      const VolumeParams &volume_params,
      const MeshParams &mesh_params,
      const SensorParams &sensor_params,
      const RayCasterParams &ray_caster_params
  );

  ~MainEngine();

  void Reset();

  // configure engines
  void ConfigMappingEngine(
      bool enable_bayesian_update
  );

  void ConfigLocalizingEngine();

  void ConfigVisualizingEngine(
      gl::Lighting &light,
      bool enable_navigation,
      bool enable_global_mesh,
      bool enable_bounding_box,
      bool enable_trajectory,
      bool enable_polygon_mode,
      bool enable_meshing,
      bool enable_ray_caster,
      bool enable_join_process,
      bool enable_color
  );

  void ConfigLoggingEngine(
      std::string path,
      bool enable_video,
      bool enable_ply
  );

  void Localizing(Sensor &sensor, int iters, float4x4 &gt);

  float4x4 Icp(PointCloud&, PointCloud&, float4x4&);

  void Mapping(pcl::PointCloud<pcl::PointXYZRGB>&,
               pcl::gpu::DeviceArray< pcl::PointXYZRGB >& pc,
               pcl::gpu::DeviceArray< pcl::Normal >&,
               pcl::gpu::DeviceArray< pcl::PrincipalCurvatures >&);

  void Mapping(PointCloud, float4x4);

  void Mapping(Sensor &sensor);

  void Meshing();

  void Recycle();

  int Visualize(float4x4 view);

  int Visualize(float4x4 view, float4x4 view_gt);

  void Log();

  void RecordBlocks(std::string prefix = "");

  void FinalLog();

  void FinalLog_without_meshing();

  void TempLog(int);

  const int &frame_count() {
    return integrated_frame_count_;
  }

  bool &enable_sdf_gradient() {
    return enable_sdf_gradient_;
  }
  bool &color_type() {
    return color_type_;
  }

private:
  // Engines
  MappingEngine map_engine_;
public:
  VisualizingEngine vis_engine_;
  LoggingEngine log_engine_;

  // Core
  ScaleTable scale_table_;
  HashTable hash_table_;
  BlockArray blocks_;
  EntryArray candidate_entries_;

  // Meshing
  Mesh mesh_;

  // Geometry
  GeometryHelper geometry_helper_;

  int integrated_frame_count_ = 0;
  bool enable_sdf_gradient_;
  bool color_type_ = 0;

  ScaleParams scale_params_;
  HashParams hash_params_;
  VolumeParams volume_params_;
  MeshParams mesh_params_;
  SensorParams sensor_params_;
  RayCasterParams ray_caster_params_;

  //timing
  double allocate_time_sum;
  double join_time_sum;
  double update_time_sum;
  double marching_cube_time_sum;
  double recycle_time_sum;
};


#endif //VH_MAP_H
