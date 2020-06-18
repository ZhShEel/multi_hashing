//
// Created by wei on 17-4-2.
//
// Geometric utilities:
// 1. Transformation of Coordinate systems with different units
// 2. Projection, reprojection and viewing frustum determination
#ifndef VH_GEOMETRY_UTIL_H
#define VH_GEOMETRY_UTIL_H

#include <pcl/features/principal_curvatures.h>
#include "../core/common.h"
#include "../core/params.h"

/// There are 3 kinds of positions (pos)
/// 1. world pos, unit: meter
/// 2. voxel pos, unit: voxel (typically 0.004m)
/// 3. block pos, unit: block (typically 8 voxels)

//////////
/// Transforms between world, voxel, and block coordinate systems
/// Semantic: A pos To B pos; A, B in {world, voxel, block}
/// float is only used to do interpolation

/// World <-> Voxel

__device__ __host__
__inline__ float squaref(float a) {
  return a*a;
}

__device__ __host__
__inline__ float gaussian(float x, float u, float squared_sigma) {
  return 1 / (sqrt(2*M_PI)*sqrt(squared_sigma))
         * exp(- squaref(x - u) / squared_sigma);
}

struct GeometryHelper {
  float voxel_size;
  float truncation_distance;
  float truncation_distance_scale;
  float approximate_size;
  float sdf_upper_bound;
  float weight_sample;
  float regularization_rough_measure;
  float join_th;

  GeometryHelper() = default;
  void Init(const VolumeParams &params) {
    voxel_size = params.voxel_size;
    truncation_distance_scale = params.truncation_distance_scale;
    approximate_size = params.approximate_size;
    truncation_distance = params.truncation_distance;
    sdf_upper_bound = params.sdf_upper_bound;
    weight_sample = params.weight_sample;
    regularization_rough_measure = params.regularization_rough_measure;
    join_th = params.join_th;
  }
  GeometryHelper(const VolumeParams &params) {
    Init(params);
  }

  __host__ __device__
  inline
  float3 WorldToVoxelf(const float3 world_pos) {
    return world_pos / voxel_size;
  }

  __host__ __device__
  inline
  int3 WorldToVoxeli(const float3 world_pos, float voxel_size_){
    const float3 p = world_pos / voxel_size_;
    return make_int3(p + make_float3(sign(p)) * 0.5f);
  }

  __host__ __device__
  inline
  int3 WorldToVoxeli(const float3 world_pos) {
    const float3 p = world_pos / voxel_size;
    return make_int3(p + make_float3(sign(p)) * 0.5f);
  }

  __host__ __device__
  inline // no offset here: value is stored on the corner of a voxel
  float3 VoxelToWorld(const int3 voxel_pos) {
    return make_float3(voxel_pos) * voxel_size;
  }

/// Voxel <-> Block
// Corner voxel with smallest xyz
  __host__ __device__
  inline
  int3 BlockToVoxel(const int3 block_pos) {
    return block_pos * BLOCK_SIDE_LENGTH;
  }

  __host__ __device__
  inline
  int3 VoxelToBlock(const int3 voxel_pos) {
    int3 pos = voxel_pos;
    pos.x -= (voxel_pos.x < 0) * (BLOCK_SIDE_LENGTH - 1);
    pos.y -= (voxel_pos.y < 0) * (BLOCK_SIDE_LENGTH - 1);
    pos.z -= (voxel_pos.z < 0) * (BLOCK_SIDE_LENGTH - 1);
    return pos / BLOCK_SIDE_LENGTH;
  }

  __host__ __device__
  inline
  uint3 VoxelToOffset(const int3 block_pos, const int3 voxel_pos) {
    int3 offset = voxel_pos - BlockToVoxel(block_pos);
    return make_uint3(offset);
  }

  __host__ __device__
  inline
  uint3 VoxelToOffset(const int3 block_pos, const int3 voxel_pos, const int scale) {
    int3 offset = (voxel_pos - BlockToVoxel(block_pos))/pow(2,scale-1);
      int3 tmp = BlockToVoxel(block_pos);
      //if(offset.x==-1)
       // printf("%d %d %d->%d %d %d scale:%d\n",voxel_pos.x,voxel_pos.y,voxel_pos.z,tmp.x,tmp.y,tmp.z,scale);
    return make_uint3(offset);
  }

  /// Block <-> World
  __host__ __device__
  inline
  float3 BlockToWorld(const int3 block_pos) {
    return VoxelToWorld(BlockToVoxel(block_pos));
  }

  __host__ __device__
  inline
  int3 WorldToBlock(const float3 world_pos, float voxel_size_) {
    return VoxelToBlock(WorldToVoxeli(world_pos, voxel_size_));
  }

  __host__ __device__
  inline
  int3 WorldToBlock(const float3 world_pos) {
    return VoxelToBlock(WorldToVoxeli(world_pos));
  }

//////////
/// Transforms between coordinates and indices
/// Idx means local idx inside a block \in [0, 512)
  __host__ __device__
  inline
  uint3 DevectorizeIndex(const uint idx) {
    uint x = idx % BLOCK_SIDE_LENGTH;
    uint y = (idx % (BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH)) / BLOCK_SIDE_LENGTH;
    uint z = idx / (BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH);
    return make_uint3(x, y, z);
  }

/// Computes the linearized index of a local virtual voxel pos; pos \in [0, 8)^3
  __host__ __device__
  inline
  uint VectorizeOffset(const uint3 voxel_local_pos) {
    return voxel_local_pos.z * BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH +
           voxel_local_pos.y * BLOCK_SIDE_LENGTH +
           voxel_local_pos.x;
  }

/// Transforms between coordinates and indices of shell voxels
// 9*9(x:8 y:0-8 z:0-8)+8*9(x:0-7 y:8 z:0-8)+8*8(x:0-7 y:0-7 z:8)
  __host__ __device__
  inline
  uint3 DevectorizeShellIndex(const uint idx) {
      if(idx>=0&&idx<81){
          uint y = idx / (BLOCK_SIDE_LENGTH + 1);
          uint z = idx % (BLOCK_SIDE_LENGTH + 1);
          return make_uint3(8,y,z);
      } 
      else if(idx>=81&&idx<153){
          int offset = idx - 81;
          uint x = idx / (BLOCK_SIDE_LENGTH + 1);
          uint z = idx % (BLOCK_SIDE_LENGTH + 1);
          return make_uint3(x,8,z);
      }
      else if(idx>=153&&idx<217){
          int offset = idx - 153;
          uint x = idx / BLOCK_SIDE_LENGTH;
          uint y = idx % BLOCK_SIDE_LENGTH;
          return make_uint3(x,y,8);
      }
      else
         return make_uint3(0,0,0);
  }

  __host__ __device__
  inline
  uint VectorizeShellOffset(uint3 voxel_local_pos) {
      uint x = voxel_local_pos.x; uint y = voxel_local_pos.y;  uint z = voxel_local_pos.z;
      if(x==8){
          return y * (BLOCK_SIDE_LENGTH + 1) + z;
      }
      else if(y==8){
          return x * (BLOCK_SIDE_LENGTH + 1) + z + 81;
      }
      else if(z==8){
          return x * BLOCK_SIDE_LENGTH + y + 153;
      }
      else
          return 255;
  }

  __host__ __device__
  inline
  bool IsInShell(uint3 pos){
      return (pos.x==8||pos.y==8||pos.z==8);
  }

  __host__ __device__
  inline
  bool IsInShell(int index){
      return (index>=0&&index<217);
  }

//////////
/// Truncating distance
  __host__ __device__
  inline
  float truncate_distance(const float z) {
    return truncation_distance +truncation_distance_scale * z;
  }

//////////
/// Projections and reprojections
/// Between the Camera coordinate system and the image plane
/// Projection
  __host__ __device__
  inline
  float2 CameraProjectToImagef(const float3 camera_pos,
                               float fx, float fy, float cx, float cy) {
    return make_float2(camera_pos.x * fx / camera_pos.z + cx,
                       camera_pos.y * fy / camera_pos.z + cy);
  }

  __host__ __device__
  inline
  int2 CameraProjectToImagei(
      const float3 camera_pos,
      float fx, float fy, float cx, float cy) {
    float2 uv = CameraProjectToImagef(camera_pos, fx, fy, cx, cy);
    return make_int2(uv + make_float2(0.5f, 0.5f));
  }

  __host__ __device__
  inline
  float3 ImageReprojectToCamera(
      uint ux, uint uy, float depth,
      float fx, float fy, float cx, float cy) {
    const float x = ((float) ux - cx) / fx;
    const float y = ((float) uy - cy) / fy;
    return make_float3(depth * x, depth * y, depth);
  }

/// R^3 -> [0, 1]^3
/// maybe used for rendering
  __host__ __device__
  inline
  float NormalizeDepth(float z, float min_depth, float max_depth) {
    return (z - min_depth) / (max_depth - min_depth);
  }

  inline
  float DenormalizeDepth(float z, float min_depth, float max_depth) {
    return z * (max_depth - min_depth) + min_depth;
  }

/// View frustum test
  __host__ __device__
  inline
  bool IsPointInCameraFrustum(const float4x4 &c_T_w,
                              const float3 world_pos,
                              const SensorParams &sensor_params) {
    float3 camera_pos = c_T_w * world_pos;
    float2 uv = CameraProjectToImagef(camera_pos,
                                      sensor_params.fx, sensor_params.fy,
                                      sensor_params.cx, sensor_params.cy);
    float3 normalized_p = make_float3(
        (2.0f * uv.x - (sensor_params.width - 1.0f)) / (sensor_params.width - 1.0f),
        ((sensor_params.height - 1.0f) - 2.0f * uv.y) / (sensor_params.height - 1.0f),
        NormalizeDepth(camera_pos.z,
                       sensor_params.min_depth_range,
                       sensor_params.max_depth_range));

    normalized_p *= 0.95;
    return !(normalized_p.x < -1.0f || normalized_p.x > 1.0f
             || normalized_p.y < -1.0f || normalized_p.y > 1.0f
             || normalized_p.z < 0.0f || normalized_p.z > 1.0f);
  }

  __host__ __device__
  inline
  bool IsBlockInCameraFrustum(float4x4 c_T_w,
                              const int3 block_pos,
                              const SensorParams &sensor_params) {
    float3 world_pos = VoxelToWorld(BlockToVoxel(block_pos))
                       + voxel_size * 0.5f * (BLOCK_SIDE_LENGTH - 1.0f);
    return IsPointInCameraFrustum(c_T_w, world_pos, sensor_params);
  }
};

#endif //VH_GEOMETRY_UTIL_H
