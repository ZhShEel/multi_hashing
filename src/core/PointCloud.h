#ifndef CORE_POINTCLOUD_H
#define CORE_POINTCLOUD_H

#include "core/common.h"
#include "core/voxel.h"
#include "core/Point.h"

#include <helper_math.h>

//First compute the normal using PCL and fill them in Point structure
//TODO: compute normal here

class __ALIGN__(8) PointCloud{
public:
  __host__ PointCloud() = default;
  __host__ explicit PointCloud(uint pc_size);

  __host__ void Alloc(uint pc_size);
  __host__ void Resize(uint pc_size);
  __host__ void Free();

  __host__ void Reset();

  __host__ void TransferPtToGPU(Point* pc);

  __host__ void TransferPtToHost(Point* pc);

  __host__ __device__ Point& operator[] (uint i) {
    return points_[i];
  }
  __host__ __device__ const Point& operator[] (uint i) const {
    return points_[i];
  }

  __host__ Point* GetGPUPtr() const{
    return points_;
  }

  // __host__ Tnode* GetKdTree() const{
  //   return kdroot_;
  // }

  __host__ uint count(){
    return pc_size_;
  }

  // __host__ void GenerateKdTree();

  // __host__ Point FindNearestByKdTree(Point);

  // __host__ void FreeKdTree();

  float3 bbx_min, bbx_max;

private:
  bool is_allocated_on_gpu_ = false;
  Point* points_;
  uint pc_size_; 
  // Tnode* kdroot_;
};


#endif // CORE_POINTCLOUD_H