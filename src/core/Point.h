#ifndef CORE_POINT_H
#define CORE_POINT_H

#include "core/common.h"
#include "core/voxel.h"

struct __ALIGN__(8) Point{
  float x,y,z;
  float normal_x,normal_y,normal_z;

  __host__ __device__
  void Clear() {
    x = y = z = 0;
    normal_x = normal_y = normal_z = 0;
  }
};

#endif