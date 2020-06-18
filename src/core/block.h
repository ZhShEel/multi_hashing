//
// Created by wei on 17-5-21.
//

#ifndef CORE_BLOCK_H
#define CORE_BLOCK_H

#include "core/common.h"
#include "core/voxel.h"

#include <helper_math.h>

#define BLOCK_LIFE 3
// Typically Block is a 8x8x8 voxel array
struct __ALIGN__(8) Block {
  int active;
  float min_active_voxel_pi;

  int inner_surfel_count;
  int boundary_surfel_count;
  int life_count_down;
  bool join_value;
  int update_mutex;

  Voxel voxels[BLOCK_SIZE];
  MeshUnit mesh_units[BLOCK_SIZE];
  PrimalDualVariables primal_dual_variables[BLOCK_SIZE];

  Voxel shells[SHELL_SIZE];
  MeshUnit shells_meshes[SHELL_SIZE];

  __host__ __device__
  void Clear() {
    active = 0;
    min_active_voxel_pi = 0;
    inner_surfel_count = 0;
    boundary_surfel_count = 0;
    life_count_down = BLOCK_LIFE;
    join_value = 0;
    update_mutex = 0;

#ifdef __CUDA_ARCH__ // __CUDA_ARCH__ is only defined for __device__
#pragma unroll 8
#endif
    for (int i = 0; i < BLOCK_SIZE; ++i) {

      voxels[i].Clear();
      mesh_units[i].Clear();
      primal_dual_variables[i].Clear();
    }
      for (int i = 0; i < SHELL_SIZE; ++i) {
          shells[i].Clear();
          shells_meshes[i].Clear();
      }
  }

  __host__ __device__
    void ClearShell(){
        for (int i = 0;i < SHELL_SIZE; ++i){
            shells[i].Clear();
            shells_meshes[i].Clear();
        }
    }

  __host__ __device__
    float getSDF(int index){
        return voxels[index].sdf;
    }

  __device__
  void Update_JoinValue(float a){
    #if __CUDA_ARCH__ >=200
    bool next = true;
    while(next){
      int v = atomicCAS(&update_mutex,0,1);
      if(v==0){
        join_value = 1;
        atomicExch(&update_mutex,0);
        next = false;
      }
    }
    #endif
  }
};

#endif // CORE_BLOCK_H
