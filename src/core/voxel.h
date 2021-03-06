//
// Created by wei on 17-10-21.
//

#ifndef CORE_VOXEL_H
#define CORE_VOXEL_H

#include "core/common.h"
#include "helper_math.h"
#include "core/mesh.h"
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Statistics typically reserved for Voxels
// float: *Laplacian* and *entropy* are intuitive statistics
// float: *duration* is time-interval that the voxel exists
struct __ALIGN__(4) Stat {
  float laplacian;
  float entropy;
  float duration;

  __host__ __device__
  void Clear() {
    laplacian = 0;
    entropy = 0;
    duration = 0;
  }
};

struct __ALIGN__(4) MeshUnit {
  // mesh
  int vertex_ptrs   [N_VERTEX];  // 3
  int vertex_mutexes[N_VERTEX];  // 3
  int triangle_ptrs [N_TRIANGLE];// 5
  short curr_cube_idx, prev_cube_idx;
  float measurement_term;
  int update_mutex;

  __host__ __device__
  void ResetMutexes() {
    vertex_mutexes[0] = FREE_PTR;
    vertex_mutexes[1] = FREE_PTR;
    vertex_mutexes[2] = FREE_PTR;
    update_mutex = 0;
  }

  __host__ __device__
  int GetVertex(int idx) {
    return vertex_ptrs[idx];
  }

  // __device__
  // void lock(){
  //   #if __CUDA_ARCH__ >=200
  //     while(atomicCAS(&update_mutex, 0, 1) != 0);
  //   #endif
  // }

  // __device__
  // void unlock(){
  //   #if __CUDA_ARCH__ >=200
  //     atomicExch(&update_mutex, 0);
  //   #endif
  // }
  __device__
  void Update_Measurement(float a){
    #if __CUDA_ARCH__ >=200
    bool next = true;
    while(next){
      int v = atomicCAS(&update_mutex,0,1);
      if(v==0){
        if(measurement_term < 0)
          measurement_term = a;
        else
          measurement_term += a;

        atomicExch(&update_mutex,0);
        next = false;
      }
    }
    #endif
  }
 
  __host__ __device__
  void Clear() {
    vertex_ptrs[0] = vertex_mutexes[0] = FREE_PTR;
    vertex_ptrs[1] = vertex_mutexes[1] = FREE_PTR;
    vertex_ptrs[2] = vertex_mutexes[2] = FREE_PTR;

    triangle_ptrs[0] = FREE_PTR;
    triangle_ptrs[1] = FREE_PTR;
    triangle_ptrs[2] = FREE_PTR;
    triangle_ptrs[3] = FREE_PTR;
    triangle_ptrs[4] = FREE_PTR;

    curr_cube_idx = prev_cube_idx = 0;
    measurement_term = -1;
  }

  /*
#ifdef __CUDACC__
  __host__ __device__
  void Clear(Mesh& mesh_) {
    for(uint i=0;i<3;i++){
        if(vertex_ptrs[i]!=FREE_PTR)
            mesh.FreeVertex(vertex_ptrs[i]);
    }
    vertex_ptrs[0] = vertex_mutexes[0] = FREE_PTR;
    vertex_ptrs[1] = vertex_mutexes[1] = FREE_PTR;
    vertex_ptrs[2] = vertex_mutexes[2] = FREE_PTR;
    
    for(uint i=0;i<5;i++){
        if(triangle_ptrs[i]!=FREE_PTR)
            mesh.FreeTriangle(triangle_ptrs[i]);
    }
    mesh.FreeTriangle(triangle_ptrs[0]);
    triangle_ptrs[0] = FREE_PTR;
    triangle_ptrs[1] = FREE_PTR;
    triangle_ptrs[2] = FREE_PTR;
    triangle_ptrs[3] = FREE_PTR;
    triangle_ptrs[4] = FREE_PTR;

    curr_cube_idx = prev_cube_idx = 0;
    measurement_term = -1;
  }
#endif
*/
};

struct __ALIGN__(4) PrimalDualVariables {
  bool   mask;
  float  sdf0, sdf_bar, inv_sigma2;
  float3 p;

  __host__ __device__
  void operator = (const PrimalDualVariables& pdv) {
    mask = pdv.mask;
    sdf0 = pdv.sdf0;
    sdf_bar = pdv.sdf_bar;
    p = pdv.p;
  }

  __host__ __device__
  void Clear() {
    mask = false;
    sdf0 = sdf_bar = 0;
    inv_sigma2 = 0;
    p = make_float3(0);
  }
};

struct __ALIGN__(8) Voxel {
  float  sdf;    // signed distance function, mu
  float  inv_sigma2; // sigma
  float  a, b;
  uchar3 color;  // color
  float3 offset;

  __host__ __device__
  void operator = (const Voxel& v) {
    sdf = v.sdf;
    inv_sigma2 = v.inv_sigma2;
    color = v.color;
    a = v.a;
    b = v.b;
  }

  __host__ __device__
  void Clear() {
    sdf = inv_sigma2 = 0.0f;
    color = make_uchar3(0, 0, 0);
    offset = make_float3(0, 0, 0);
    a = 10; b = 10;
  }


  // __host__ __device__
  // void Update(float sdf_, float inv_sigma2_, uchar3 color_){
  //   float3 c_prev  = make_float3(color.x, color.y, color.z);
  //   float3 c_delta = make_float3(color_.x, color_.y, color_.z);
  //   float3 c_curr  = 0.5f * c_prev + 0.5f * c_delta;
  //   color = make_uchar3(c_curr.x + 0.5f, c_curr.y + 0.5f, c_curr.z + 0.5f);

  //   sdf = (sdf * inv_sigma2 +sdf_ * inv_sigma2_) / (inv_sigma2 + inv_sigma2_);
  //   // sdf = (sdf * inv_sigma2 + delta.sdf * delta.inv_sigma2) / (inv_sigma2 + delta.inv_sigma2); 
  //   inv_sigma2 = fminf(inv_sigma2 + inv_sigma2_, 255.0f);
  // }

  __host__ __device__
  void Check(int3 pos, int ptr, int local_id){
    
    // sdf = (sdf * inv_sigma2 + delta.sdf * delta.inv_sigma2) / (inv_sigma2 + delta.inv_sigma2); 
    if(pos.x==23&&local_id<=50&&pos.z==0&&pos.y>=11&&pos.y<=19)
        printf("voxel check->sdf:%f weight:%f pos:%d %d %d ptr:%d local_id:%d\n",sdf,inv_sigma2,pos.x,pos.y,pos.z,ptr,local_id);
 
  }

  __host__ __device__
  void Update(const Voxel delta) {
    // float3 c_prev  = make_float3(color.x, color.y, color.z);
    // float3 c_delta = make_float3(delta.color.x, delta.color.y, delta.color.z);
    // float3 c_curr  = 0.5f * c_prev + 0.5f * c_delta;
    // color = make_uchar3(c_curr.x + 0.5f, c_curr.y + 0.5f, c_curr.z + 0.5f);
    float temp = sdf;
    float new_value = (sdf * inv_sigma2 + delta.sdf * delta.inv_sigma2) / (inv_sigma2 + delta.inv_sigma2);
    sdf = new_value;
    // sdf = (sdf * inv_sigma2 + delta.sdf * delta.inv_sigma2) / (inv_sigma2 + delta.inv_sigma2); 
    inv_sigma2 = fminf(inv_sigma2 + delta.inv_sigma2, 255.0f);
  }
};

#endif // CORE_VOXEL_H
