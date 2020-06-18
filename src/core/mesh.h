//
// Created by wei on 17-5-21.
//

#ifndef CORE_MESH_H
#define CORE_MESH_H

#include "core/common.h"
#include "core/params.h"
#include "core/vertex.h"
#include "core/triangle.h"

#include <helper_cuda.h>
#include <helper_math.h>

class Mesh {
public:
  __host__ Mesh() = default;
  // __host__ ~Mesh();

  __host__ void Alloc(const MeshParams &mesh_params);
  __host__ void Resize(const MeshParams &mesh_params);
  __host__ void Free();
  __host__ void Reset();

  const MeshParams& params() {
    return mesh_params_;
  }

  __device__ __host__ Vertex& vertex(uint i) {
    return vertices[i];
  }
  __device__ __host__ Triangle& triangle(uint i) {
    return triangles[i];
  }

  __host__ uint vertex_heap_count();
  __host__ uint triangle_heap_count();

private:
  bool is_allocated_on_gpu_ = false;
  uint*     vertex_heap_;
  uint*     vertex_heap_counter_;
  Vertex*   vertices;

  uint*     triangle_heap_;
  uint*     triangle_heap_counter_;
  Triangle* triangles;

#ifdef __CUDACC__
public:
  __host__ __device__
  uint GetRestVertex(){
    return vertex_heap_counter_[0];
  }

  __host__ __device__
  uint GetRestTriangle(){
    return triangle_heap_counter_[0];
  }

  __device__
  uint AllocVertex() {
    uint addr = atomicSub(&vertex_heap_counter_[0], 1);
    if (addr < MEMORY_LIMIT) {
      printf("vertex heap: %d -> %d\n", addr, vertex_heap_[addr]);
    }
    return vertex_heap_[addr];
  }
  __device__
  void FreeVertex(uint ptr) {
    uint addr = atomicAdd(&vertex_heap_counter_[0], 1);
    vertex_heap_[addr + 1] = ptr;
  }

  __device__
  uint AllocTriangle() {
    uint addr = atomicSub(&triangle_heap_counter_[0], 1);
    if (addr < MEMORY_LIMIT) {
      printf("triangle heap: %d -> %d\n", addr, triangle_heap_[addr]);
    }
    return triangle_heap_[addr];
  }
  __device__
  void FreeTriangle(uint ptr) {
    uint addr = atomicAdd(&triangle_heap_counter_[0], 1);
    triangle_heap_[addr + 1] = ptr;
  }

  /// Release is NOT always a FREE operation
  __device__
  void ReleaseTriangle(Triangle& triangle) {
    int3 vertex_ptrs = triangle.vertex_ptrs;
    atomicSub(&vertices[vertex_ptrs.x].ref_count, 1);
    atomicSub(&vertices[vertex_ptrs.y].ref_count, 1);
    atomicSub(&vertices[vertex_ptrs.z].ref_count, 1);
  }

  __device__
  void AssignTriangle(Triangle& triangle, int3 vertex_ptrs, int scale) {
    triangle.vertex_ptrs = vertex_ptrs;
    triangle.scale = scale;
    //printf("vertex_ptr:%d %d %d\n",vertex_ptrs.x,vertex_ptrs.y,vertex_ptrs.z);
    //if(vertex_ptrs.x<0||vertex_ptrs.y<0||vertex_ptrs.z<0||vertex_ptrs.x>=4000000||vertex_ptrs.y>=4000000||vertex_ptrs.z>=4000000){
     // printf("vertex_ptr:%d %d %d\n",vertex_ptrs.x, vertex_ptrs.y, vertex_ptrs.z);
    //  return;
   // }

    atomicAdd(&vertices[vertex_ptrs.x].ref_count, 1);
    atomicAdd(&vertices[vertex_ptrs.y].ref_count, 1);
    atomicAdd(&vertices[vertex_ptrs.z].ref_count, 1);
  }

  __device__
  void ComputeTriangleNormal(Triangle& triangle) {
    int3 vertex_ptrs = triangle.vertex_ptrs;
    float3 p0 = vertices[vertex_ptrs.x].pos;
    float3 p1 = vertices[vertex_ptrs.y].pos;
    float3 p2 = vertices[vertex_ptrs.z].pos;
    float3 n = normalize(cross(p2 - p0, p1 - p0));
    if(isnan(n.x)||isnan(n.y)||isnan(n.z))
      return;
    //printf("p0:%f %f %f(%d) - p1:%f %f %f(%d) - p2:%f %f %f(%d) n:%f %f %f\n",p0.x,p0.y,p0.z,vertex_ptrs.x,p1.x,p1.y,p1.z,vertex_ptrs.y,p2.x,p2.y,p2.z,vertex_ptrs.z,n.x,n.y,n.z);
    vertices[vertex_ptrs.x].normal = n;
    vertices[vertex_ptrs.y].normal = n;
    vertices[vertex_ptrs.z].normal = n;
    return;
  }
#endif // __CUDACC__
  MeshParams mesh_params_;

};

#endif //VOXEL_HASHING_MESH_H
