//
// Created by wei on 17-10-22.
//

#include "mapping/recycle.h"

////////////////////
/// Device code
////////////////////
#include "core/common.h"
#include "core/entry_array.h"
#include "core/block_array.h"
#include "helper_math.h"

__global__
void StarveOccupiedBlocksKernel(
    EntryArray candidate_entries,
    BlockArray blocks
) {
  const uint idx = blockIdx.x;
  const HashEntry& entry = candidate_entries[idx];
  float inv_sigma2 = blocks[entry.ptr].voxels[threadIdx.x].inv_sigma2;
  inv_sigma2 = fmaxf(0, inv_sigma2 - 1.0f);
  blocks[entry.ptr].voxels[threadIdx.x].inv_sigma2 = inv_sigma2;
}

/// Collect dead voxels
__global__
void CollectGarbageBlockArrayKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    GeometryHelper geometry_helper
) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = candidate_entries[idx];

  Voxel v0 = blocks[entry.ptr].voxels[2*threadIdx.x+0];
  Voxel v1 = blocks[entry.ptr].voxels[2*threadIdx.x+1];

  float sdf0 = v0.sdf, sdf1 = v1.sdf;
  if (v0.inv_sigma2 < EPSILON)	sdf0 = PINF;
  if (v1.inv_sigma2 < EPSILON)	sdf1 = PINF;

  __shared__ float	shared_min_sdf   [BLOCK_SIZE / 2];
  __shared__ float	shared_max_inv_sigma2[BLOCK_SIZE / 2];
  shared_min_sdf[threadIdx.x] = fminf(fabsf(sdf0), fabsf(sdf1));
  shared_max_inv_sigma2[threadIdx.x] = fmaxf(v0.inv_sigma2, v1.inv_sigma2);

  /// reducing operation
#pragma unroll 1
  for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {

    __syncthreads();
    if ((threadIdx.x  & (stride-1)) == (stride-1)) {
      shared_min_sdf[threadIdx.x] = fminf(shared_min_sdf[threadIdx.x-stride/2],
                                          shared_min_sdf[threadIdx.x]);
      shared_max_inv_sigma2[threadIdx.x] = fmaxf(shared_max_inv_sigma2[threadIdx.x-stride/2],
                                             shared_max_inv_sigma2[threadIdx.x]);
    }
  }
  __syncthreads();

  if (threadIdx.x == blockDim.x - 1) {
    float min_sdf = shared_min_sdf[threadIdx.x];
    float max_inv_sigma2 = shared_max_inv_sigma2[threadIdx.x];

    // TODO(wei): check this weird reference
    float t = geometry_helper.truncate_distance(5.0f);

    // TODO(wei): add || valid_triangles == 0 when memory leak is dealt with
    candidate_entries.flag(idx) =
        (min_sdf >= t || max_inv_sigma2 < EPSILON) ? (uchar)1 : (uchar)0;
  }
}


/// Collect dead voxels
__global__
void CollectLowSurfelBlocksKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    int processing_block_count
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= processing_block_count) {
    return;
  }
  const HashEntry& entry = candidate_entries[idx];

  candidate_entries.flag(idx) = 0;
  if (blocks[entry.ptr].inner_surfel_count < 10
      && blocks[entry.ptr].boundary_surfel_count == 0) {
    blocks[entry.ptr].life_count_down--;
  } else {
    blocks[entry.ptr].life_count_down = BLOCK_LIFE;
  }
  __syncthreads();

  const int3 offsets[6] = {
      {0,0,1},{0,0,-1},{0,1,0},{0,-1,0},{1,0,0},{-1,0,0}
  };
  if (blocks[entry.ptr].life_count_down <= 0) {
    for (int j = 0; j < 6; ++j) {
      HashEntry query_entry = hash_table.GetEntry(entry.pos + offsets[j]);
      if (query_entry.ptr == FREE_ENTRY) continue;
      if (blocks[query_entry.ptr].life_count_down != 0)
        return;
    }
    candidate_entries.flag(idx) = 1;
  }
}

/// !!! Their mesh not recycled
__global__
void RecycleGarbageTrianglesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh       mesh_m,
    HashTable  hash_table
) {
  const uint idx = blockIdx.x;
  if (candidate_entries.flag(idx) == 0) return;

  const HashEntry& entry = candidate_entries[idx];
  const uint local_idx = threadIdx.x;  //inside an SDF block
  Voxel &voxel = blocks[entry.ptr].voxels[local_idx];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[local_idx];

  for (int i = 0; i < N_TRIANGLE; ++i) {
    int triangle_ptr = mesh_unit.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) continue;

    // Clear ref_count of its pointed vertices
    mesh_m.ReleaseTriangle(mesh_m.triangle(triangle_ptr));
    mesh_m.triangle(triangle_ptr).Clear();
    mesh_m.FreeTriangle(triangle_ptr);
    mesh_unit.triangle_ptrs[i] = FREE_PTR;
  }
}

__global__
void RecycleGarbageVerticesKernel(
    EntryArray candidate_entries,
    BlockArray       blocks,
    Mesh             mesh_m,
    HashTable        hash_table,
    ScaleTable       scale_table
) {
  if (candidate_entries.flag(blockIdx.x) == 0) return;
  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  MeshUnit &cube = blocks[entry.ptr].mesh_units[local_idx];

  __shared__ int valid_vertex_count;
  if (threadIdx.x == 0) valid_vertex_count = 0;
  __syncthreads();

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (cube.vertex_ptrs[i] != FREE_PTR) {
      if (mesh_m.vertex(cube.vertex_ptrs[i]).ref_count <= 0) {
        mesh_m.vertex(cube.vertex_ptrs[i]).Clear();
        mesh_m.FreeVertex(cube.vertex_ptrs[i]);
        cube.vertex_ptrs[i] = FREE_PTR;
      }
      else {
        atomicAdd(&valid_vertex_count, 1);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && valid_vertex_count == 0) {
    int3 ancest = scale_table.GetAncestor(entry.pos);
    if (entry.pos.x==ancest.x && entry.pos.y==ancest.y && entry.pos.z==ancest.z)
      return;
    if (hash_table.FreeEntry(entry.pos)) {
      blocks[entry.ptr].Clear();
    }
  }
}

void StarveOccupiedBlockArray(
    EntryArray& candidate_entries,
    BlockArray& blocks
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  StarveOccupiedBlocksKernel<<<grid_size, block_size >>>(candidate_entries, blocks);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void CollectGarbageBlockArray(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    GeometryHelper& geometry_helper
) {
  const uint threads_per_block = BLOCK_SIZE / 2;

  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  CollectGarbageBlockArrayKernel <<<grid_size, block_size >>>(
      candidate_entries,
          blocks,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void CollectLowSurfelBlocks(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    HashTable& hash_table,
    GeometryHelper& geometry_helper
) {
  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const int threads_per_block = 64;
  const dim3 grid_size((processing_block_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  CollectLowSurfelBlocksKernel <<<grid_size, block_size >>>(
      candidate_entries,
          blocks,
          hash_table,
          geometry_helper,
          processing_block_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

// TODO(wei): Check vertex / triangles in detail
// including garbage collection
void RecycleGarbageBlockArray(
    EntryArray &candidate_entries,
    BlockArray& blocks,
    Mesh&      mesh_m,
    HashTable& hash_table,
    ScaleTable& scale_table
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  RecycleGarbageTrianglesKernel <<<grid_size, block_size >>>(
      candidate_entries, blocks, mesh_m, hash_table);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

    RecycleGarbageVerticesKernel <<<grid_size, block_size >>>(
        candidate_entries, blocks, mesh_m, hash_table, scale_table);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

