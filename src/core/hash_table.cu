#include <unordered_set>
#include <device_launch_parameters.h>

#include "core/hash_table.h"

////////////////////
/// Device code
////////////////////
__global__
void HashTableResetBucketMutexesKernel(
    int *bucket_mutexes,
    uint bucket_count
) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < bucket_count) {
    bucket_mutexes[idx] = FREE_ENTRY;
  }
}

__global__
void HashTableResetNeighborMutexesKernel(
    HashEntry *entry,
    uint bucket_count
) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < bucket_count) {
    entry[idx].mutex = 0;
  }
}


__global__
void HashTableResetHeapKernel(
    uint *heap,
    uint value_capacity
) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < value_capacity) {
    heap[idx] = value_capacity - idx - 1;
  }
}

__global__
void HashTableResetEntriesKernel(
    HashEntry *entries,
    uint entry_count
) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < entry_count) {
    entries[idx].Clear();
  }
}

__global__
void HashTableResetEntriesKernel(
     ScaleInfo *scales,
     uint entry_count
){
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < entry_count) {
    scales[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////
HashTable::HashTable(const HashParams &params) {
  Alloc(params);
  Reset();
}

//HashTable::~HashTable() {
//  Free();
//}

void HashTable::Alloc(const HashParams &params) {
  if (!is_allocated_on_gpu_) {
    /// Parameters
    bucket_count = params.bucket_count;
    bucket_size = params.bucket_size;
    entry_count = params.entry_count;
    value_capacity = params.value_capacity;
    linked_list_size = params.linked_list_size;

    /// Values
    checkCudaErrors(cudaMalloc(&heap_,
                               sizeof(uint) * params.value_capacity));
    checkCudaErrors(cudaMalloc(&heap_counter_,
                               sizeof(uint)));

    /// Entries
    checkCudaErrors(cudaMalloc(&entries_,
                               sizeof(HashEntry) * params.entry_count));

    /// Mutexes
    checkCudaErrors(cudaMalloc(&bucket_mutexes_,
                               sizeof(int) * params.bucket_count));
    is_allocated_on_gpu_ = true;
  }
}

void HashTable::Free() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(heap_));
    checkCudaErrors(cudaFree(heap_counter_));

    checkCudaErrors(cudaFree(entries_));
    checkCudaErrors(cudaFree(bucket_mutexes_));

    is_allocated_on_gpu_ = false;
  }
}

void HashTable::Resize(const HashParams &params) {
  Alloc(params);
  Reset();
}
/// Reset
void HashTable::Reset() {
  /// Reset mutexes
  ResetMutexes();

  {
    /// Reset entries
    const int threads_per_block = 64;
    const dim3 grid_size((entry_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetEntriesKernel <<<grid_size, block_size>>>(entries_, entry_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    /// Reset allocated memory
    uint heap_counter_init = value_capacity - 1;
    checkCudaErrors(cudaMemcpy(heap_counter_, &heap_counter_init,
                               sizeof(uint),
                               cudaMemcpyHostToDevice));

    const int threads_per_block = 64;
    const dim3 grid_size((value_capacity + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetHeapKernel <<<grid_size, block_size>>>(heap_, value_capacity);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

void HashTable::ResetMutexes() {
  const int threads_per_block = 64;
  const dim3 grid_size((bucket_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);
  HashTableResetBucketMutexesKernel <<<grid_size, block_size>>>(bucket_mutexes_, bucket_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

///////////////////
/// scale table ///
///////////////////
ScaleTable::ScaleTable(const ScaleParams &params) {
  Alloc(params);
  Reset();
}

//HashTable::~HashTable() {
//  Free();
//}

void ScaleTable::Alloc(const ScaleParams &params) {
  if (!is_allocated_on_gpu_) {
    /// Parameters
    bucket_count = params.bucket_count;
    bucket_size = params.bucket_size;
    entry_count = params.entry_count;
    value_capacity = params.value_capacity;
    linked_list_size = params.linked_list_size;
    curvature_th = params.curvature_th;
    cd_th = params.cd_th;
    printf("params:%d %d %d\n",bucket_count,bucket_size,entry_count);
    /// Values
    checkCudaErrors(cudaMalloc(&heap_,
                               sizeof(uint) * params.value_capacity));
    checkCudaErrors(cudaMalloc(&heap_counter_,
                               sizeof(uint)));

    /// Entries
    checkCudaErrors(cudaMalloc(&scale_,
                               sizeof(ScaleInfo) * params.entry_count));

    /// Mutexes
    checkCudaErrors(cudaMalloc(&bucket_mutexes_,
                               sizeof(int) * params.bucket_count));
    is_allocated_on_gpu_ = true;
  }
}

void ScaleTable::Free() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(heap_));
    checkCudaErrors(cudaFree(heap_counter_));

    checkCudaErrors(cudaFree(scale_));
    checkCudaErrors(cudaFree(bucket_mutexes_));

    is_allocated_on_gpu_ = false;
  }
}

void ScaleTable::Resize(const ScaleParams &params) {
  Alloc(params);
  Reset();
}
/// Reset
void ScaleTable::Reset() {
  /// Reset mutexes
  printf("start reset mutex\n");
  ResetMutexes();

  {
    /// Reset entries
    const int threads_per_block = 64;
    const dim3 grid_size((entry_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetEntriesKernel <<<grid_size, block_size>>>(scale_, entry_count); //here
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    /// Reset allocated memory
    uint heap_counter_init = value_capacity - 1;
    checkCudaErrors(cudaMemcpy(heap_counter_, &heap_counter_init,
                               sizeof(uint),
                               cudaMemcpyHostToDevice));

    const int threads_per_block = 64;
    const dim3 grid_size((value_capacity + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetHeapKernel <<<grid_size, block_size>>>(heap_, value_capacity);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

void ScaleTable::ResetMutexes() {
  const int threads_per_block = 64;
  const dim3 grid_size((bucket_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);
 // printf("see:%d\n", bucket_count);
  HashTableResetBucketMutexesKernel <<<grid_size, block_size>>>(bucket_mutexes_, bucket_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void HashTable::ResetNeighborMutexes() {
  const int threads_per_block = 64;
  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);
  //printf("see:%d\n", bucket_count);
  HashTableResetNeighborMutexesKernel <<<grid_size, block_size>>>(entries_, entry_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

