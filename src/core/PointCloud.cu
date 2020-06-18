#include "PointCloud.h"
#include "helper_cuda.h"

#include <device_launch_parameters.h>

__global__
void BlockArrayResetKernel(
	Point* points,
    int pc_size
){
	const uint block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx < pc_size) {
      points[block_idx].Clear();
    }
}

__host__
void PointCloud::Alloc(uint pc_size){
	if (! is_allocated_on_gpu_) {
	    pc_size_ = pc_size;
	    // printf("block:%d %d\n",sizeof(Block),block_count);
	    checkCudaErrors(cudaMalloc(&points_, sizeof(Point) * pc_size));
	    is_allocated_on_gpu_ = true;
	}
}

__host__
void PointCloud::Free(){
	if (is_allocated_on_gpu_) {
	  checkCudaErrors(cudaFree(points_));
	  pc_size_ = 0;
	  points_ = NULL;
	  is_allocated_on_gpu_ = false;
	}
}

__host__
void PointCloud::Resize(uint pc_size){
	if (is_allocated_on_gpu_) {
	    Free();
	  }
	Alloc(pc_size);
	Reset();
}

__host__
void PointCloud::TransferPtToGPU(Point* pc)
{
	if(is_allocated_on_gpu_){
		checkCudaErrors(cudaMemcpy(points_, pc, sizeof(Point)*pc_size_, cudaMemcpyHostToDevice));
	}
}

__host__
void PointCloud::TransferPtToHost(Point* pc){
	if(is_allocated_on_gpu_){
		checkCudaErrors(cudaMemcpy(pc, points_, sizeof(Point)*pc_size_, cudaMemcpyDeviceToHost));
	}
}


__host__
void PointCloud::Reset() {
	const uint threads_per_block = 64;

  if (pc_size_ == 0) return;

  // NOTE: this block is the parallel unit in CUDA, not the data structure Block
  const uint blocks = (pc_size_ + threads_per_block - 1) / threads_per_block;

  const dim3 grid_size(blocks, 1);
  const dim3 block_size(threads_per_block, 1);

  BlockArrayResetKernel<<<grid_size, block_size>>>(points_, pc_size_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

// __host__ 
// void PointCloud::GenerateKdTree() {
// 	kdroot_ = NULL;  
// 	int id = 0;
//     kdroot_ = build_kdtree(points_, id, kdroot_); 
// 	return;
// }

// __host__
// Point PointCloud::FindNearestByKdTree(Point target){
// 	Point nearestpoint;  
//     float distance;  

//     searchNearest(kdroot_, target, nearestpoint, distance);  

//     return nearestpoint;
//     // cout<<"The nearest distance is "<<distance<<",and the nearest point is "<<nearestpoint.x<<","<<nearestpoint.y<<endl;  
//     // cout <<"Enter search point"<<endl;  

// }

// __host__
// void FreeKdTreeNode(Tnode* node){
//     if(node == NULL)
//     	return;
//     if(node->left != NULL)
//     	FreeKdTreeNode(node->left);
//     if(node->right != NULL)
//     	FreeKdTreeNode(node->right);
//     delete node;
// }

// __host__
// void PointCloud::FreeKdTree(){
// 	FreeKdTreeNode(kdroot_);
// 	return;
// }