#include "meshing/remeshing.h"

__global__
void AddExistedCubetoAllocKernel(
  EntryArray candidate_entries_,
  uint size,
  Point* pc
){
  uint idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  pc[idx].x = candidate_entries_[idx].pos.x;
  pc[idx].y = candidate_entries_[idx].pos.y;
  pc[idx].z = candidate_entries_[idx].pos.z;
  return;
}

__global__
void FindKNeighborRemeshingKernel(
      ScaleTable scale_table,
      HashTable hash_table,
      Point* pc,
      uint pc_size,
      uint* candidate_points,
      uint candidate_number,
      Point* pc_node,
      uint existed_number,
      float3* knn,
      int k_num,
      GeometryHelper geometry_helper) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= candidate_number)
    return;
  //fill knn[x] - knn[x+k_num-1]
  int* nearest_index = new int[k_num];
  float* nearest_dis = new float[k_num];
  for(int i=0;i<k_num;i++){
    nearest_index[i] = -1;
    nearest_dis[i] = 999999;
  }
  for(int i=0;i<pc_size;i++){
      if(i!=candidate_points[x]){
        float dis = sqrt((pc[i].x-pc[candidate_points[x]].x)*(pc[i].x-pc[candidate_points[x]].x)
                        +(pc[i].y-pc[candidate_points[x]].y)*(pc[i].y-pc[candidate_points[x]].y)
                        +(pc[i].z-pc[candidate_points[x]].z)*(pc[i].z-pc[candidate_points[x]].z));
        int insert_position = 0;
        for(int j=k_num-1;j>=0;j--){
          if(dis>nearest_dis[j]){
            insert_position = j+1;
            break;
          }
        }
        if(insert_position<k_num){
          for(int j=insert_position+1;j<k_num;j++){
            knn[x+j] = knn[x+j-1];
            nearest_dis[j] = nearest_dis[j-1];
            nearest_index[j] = nearest_index[j-1];
          }
          knn[x+insert_position] = make_float3(pc[nearest_index[insert_position]].x,
                                             pc[nearest_index[insert_position]].y,
                                             pc[nearest_index[insert_position]].z);
          nearest_dis[insert_position] = dis;
          nearest_index[insert_position] = i;
        }
      }
  }


  for(int i=0;i<existed_number;i++){
    float dis = sqrt((pc_node[i].x-pc[candidate_points[x]].x)*(pc_node[i].x-pc[candidate_points[x]].x)
                      +(pc_node[i].y-pc[candidate_points[x]].y)*(pc_node[i].y-pc[candidate_points[x]].y)
                      +(pc_node[i].z-pc[candidate_points[x]].z)*(pc_node[i].z-pc[candidate_points[x]].z));
      int insert_position = 0;
      for(int j=k_num-1;j>=0;j--){
        if(dis>nearest_dis[j]){
          insert_position = j+1;
          break;
        }
      }
      if(insert_position<k_num){
        for(int j=insert_position+1;j<k_num;j++){
          knn[x+j] = knn[x+j-1];
          nearest_dis[j] = nearest_dis[j-1];
          nearest_index[j] = nearest_index[j-1];
        }
        knn[x+insert_position] = make_float3(pc_node[nearest_index[insert_position]].x,
                                             pc_node[nearest_index[insert_position]].y,
                                             pc_node[nearest_index[insert_position]].z);
        nearest_dis[insert_position] = dis;
        nearest_index[insert_position] = i;
      }
  }
  delete[] nearest_index;
  delete[] nearest_dis;

}

__global__
void ProjectNeighborKernel(
	Point* pc,
	uint pc_size,
  uint* candidate_points,
  uint candidate_number,
	float3* knn,
	int k_num,
	GeometryHelper geometry_helper
){
	uint idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx > candidate_number)
    return;
  
  Point this_point = pc[candidate_points[idx]];
  printf("points:%f %f %f-%f %f %f\n",this_point.x,this_point.y,this_point.z,this_point.normal_x,this_point.normal_y,this_point.normal_z);
  //project to a plane
}

void Remeshing(
	ScaleTable& scale_table,
    HashTable& hash_table,
    PointCloud& pc_gpu,
    EntryArray& candidate_entries_,
    GeometryHelper& geometry_helper
){
  //1.choose candidate remeshing point
  //candidate points : int[]
  //TODO: after step 1, replace the pc_gpu with the candidate remeshing points.
  const uint threads_per_block = 8;
  uint candidate_number = pc_gpu.count();
  uint* candidate_points = new uint[candidate_number];

  //use all points as remeshing points for now.
  for(int i=0;i<candidate_number;i++)
    candidate_points[i] = i;

  //2.find neighbors of remeshing points.
  //add all exist vertexs to pc_node
	Point* pc_node_; 
	const uint existed_number = candidate_entries_.count();  //TODO: change to candidate remeshing points
	checkCudaErrors(cudaMalloc(&pc_node_, sizeof(Point) * existed_number));

  if(existed_number > 0){
    const dim3 grid_size1((existed_number + threads_per_block - 1)
                         /threads_per_block,
                         1);
    const dim3 block_size1(threads_per_block, 1);
  
    AddExistedCubetoAllocKernel<<<grid_size1, block_size1>>>(
      candidate_entries_,
      existed_number,
      pc_node_
    );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  //knn
	const uint k_num = 5;
  float3* knn;
  //near to far : for point i, the position of jth neighbor is ( i*k_num + j )
  checkCudaErrors(cudaMalloc(&knn, sizeof(float3) * candidate_number * k_num));

	const dim3 grid_size((candidate_number + threads_per_block - 1)
                       /threads_per_block,
                       1);
  const dim3 block_size(threads_per_block, 1);

	FindKNeighborRemeshingKernel<<<grid_size, block_size>>>(
	    scale_table,
	    hash_table,
	    pc_gpu.GetGPUPtr(),
	    pc_gpu.count(),
      candidate_points,
      candidate_number,
	    pc_node_,
	    existed_number,
	    knn,
	    k_num,
	    geometry_helper
	  );

	//3.project neighbors to remeshing point's normal plane.
	ProjectNeighborKernel <<<grid_size, block_size>>>(
		pc_gpu.GetGPUPtr(),
		pc_gpu.count(),
    candidate_points,
    candidate_number,
		knn,
		k_num,
		geometry_helper
	  );

}

