//
// Created by wei on 17-10-21.
//

#include "compact_mesh.h"
#include "helper_cuda.h"
////////////////////
/// class CompactMesh
////////////////////

/// Life cycle
//CompactMesh::~CompactMesh() {
//  Free();
//}

void CompactMesh::Alloc(const MeshParams &mesh_params) {
  if (! is_allocated_on_gpu_) {
    checkCudaErrors(cudaMalloc(&vertex_remapper_,
                               sizeof(int) * mesh_params.max_vertex_count));

    checkCudaErrors(cudaMalloc(&vertex_counter_,
                               sizeof(uint)));
    checkCudaErrors(cudaMalloc(&vertices_ref_count_,
                               sizeof(int) * mesh_params.max_vertex_count));
    checkCudaErrors(cudaMalloc(&vertices_,
                               sizeof(float3) * mesh_params.max_vertex_count));
    checkCudaErrors(cudaMalloc(&normals_,
                               sizeof(float3) * mesh_params.max_vertex_count));
    checkCudaErrors(cudaMalloc(&colors_,
                               sizeof(float3) * mesh_params.max_vertex_count));
    
    checkCudaErrors(cudaMalloc(&scales_,
                               sizeof(int) * mesh_params.max_triangle_count));
    checkCudaErrors(cudaMalloc(&triangle_counter_,
                               sizeof(uint)));
    checkCudaErrors(cudaMalloc(&triangles_ref_count_,
                               sizeof(int) * mesh_params.max_triangle_count));
    printf("%zu", sizeof(int3));
    checkCudaErrors(cudaMalloc(&triangles_,
                               sizeof(int3) * mesh_params.max_triangle_count));
    is_allocated_on_gpu_ = true;
  }
}

void CompactMesh::Free() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(vertex_remapper_));

    checkCudaErrors(cudaFree(vertex_counter_));
    checkCudaErrors(cudaFree(vertices_ref_count_));
    checkCudaErrors(cudaFree(vertices_));
    checkCudaErrors(cudaFree(normals_));
    checkCudaErrors(cudaFree(colors_));
    
    checkCudaErrors(cudaFree(scales_));
    checkCudaErrors(cudaFree(triangle_counter_));
    checkCudaErrors(cudaFree(triangles_ref_count_));
    checkCudaErrors(cudaFree(triangles_));
  }
}

void CompactMesh::Resize(const MeshParams &mesh_params) {
  mesh_params_ = mesh_params;
  if (is_allocated_on_gpu_) {
    Free();
  }
  Alloc(mesh_params);
  Reset();
}

/// Reset
void CompactMesh::Reset() {
  printf("vertex_counter_:%d\n",mesh_params_.max_vertex_count);
  checkCudaErrors(cudaMemset(vertex_remapper_, 0xff,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(vertices_ref_count_, 0,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(vertex_counter_,
                             0, sizeof(uint)));
  checkCudaErrors(cudaMemset(triangles_ref_count_, 0,
                             sizeof(int) * mesh_params_.max_triangle_count));
  checkCudaErrors(cudaMemset(triangle_counter_,
                             0, sizeof(uint)));
}

uint CompactMesh::vertex_count() {
  uint compact_vertex_count;
  checkCudaErrors(cudaMemcpy(&compact_vertex_count,
                             vertex_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_vertex_count;
}

uint CompactMesh::triangle_count() {
  uint compact_triangle_count;
  checkCudaErrors(cudaMemcpy(&compact_triangle_count,
                             triangle_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_triangle_count;
}


__global__
void FindKNeighborRemeshingKernel(
      Point* pc,
      uint pc_size,
      uint* candidate_points,
      uint candidate_number,
      Point* pc_node,
      uint existed_number,
      float3* knn,
      int k_num) {

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
  // printf("x:%d pc_size:%d candi:%d\n",x,pc_size,candidate_number);
  for(int i=0;i<pc_size;i++){
      if(i!=candidate_points[x]){
        // printf("i:%d candi:%d sum:%d\n",i,candidate_points[x],pc_size);
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


void CompactMesh::remeshing(PointCloud& pc){
  const uint threads_per_block = 8;
  uint candidate_number = pc.count();
  uint* candidate_points_cpu = new uint[candidate_number];

  //use all points as remeshing points for now.
  for(int i=0;i<candidate_number;i++)
    candidate_points_cpu[i] = i;

  //2.find neighbors of remeshing points.
  //add all exist vertexs to pc_node
  Point* pc_node_; 
  uint* candidate_points;
  uint compact_vertex_count = vertex_count();
  uint compact_triangle_count = triangle_count();
  printf("Vertices:%d\n",compact_vertex_count);
  printf("Triangles:%d\n",compact_triangle_count);
  printf("pc_num:%d\n",candidate_number);
  // LOG(INFO) << "Vertices: " << compact_vertex_count;
  // LOG(INFO) << "Triangles: " << compact_triangle_count;

  float3* vertices = new float3[compact_vertex_count];
  int3* triangles  = new int3  [compact_triangle_count];

  checkCudaErrors(cudaMemcpy(vertices, vertices_,
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(triangles, triangles_,
                             sizeof(int3) * compact_triangle_count,
                             cudaMemcpyDeviceToHost));

  const uint existed_number = compact_vertex_count;  //TODO: change to candidate remeshing points
  Point* pc_node_cpu = new Point[existed_number];
  checkCudaErrors(cudaMalloc(&pc_node_, sizeof(Point) * existed_number));
  checkCudaErrors(cudaMalloc(&candidate_points, sizeof(uint) * candidate_number));
  
  for(int i=0;i<existed_number;i++){
    pc_node_cpu[i].x = vertices[i].x;
    pc_node_cpu[i].y = vertices[i].y;
    pc_node_cpu[i].z = vertices[i].z;
  }

  checkCudaErrors(cudaMemcpy(pc_node_, pc_node_cpu,
                             sizeof(Point) * existed_number,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(candidate_points, candidate_points_cpu,
                             sizeof(uint) * candidate_number,
                             cudaMemcpyHostToDevice));
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
      pc.GetGPUPtr(),
      pc.count(),
      candidate_points,
      candidate_number,
      pc_node_,
      existed_number,
      knn,
      k_num
  );

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  
  // //3.project neighbors to remeshing point's normal plane.
  // ProjectNeighborKernel <<<grid_size, block_size>>>(
  //   pc_gpu.GetGPUPtr(),
  //   pc_gpu.count(),
  //   candidate_points,
  //   candidate_number,
  //   knn,
  //   k_num,
  //   geometry_helper
  //   );
}
