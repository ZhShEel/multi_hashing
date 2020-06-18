//
// Created by wei on 17-10-22.
//
#include <util/timer.h>
#include "mapping/allocate.h"

// __global__
// void AllocBlockArrayKernel(HashTable   hash_table,
//                            SensorData  sensor_data,
//                            SensorParams sensor_params,
//                            float4x4     w_T_c,
//                            GeometryHelper geometry_helper) {

//   const uint x = blockIdx.x * blockDim.x + threadIdx.x;
//   const uint y = blockIdx.y * blockDim.y + threadIdx.y;

//   if (x >= sensor_params.width || y >= sensor_params.height)
//     return;

//   /// TODO(wei): change it here
//   /// 1. Get observed data
//   float depth = tex2D<float>(sensor_data.depth_texture, x, y);
//   if (depth == MINF || depth == 0.0f
//       || depth >= geometry_helper.sdf_upper_bound)
//     return;

//   float truncation = geometry_helper.truncate_distance(depth);
//   float near_depth = fminf(geometry_helper.sdf_upper_bound, depth - truncation);
//   float far_depth = fminf(geometry_helper.sdf_upper_bound, depth + truncation);
//   if (near_depth >= far_depth) return;

//   float3 camera_pos_near = geometry_helper.ImageReprojectToCamera(x, y, near_depth,
//                                                             sensor_params.fx, sensor_params.fy,
//                                                             sensor_params.cx, sensor_params.cy);
//   float3 camera_pos_far  = geometry_helper.ImageReprojectToCamera(x, y, far_depth,
//                                                             sensor_params.fx, sensor_params.fy,
//                                                             sensor_params.cx, sensor_params.cy);

//   /// 2. Set range where blocks are allocated
//   float3 world_pos_near  = w_T_c * camera_pos_near;
//   float3 world_pos_far   = w_T_c * camera_pos_far;
//   float3 world_ray_dir = normalize(world_pos_far - world_pos_near);

//   //uniform resolution
  
//   int3 block_pos_near = geometry_helper.WorldToBlock(world_pos_near);
//   int3 block_pos_far  = geometry_helper.WorldToBlock(world_pos_far);
//   //now
//   // float voxel_size = geometry_helper.voxel_size;//emmm
//   // printf("size:%f\n",voxel_size);
//   // int3 block_pos_near = geometry_helper.WorldToBlock(world_pos_near, voxel_size);
//   // int3 block_pos_far  = geometry_helper.WorldToBlock(world_pos_far, voxel_size);

//   float3 block_step = make_float3(sign(world_ray_dir));

//   /// 3. Init zig-zag steps
//   float3 world_pos_nearest_voxel_center
//       = geometry_helper.BlockToWorld(block_pos_near + make_int3(clamp(block_step, 0.0, 1.0f)))
//         - 0.5f * geometry_helper.voxel_size;
//   float3 t = (world_pos_nearest_voxel_center - world_pos_near) / world_ray_dir;
//   float3 dt = (block_step * BLOCK_SIDE_LENGTH * geometry_helper.voxel_size) / world_ray_dir;
//   int3 block_pos_bound = make_int3(make_float3(block_pos_far) + block_step);

//   if (world_ray_dir.x == 0.0f) {
//     t.x = PINF;
//     dt.x = PINF;
//   }
//   if (world_ray_dir.y == 0.0f) {
//     t.y = PINF;
//     dt.y = PINF;
//   }
//   if (world_ray_dir.z == 0.0f) {
//     t.z = PINF;
//     dt.z = PINF;
//   }

//   int3 block_pos_curr = block_pos_near;
//   /// 4. Go a zig-zag path to ensure all voxels are visited
//   const uint kMaxIterTime = 1024;
// #pragma unroll 1
//   for (uint iter = 0; iter < kMaxIterTime; ++iter) {
//     if (geometry_helper.IsBlockInCameraFrustum(
//         w_T_c.getInverse(),
//         block_pos_curr,
//         sensor_params)) {
//       /// Disable streaming at current
//       // && !isSDFBlockStreamedOut(idCurrentVoxel, hash_table, is_streamed_mask)) {
//       hash_table.AllocEntry(block_pos_curr);
//     }

//     // Traverse voxel grid
//     if (t.x < t.y && t.x < t.z) {
//       block_pos_curr.x += block_step.x;
//       if (block_pos_curr.x == block_pos_bound.x) return;
//       t.x += dt.x;
//     } else if (t.y < t.z) {
//       block_pos_curr.y += block_step.y;
//       if (block_pos_curr.y == block_pos_bound.y) return;
//       t.y += dt.y;
//     } else {
//       block_pos_curr.z += block_step.z;
//       if (block_pos_curr.z == block_pos_bound.z) return;
//       t.z += dt.z;
//     }
//   }
// }
__global__
void FindKNeighborKernel(
      ScaleTable scale_table,
      HashTable hash_table,
      Point* pc,
      uint pc_size,
      Point* pc_node,
      uint existed_number,
      float3* knn,
      int k_num,
      GeometryHelper geometry_helper) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= pc_size)
    return;
  //fill knn[x] - knn[x+k_num-1]
  int* nearest_index = new int[k_num];
  float* nearest_dis = new float[k_num];
  for(int i=0;i<k_num;i++){
    nearest_index[i] = -1;
    nearest_dis[i] = 999999;
  }
  for(int i=0;i<pc_size;i++){
      if(i!=x){
        float dis = sqrt((pc[i].x-pc[x].x)*(pc[i].x-pc[x].x)
                        +(pc[i].y-pc[x].y)*(pc[i].y-pc[x].y)
                        +(pc[i].z-pc[x].z)*(pc[i].z-pc[x].z));
        int insert_position = 0;
        for(int j=k_num-1;j>=0;j--){
          if(dis>nearest_dis[j]){
            insert_position = j+1;
            break;
          }
        }
        if(insert_position<k_num){
          for(int j=k_num-1;j>insert_position;j--){
            knn[x+j] = knn[x+j-1];
            nearest_dis[j] = nearest_dis[j-1];
            nearest_index[j] = nearest_index[j-1];
          }
          // for(int j=insert_position+1;j<k_num;j++){
          //   knn[x+j] = knn[x+j-1];
          //   nearest_dis[j] = nearest_dis[j-1];
          //   nearest_index[j] = nearest_index[j-1];
          // }
          // knn[x+insert_position] = make_float3(pc[nearest_index[insert_position]].x,
          //                                    pc[nearest_index[insert_position]].y,
          //                                    pc[nearest_index[insert_position]].z);
          knn[x+insert_position] = make_float3(pc[i].x,pc[i].y,pc[i].z);
          nearest_dis[insert_position] = dis;
          nearest_index[insert_position] = i;
        }
      }
  }
  // for(int i=0;i<k_num;i++)
  //   knn[x+i] = make_float3(pc[nearest_index[i]].x, pc[nearest_index[i]].y, pc[nearest_index[i]].z);

  //existed node

  for(int i=0;i<existed_number;i++){
    float dis = sqrt((pc_node[i].x-pc[x].x)*(pc_node[i].x-pc[x].x)
                      +(pc_node[i].y-pc[x].y)*(pc_node[i].y-pc[x].y)
                      +(pc_node[i].z-pc[x].z)*(pc_node[i].z-pc[x].z));
      int insert_position = 0;
      for(int j=k_num-1;j>=0;j--){
        if(dis>nearest_dis[j]){
          insert_position = j+1;
          break;
        }
      }
      if(insert_position<k_num){
        for(int j=k_num-1;j>insert_position;j--){
          knn[x+j] = knn[x+j-1];
          nearest_dis[j] = nearest_dis[j-1];
          nearest_index[j] = nearest_index[j-1];
        }
        knn[x+insert_position] = make_float3(pc[i].x,pc[i].y,pc[i].z);
        nearest_dis[insert_position] = dis;
        nearest_index[insert_position] = i;
      }
  }
  delete[] nearest_index;
  delete[] nearest_dis;

}

__device__
int MaxAbs3(int a,int b,int c){
  int ans;
  a = abs(a); b = abs(b); c = abs(c);
  ans = a;
  if(b > ans)
    ans = b;
  if(c > ans)
    ans = c;
  return ans;
}

__device__
void CheckWrongVoxel(
    HashTable hash_table,
    BlockArray blocks
    ){
        for(int y=11;y<=19;y++){
            int x=23, z=0;
            HashEntry entry = hash_table.GetEntry(make_int3(x,y,z));
            if(entry.ptr>=0){
                Block block = blocks[entry.ptr];
                for(int j=0;j<50;j++){
                    block.voxels[j].Check(entry.pos,entry.ptr,j);
                }
            }
            else
                printf("FREE PTR:%d %d %d ptr:%d\n",x,y,z,entry.ptr);
        }
    }
__global__
void c(HashTable hash_table, BlockArray blocks){ CheckWrongVoxel(hash_table, blocks); }


__global__
void AllocBlockArrayKernel(
      ScaleTable scale_table,
      HashTable hash_table,
      BlockArray blocks,
      Point* pc,
      uint pc_size,
      Point* pc_node,
      uint existed_number,
      float3* knn,
      int k_num,
      int step_,
      GeometryHelper geometry_helper) {

  const uint x = blockIdx.x;

  if (x >= pc_size)
    return;
  
  uint step_length = step_;
  uint3 offset_o = geometry_helper.DevectorizeIndex(threadIdx.x);

  int3 offset = make_int3(offset_o) - (step_length-1)/2;

  // if(pc[x].x>bbx_max.x||pc[x].x<bbx_min.x||pc[x].y>bbx_max.y||
  //    pc[x].y<bbx_min.y||pc[x].z>bbx_max.z||pc[x].z<bbx_min.z)
  //   return;
  // for(int i=0;i<k_num;i++){
  //   printf("neighbor:%d:%f %f %f of %d:%f %f %f\n",knn[x+i],pc[knn[x+i]].x,pc[knn[x+i]].y,pc[knn[x+i]].z,
  //     x,pc[x].x,pc[x].y,pc[x].z);
  // }
  // printf("size:%d/%d: %f %f %f\n",x,pc_size,pc[x].x,pc[x].y,pc[x].z);
  float3 world_point_pos = make_float3(pc[x].x,pc[x].y,pc[x].z);

   /* adaptive alloc
  for(int i=0;i<k_num;i++){
    // printf("x+i:%d\n",x+i);
    // printf("point:%f %f %f\n",knn[x+i].x,knn[x+i].y,knn[x+i].z);
    float3 end_point_pos = knn[x+i];
    float3 travel_ray = end_point_pos - world_point_pos;
    
    //TODO:can change
    // float3 step = make_float3(geometry_helper.voxel_size,geometry_helper.voxel_size,geometry_helper.voxel_size);
    float3 step_num = travel_ray/geometry_helper.voxel_size;
    // printf("see:%f %f %f\n",step_num.x,step_num.y,step_num.z);
    int step_Num = MaxAbs3((int)step_num.x,(int)step_num.y,(int)step_num.z);
    float3 step = travel_ray / step_Num;
    for(int j=0;j<step_Num;j++){
      if(j>10)
        break;
      float3 new_foot_pos = world_point_pos + step * j;

      int3 block_pos_finest = geometry_helper.WorldToBlock(new_foot_pos);
      int curr_scale = scale_table.GetScale(block_pos_finest).scale; //block_position
      int3 block_pos_near = geometry_helper.WorldToBlock(new_foot_pos, curr_scale * geometry_helper.voxel_size) * curr_scale;

      //int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos);

      int3 block_pos_curr = block_pos_near;
      // printf("pos:%d %d %d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z);
      scale_table.AllocScale(block_pos_finest);
      hash_table.AllocEntry(block_pos_curr);

    }

  }
  */
  // printf("offset:%d %d %d %d\n",offset.x,offset.y,offset.z,step_length);
  float3 world_point_pos_curr = world_point_pos + make_float3(offset.x,offset.y,offset.z) * geometry_helper.voxel_size * step_;
  float voxel_size = geometry_helper.voxel_size;
  int3 block_pos_finest = geometry_helper.WorldToBlock(world_point_pos_curr);

  int curr_scale = scale_table.GetScale(block_pos_finest).scale; //block_position
  //int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos_curr, curr_scale * voxel_size) * curr_scale;
  int3 block_pos_near = scale_table.GetAncestor(block_pos_finest);
  // if(curr_scale>1)
  //   printf("origin:%d %d %d, ancestor:%d %d %d\n", block_pos_finest.x,block_pos_finest.y,block_pos_finest.z, block_pos_near.x, block_pos_near.y,block_pos_near.z);

  //int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos);
  /*
  if(block_pos_finest.x==23&&block_pos_finest.y>=11&&block_pos_finest.y<=19&&block_pos_finest.z==0){
    int cu_ptr = hash_table.GetEntry(block_pos_finest).ptr;
    printf("block_pos_finest:%d %d %d scale:%d ptr:%d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z,curr_scale,cu_ptr);
    if(cu_ptr>0){
        for(int ii=0;ii<30;ii++)
          printf("this_ptr now:%d -> sdf:%f\n",cu_ptr,blocks[cu_ptr].voxels[ii].sdf);
     }
  }
  */
  
  int3 block_pos_curr = block_pos_near;
  if(curr_scale<=0){
      hash_table.AllocEntry(block_pos_finest);
  }
  // printf("pos:%d %d %d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z);
 //printf("alloc entry:%d %d %d  scale:%d %d %d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z,block_pos_curr.x,block_pos_curr.y,block_pos_curr.z);

  scale_table.AllocScale(block_pos_finest);
  //int3 aa = scale_table.GetAncestor(block_pos_finest);
 // printf("alloc:%d %d %d->%d %d %d(scale:%d)\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z,aa.x,aa.y,aa.z,curr_scale);
 // hash_table.AllocEntry(block_pos_curr);

   //if(scale_table.GetScale(block_pos_finest).scale==-1)
   // scale_table.SetScale(block_pos_finest,1);
  //if(scale_table.GetScale(block_pos_finest).scale==-1)
 //   printf("fin:%d %d %d->curr:%d %d %d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z,block_pos_curr.x,block_pos_curr.y,block_pos_curr.z);
//int curr_2_scale = scale_table.GetScale(block_pos_curr).scale;
  //if(curr_2_scale!=1){
   // printf("scale:%d %d\n",curr_2_scale, hash_table.GetEntry(block_pos_curr).ptr);
  //}
  // if(curr_scale!=1)
    // printf("scale:%d(%d %d %d)\n",curr_scale, block_pos_curr.x,block_pos_curr.y,block_pos_curr.z);

}

__global__
void AllocScaleBlockKernel(
    HashTable hash_table,
    ScaleTable scale_table,
    EntryArray candidate_entries_
    ){
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        HashEntry& entry = candidate_entries_[idx];
/*
        int3 aa = scale_table.GetAncestor(entry.pos);
        if(aa.x!=entry.pos.x||aa.y!=entry.pos.y||aa.z!=entry.pos.z)
          printf("WRONG ANCESTOR:%d %d %d->%d %d %d scale:%d->%d\n",aa.x,aa.y,aa.z,entry.pos.x,entry.pos.y,entry.pos.z,scale_table.GetScale(aa).scale,scale_table.GetScale(entry.pos).scale);
*/
        scale_table.AllocScale(entry.pos);
        return;
    }

// __global__
// void AllocBlockArrayKernel(ScaleTable  scale_table,
//                            HashTable   hash_table,
//                            pcl::gpu::PtrSz< pcl::PointXYZRGB > pc,
//                            uint pc_size,
//                            float3 bbx_min,
//                            float3 bbx_max,
//                            GeometryHelper geometry_helper) {
//   //TODO:
//   const uint x = blockIdx.x * blockDim.x + threadIdx.x;

//   if (x >= pc_size)
//     return;
//   // if(pc[x].x>bbx_max.x||pc[x].x<bbx_min.x||pc[x].y>bbx_max.y||
//   //    pc[x].y<bbx_min.y||pc[x].z>bbx_max.z||pc[x].z<bbx_min.z)
//   //   return;

//   float3 world_point_pos = make_float3(pc[x].x,pc[x].y,pc[x].z);
//   /// 2. Set range where blocks are allocated

//   float voxel_size = geometry_helper.voxel_size;
//   int3 block_pos_finest = geometry_helper.WorldToBlock(world_point_pos);

//   //????? no 
//   int curr_scale = scale_table.GetScale(block_pos_finest).scale; //block_position
//   int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos, curr_scale * voxel_size) * curr_scale;

//   //int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos);

//   int3 block_pos_curr = block_pos_near;
//   // printf("pos:%d %d %d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z);
//   scale_table.AllocScale(block_pos_finest);
//   hash_table.AllocEntry(block_pos_curr);

//   // int step = 0;
//   // float voxel_size_ = 5;
//   // for(float i=pc[x].x-step*voxel_size_;i<=pc[x].x+step*voxel_size_;i+=voxel_size_){
//   //   for(float j=pc[x].y-step*voxel_size_;j<=pc[x].y+step*voxel_size_;j+=voxel_size_){
//   //     for(float k=pc[x].z-step*voxel_size_;k<=pc[x].z+step*voxel_size_;k+=voxel_size_){
//   //       float3 world_point_pos = make_float3(i,j,k);
//   //       int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos);

//   //       int3 block_pos_curr = block_pos_near;
        
//   //       hash_table.AllocEntry(block_pos_curr);
//   //     }
//   //   }
//   // }

// }

// double AllocBlockArray(
//     HashTable& hash_table,
//     Sensor& sensor,
//     GeometryHelper& geometry_helper
// ) {
//   Timer timer;
//   timer.Tick();
//   hash_table.ResetMutexes();

//   const uint threads_per_block = 8;
//   const dim3 grid_size((sensor.sensor_params().width + threads_per_block - 1)
//                        /threads_per_block,
//                        (sensor.sensor_params().height + threads_per_block - 1)
//                        /threads_per_block);
//   const dim3 block_size(threads_per_block, threads_per_block);

//   AllocBlockArrayKernel<<<grid_size, block_size>>>(
//       hash_table,
//       sensor.data(),
//       sensor.sensor_params(), sensor.wTc(),
//       geometry_helper);
//   checkCudaErrors(cudaDeviceSynchronize());
//   checkCudaErrors(cudaGetLastError());
//   return timer.Tock();
// }

__global__
void AddExistedCubeToAllocKernel(
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
void AllocFilteredBlockArrayKernel(
      ScaleTable scale_table,
      HashTable hash_table,
      Point* pc,
      uint pc_size,
      Point* pc_node,
      uint existed_number,
      float3* knn,
      int k_num,
      int step_,
      GeometryHelper geometry_helper) {

  const uint x = blockIdx.x;

  if (x >= pc_size)
    return;
  
  uint step_length = step_;
  int x1 = threadIdx.x % step_length;
  int y1 = (threadIdx.x % (step_length * step_length)) / step_length;
  int z1 = threadIdx.x / (step_length * step_length);
  int3 offset =  make_int3(x1, y1, z1) - (step_length-1)/2;


  float3 world_point_pos = make_float3(pc[x].x,pc[x].y,pc[x].z);
  // printf("offset:%d %d %d %d\n",offset.x,offset.y,offset.z,step_length);
  float3 world_point_pos_curr = world_point_pos + make_float3(offset.x,offset.y,offset.z) * geometry_helper.voxel_size * step_;
  float voxel_size = geometry_helper.voxel_size;
  int3 block_pos_finest = geometry_helper.WorldToBlock(world_point_pos_curr);

  int curr_scale = scale_table.GetScale(block_pos_finest).scale; //block_position
  int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos_curr, pow(2,curr_scale-1) * voxel_size) * pow(2,curr_scale-1);

  //int3 block_pos_near = geometry_helper.WorldToBlock(world_point_pos);

  int3 block_pos_curr = block_pos_near;
  // printf("pos:%d %d %d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z);

  //add a filtered (adaptive)
  // if(){
    //printf("alloc entry:%d %d %d  scale:%d %d %d\n",block_pos_finest.x,block_pos_finest.y,block_pos_finest.z,block_pos_curr.x,block_pos_curr.y,block_pos_curr.z);
    scale_table.AllocScale(block_pos_finest);
    hash_table.AllocEntry(block_pos_curr);
  // }
}


double AllocBlockArray(
  ScaleTable& scale_table,
  HashTable& hash_table,
  BlockArray& blocks,
  PointCloud& pc_gpu,
  EntryArray& candidate_entries_,
  GeometryHelper& geometry_helper
) {
  Timer timer;
  timer.Tick();
  scale_table.ResetMutexes();
  hash_table.ResetMutexes();
 
  const uint threads_per_block = 32;
  const dim3 grid_size((pc_gpu.count() + threads_per_block - 1)
                       /threads_per_block,
                       1);
  const dim3 block_size(threads_per_block, 1);

  std::cout<<"bbx:"<<pc_gpu.bbx_min.x<<" "<<pc_gpu.bbx_min.y<<" "<<pc_gpu.bbx_min.z<<" "
           <<"->"  <<pc_gpu.bbx_max.x<<" "<<pc_gpu.bbx_max.y<<" "<<pc_gpu.bbx_max.z<<std::endl;

  //add the existed meshes of last frames to here.
  //for now, just add the corner point to pc_gpu directly.

  const uint existed_number = candidate_entries_.count();

  //pc_node:existed blocks' position used as new input points.
  Point* pc_node;
  checkCudaErrors(cudaMalloc(&pc_node, sizeof(Point) * existed_number));

  if(existed_number > 0){
    const dim3 grid_size1((existed_number + threads_per_block - 1)
                         /threads_per_block,
                         1);
    const dim3 block_size1(threads_per_block, 1);
  
    AddExistedCubeToAllocKernel<<<grid_size1, block_size1>>>(
      candidate_entries_,
      existed_number,
      pc_node
    );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
 
  const uint k_num = 5;
  float3* knn;
  //near to far : for point i, the position of jth neighbor is ( i*k_num + j )
  checkCudaErrors(cudaMalloc(&knn, sizeof(float3) * pc_gpu.count() * k_num));
  FindKNeighborKernel<<<grid_size, block_size>>>(
    scale_table,
    hash_table,
    pc_gpu.GetGPUPtr(),
    pc_gpu.count(),
    pc_node,
    existed_number,
    knn,
    k_num,
    geometry_helper
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  // uint step_ = (geometry_helper.approximate_size / geometry_helper.voxel_size)*2+1;
  uint step_ = 1;
  const dim3 grid_size2(pc_gpu.count(),
                       1);
  const dim3 block_size2(step_*step_*step_, 1);


  AllocBlockArrayKernel<<<grid_size2, block_size2>>>(
      scale_table,
      hash_table,
      blocks,
      pc_gpu.GetGPUPtr(),
      pc_gpu.count(),
      pc_node,
      existed_number,
      knn,
      k_num,
      step_,
      geometry_helper);
 // c<<<dim3(1,1),dim3(1,1)>>>(hash_table,blocks);
  // AllocFilteredBlockArrayKernel<<<grid_size2, block_size2>>>(
  //     scale_table,
  //     hash_table,
  //     pc_gpu.GetGPUPtr(),
  //     pc_gpu.count(),
  //     pc_node,
  //     existed_number,
  //     knn,
  //     k_num,
  //     step_,
  //     geometry_helper);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError()); 
  checkCudaErrors(cudaFree(pc_node));
  checkCudaErrors(cudaFree(knn));
  // checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaGetLastError());
  return timer.Tock();
}

double AllocScaleBlock(
    HashTable& hash_table,
    ScaleTable& scale_table,
    BlockArray& block,
    EntryArray& candidate_entries_
    ){
        Timer timer;
        timer.Tick();
        scale_table.ResetMutexes();
        uint entry_count = candidate_entries_.count();
        if(entry_count<=0)
            return timer.Tock();

        const int threads_per_block = 8;
        const dim3 grid_size((entry_count+threads_per_block-1)/threads_per_block,1);
        const dim3 block_size(threads_per_block,1);

        AllocScaleBlockKernel<<<grid_size, block_size>>>(
            hash_table,
            scale_table,
            candidate_entries_
        );
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        return timer.Tock();
    }


// double AllocBlockArray(
//   ScaleTable& scale_table,
//   HashTable& hash_table,
//   pcl::PointCloud<pcl::PointXYZRGB> pc_,
//   pcl::gpu::DeviceArray< pcl::PointXYZRGB >& pc,
//   GeometryHelper& geometry_helper
// ) {
//   Timer timer;
//   timer.Tick();
//   scale_table.ResetMutexes();
//   hash_table.ResetMutexes();

//   const uint threads_per_block = 8;
//   const dim3 grid_size((pc.size() + threads_per_block - 1)
//                        /threads_per_block,
//                        1);
//   const dim3 block_size(threads_per_block, 1);
//   float3 bbx_min = make_float3(999999,999999,999999);
//   float3 bbx_max = make_float3(-999999,-999999,-999999);
//   for(int bb=0;bb<pc.size();bb++){
//       if(pc_[bb].x<bbx_min.x)
//           bbx_min.x = pc_[bb].x;
//       if(pc_[bb].y<bbx_min.y)
//           bbx_min.y = pc_[bb].y;
//       if(pc_[bb].z<bbx_min.z)
//           bbx_min.z = pc_[bb].z;
//       if(pc_[bb].x>bbx_min.x)
//           bbx_max.x = pc_[bb].x;
//       if(pc_[bb].y>bbx_min.y)
//           bbx_max.y = pc_[bb].y;
//       if(pc_[bb].z>bbx_min.z)
//           bbx_max.z = pc_[bb].z;
//   }
//   std::cout<<"bbx:"<<bbx_min.x<<" "<<bbx_min.y<<" "<<bbx_min.z<<" "
//            <<"->"  <<bbx_max.x<<" "<<bbx_max.y<<" "<<bbx_max.z<<std::endl;
//   AllocBlockArrayKernel<<<grid_size, block_size>>>(
//       scale_table,
//       hash_table,
//       pc,
//       pc.size(),
//       bbx_min,
//       bbx_max,
//       geometry_helper);
//   checkCudaErrors(cudaDeviceSynchronize());
//   checkCudaErrors(cudaGetLastError());
//   return timer.Tock();
// }
