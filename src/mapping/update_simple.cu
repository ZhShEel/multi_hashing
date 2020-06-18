#include <device_launch_parameters.h>
#include <util/timer.h>

#include "core/block_array.h"
#include "mapping/update_simple.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/spatial_query.h"

////////////////////
/// Device code
////////////////////

__global__
void PrepareKnnQueryDataKernel(
  EntryArray candidate_entries,
  int entry_size,
  int dim,
  float* query,
  BlockArray blocks,
  GeometryHelper geometry_helper
  ){

  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);
  
  uint offset = (blockIdx.x * blockDim.x + threadIdx.x) * dim;
  // printf("see:%d %d(%f %f %f)\n",blockIdx.x * blockDim.x + threadIdx.x,offset,
  //   world_pos.x,world_pos.y,world_pos.z);
  query[offset] = world_pos.x;
  query[offset + 1] = world_pos.y;
  query[offset + 2] = world_pos.z;

  return;
}

__global__
void PrepareKnnRefDataKernel(
  Point* pc,
  int pc_size,
  int dim,
  float* ref
  ){
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx > pc_size)
    return;
  Point p = pc[idx];
  uint offset = idx * dim;
  ref[offset] = p.x;
  ref[offset + 1] = p.y;
  ref[offset + 2] = p.z;
  return;
}
/*
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
*/
__global__
void UpdateBlocksSimpleKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Point* pc,
    uint pc_size,
    HashTable hash_table,
    ScaleTable scale_table,
    // float* knn_dist,
    // int* knn_index,
    float4x4 cTw,
    float voxel_size,
    GeometryHelper geometry_helper
){
  
  /// 1. Set view
  float3 from_point = make_float3(cTw.m14,cTw.m24,cTw.m34);
  //from ------- pc[pointIdxNKNSearch] ---------world_pos

  // printf("from point :%f %f %f\n",from_point.x,from_point.y,from_point.z);
  /// 2. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int curr_scale = scale_table.GetScale(entry.pos).scale;
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx)) * pow(2,curr_scale-1);
/*
    if(threadIdx.x==0){
        int3 aa = scale_table.GetAncestor(entry.pos);
        if(entry.pos.x!=aa.x||entry.pos.y!=aa.y||entry.pos.z!=aa.z)
        printf("WRONG ANCESTOR:%d %d %d->%d %d %d\n",entry.pos.x,entry.pos.y,entry.pos.z,aa.x,aa.y,aa.z);
    }
*/
  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  int pointIdxNKNSearch = 0;
  float pointNKNSquaedDistance;

  float minnum = 999999;

  //TODO: this cost 0.1s, cut it
  for(int j=0;j<pc_size;j++){
    // float curr_cos = (world_pos.x * pc[j].x + world_pos.y * pc[j].y + world_pos.z * pc[j].z)/
    // (sqrt(world_pos.x*world_pos.x+world_pos.y*world_pos.y+world_pos.z*world_pos.z) * sqrt(pc[j].x*pc[j].x+pc[j].y*pc[j].y+pc[j].z*pc[j].z));
    // float curr_view2voxel = sqrt((world_pos.x-from_point.x)*(world_pos.x-from_point.x)+(world_pos.y-from_point.y)*(world_pos.y-from_point.y)+(world_pos.z-from_point.z)*(world_pos.z-from_point.z));
    // float curr_view2pc = sqrt((pc[j].x-from_point.x)*(pc[j].x-from_point.x)+(pc[j].y-from_point.y)*(pc[j].y-from_point.y)+(pc[j].z-from_point.z)*(pc[j].z-from_point.z));
    // float cd = curr_cos * curr_view2voxel - curr_view2pc;

float cd = sqrt((world_pos.x-pc[j].x)*(world_pos.x-pc[j].x)+(world_pos.y-pc[j].y)*(world_pos.y-pc[j].y)
      +(world_pos.z-pc[j].z)*(world_pos.z-pc[j].z));
    if(abs(cd) < minnum){
      minnum = abs(cd);
      pointIdxNKNSearch = j;
      pointNKNSquaedDistance = abs(cd);
    }
  }
  // printf("sdf:%f %d\n",pointNKNSquaedDistance, pointIdxNKNSearch);
  // knn methods
  // float cd = knn_dist[blockIdx.x * blockDim.x + threadIdx.x];
  // pointIdxNKNSearch = knn_index[blockIdx.x * blockDim.x + threadIdx.x];
  // pointNKNSquaedDistance = cd;
  // printf("see:%d :%f\n", pointIdxNKNSearch, pointNKNSquaedDistance);

  float trun_param = 1.5;
  float trun_dis = trun_param * scale_table.GetScale(entry.pos).scale * voxel_size;


  float sdf = sqrt((world_pos.x-pc[pointIdxNKNSearch].x)*(world_pos.x-pc[pointIdxNKNSearch].x) + 
      (world_pos.y-pc[pointIdxNKNSearch].y) * (world_pos.y-pc[pointIdxNKNSearch].y) + 
      (world_pos.z-pc[pointIdxNKNSearch].z) * (world_pos.z-pc[pointIdxNKNSearch].z));
    // printf("sdf:%f from(%f %f %f)and(%f %f %f) trun:%f\n",sdf,world_pos.x,world_pos.y,world_pos.z,pc[pointIdxNKNSearch].x,pc[pointIdxNKNSearch].y,pc[pointIdxNKNSearch].z, 
   // trun_dis);
  // printf("trun:%f\n",trun_dis);
  
  //use camera pos
  // float cos = (world_pos.x * pc[pointIdxNKNSearch].x + world_pos.y * pc[pointIdxNKNSearch].y + world_pos.z * pc[pointIdxNKNSearch].z)/
  //   (sqrt(world_pos.x*world_pos.x+world_pos.y*world_pos.y+world_pos.z*world_pos.z) * sqrt(pc[pointIdxNKNSearch].x*pc[pointIdxNKNSearch].x+pc[pointIdxNKNSearch].y*pc[pointIdxNKNSearch].y+pc[pointIdxNKNSearch].z*pc[pointIdxNKNSearch].z));
  // float view2voxel = sqrt((world_pos.x-from_point.x)*(world_pos.x-from_point.x)+(world_pos.y-from_point.y)*(world_pos.y-from_point.y)+(world_pos.z-from_point.z)*(world_pos.z-from_point.z));
  // float view2pc = sqrt((pc[pointIdxNKNSearch].x-from_point.x)*(pc[pointIdxNKNSearch].x-from_point.x)+(pc[pointIdxNKNSearch].y-from_point.y)*(pc[pointIdxNKNSearch].y-from_point.y)+(pc[pointIdxNKNSearch].z-from_point.z)*(pc[pointIdxNKNSearch].z-from_point.z));
  // // printf("from:%f %f %f left:%f(%f %f %f) right:%f(%f %f %f) sdf:%f\n", from_point.x,from_point.y,from_point.z,
  // //   view2voxel*cos, world_pos.x,world_pos.y,world_pos.z,view2pc,
  // //   pc[pointIdxNKNSearch].x,pc[pointIdxNKNSearch].y,pc[pointIdxNKNSearch].z,sdf);
  // if(view2voxel*cos > view2pc)
  //   sdf = -sdf;

  // use normal
  float cosineX = (world_pos.x-pc[pointIdxNKNSearch].x)*(pc[pointIdxNKNSearch].normal_x) + 
      (world_pos.y-pc[pointIdxNKNSearch].y)*(pc[pointIdxNKNSearch].normal_y) + 
    (world_pos.z-pc[pointIdxNKNSearch].z)*(pc[pointIdxNKNSearch].normal_z);
  float cosineY = sdf*sqrt(pc[pointIdxNKNSearch].normal_x*pc[pointIdxNKNSearch].normal_x
    + pc[pointIdxNKNSearch].normal_y*pc[pointIdxNKNSearch].normal_y +
    pc[pointIdxNKNSearch].normal_z*pc[pointIdxNKNSearch].normal_z);
  float cosine = cosineX/cosineY;

  sdf = sdf * cosine;
  if(sdf > trun_dis)
    sdf = trun_dis;
  if(sdf < -trun_dis)
    sdf = -trun_dis;
/*
  if(scale_table.GetScale(entry.pos).scale>=4&&voxel_pos.x==0&&voxel_pos.y==-136&&voxel_pos.z==8){
      printf("before scale:%d sdf:%f inv:%f pos:%d %d %d->%d %d %d\n",scale_table.GetScale(entry.pos).scale, sdf, this_voxel.inv_sigma2,entry.pos.x, entry.pos.y,entry.pos.z, voxel_pos.x, voxel_pos.y, voxel_pos.z);
  }
  */
  if(abs(sdf)>=trun_dis)
    return;
  // if(cosine<0)
  //   sdf = -sdf;
  // printf("sdf:%f cosine:%f trun:%f\n",sdf,cosine,trun_dis);

  // this_voxel.sdf = (this_voxel.sdf * this_voxel.inv_sigma2 + sdf * float(1.0)) / (this_voxel.inv_sigma2 + float(1.0));
  // this_voxel.inv_sigma2 = fminf(this_voxel.inv_sigma2 + float(1.0), 255.0f);

  uchar inv_sigma2 = 1;

  Voxel delta;
  delta.sdf = sdf;
  delta.inv_sigma2 = float(inv_sigma2);
  
  delta.color = make_uchar3(0, 255, 0);
  // this_voxel.sdf = (this_voxel.sdf * this_voxel.inv_sigma2 + delta.sdf * delta.inv_sigma2) / (this_voxelinv_sigma2 + delta.inv_sigma2);  //this sentence cost 0.1s???????
  float prev_sdf = this_voxel.sdf;
  
  this_voxel.Update(delta);
  
  //if(entry.pos.x==23&&entry.pos.y==9&&entry.pos.z==0)
  //if(this_voxel.inv_sigma2>=2&&prev_sdf<=0&&this_voxel.sdf>0)
    // printf("sdf:%f+%f->%f(%f) pos:%d %d %d->%d %d %d\n",prev_sdf,sdf,this_voxel.sdf,this_voxel.inv_sigma2,entry.pos.x,entry.pos.y,entry.pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z);
  //if(entry.pos.z==0){
  
  
  //if(scale_table.GetScale(entry.pos).scale>2&&blockIdx.x<50){
  //if(this_voxel.inv_sigma2>1&&voxel_pos.z==0&&sdf<0){
  
 /*
  //if(scale_table.GetScale(entry.pos).scale>=3){
  //if(entry.pos.x>7&&entry.pos.y>7&&entry.pos.z==0&&sdf-0.02*voxel_pos.z>0.001){
  if(scale_table.GetScale(entry.pos).scale>=4&&voxel_pos.x==0&&voxel_pos.y==-136&&voxel_pos.z==8){
 
  //if(entry.pos.x==23 && entry.pos.z == 0&&entry.pos.y>=11&&entry.pos.y<=19&&threadIdx.x<=50){
  float3 block_world_pos = geometry_helper.VoxelToWorld(voxel_base_pos);
  float3 block_end_world_pos = block_world_pos + make_float3(geometry_helper.voxel_size*8);
      uint3 oo = geometry_helper.DevectorizeIndex(threadIdx.x);
  printf("scale:%d sdf:%f->%f pos:%d %d %d(%d %d %d)[%d %d %d] normal:%f %f %f weight:%f(%f %f %f)-(%f %f %f) = %f, sign:%f %f %f block:(%f %f %f)%d->(%f %f %f)%d ptr:%d\n",
    scale_table.GetScale(entry.pos).scale,prev_sdf,sdf, entry.pos.x, entry.pos.y, entry.pos.z, voxel_pos.x, voxel_pos.y, voxel_pos.z, oo.x,oo.y,oo.z,
    pc[pointIdxNKNSearch].normal_x,pc[pointIdxNKNSearch].normal_y,pc[pointIdxNKNSearch].normal_z,
    this_voxel.inv_sigma2, world_pos.x, world_pos.y, world_pos.z, 
    pc[pointIdxNKNSearch].x, pc[pointIdxNKNSearch].y, pc[pointIdxNKNSearch].z, pointNKNSquaedDistance, 
    cosineX, cosineY, cosine,
    block_world_pos.x,block_world_pos.y,block_world_pos.z,blockIdx.x,
    block_end_world_pos.x,block_end_world_pos.y,block_end_world_pos.z,threadIdx.x, entry.ptr);
  }
*/
}

__global__
void UpdateShellsSimpleKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Point* pc,
    uint pc_size,
    HashTable hash_table,
    ScaleTable scale_table,
    // float* knn_dist,
    // int* knn_index,
    float4x4 cTw,
    float voxel_size,
    GeometryHelper geometry_helper
){
  
  /// 1. Set view
  float3 from_point = make_float3(cTw.m14,cTw.m24,cTw.m34);
  //from ------- pc[pointIdxNKNSearch] ---------world_pos

  // printf("from point :%f %f %f\n",from_point.x,from_point.y,from_point.z);
  /// 2. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int curr_scale = scale_table.GetScale(entry.pos).scale;
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeShellIndex(local_idx)) * pow(2,curr_scale-1);

  Voxel &this_voxel = blocks[entry.ptr].shells[local_idx];
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  int pointIdxNKNSearch = 0;
  float pointNKNSquaedDistance;

  float minnum = 999999;

  //TODO: this cost 0.1s, cut it
  for(int j=0;j<pc_size;j++){
    // float curr_cos = (world_pos.x * pc[j].x + world_pos.y * pc[j].y + world_pos.z * pc[j].z)/
    // (sqrt(world_pos.x*world_pos.x+world_pos.y*world_pos.y+world_pos.z*world_pos.z) * sqrt(pc[j].x*pc[j].x+pc[j].y*pc[j].y+pc[j].z*pc[j].z));
    // float curr_view2voxel = sqrt((world_pos.x-from_point.x)*(world_pos.x-from_point.x)+(world_pos.y-from_point.y)*(world_pos.y-from_point.y)+(world_pos.z-from_point.z)*(world_pos.z-from_point.z));
    // float curr_view2pc = sqrt((pc[j].x-from_point.x)*(pc[j].x-from_point.x)+(pc[j].y-from_point.y)*(pc[j].y-from_point.y)+(pc[j].z-from_point.z)*(pc[j].z-from_point.z));
    // float cd = curr_cos * curr_view2voxel - curr_view2pc;

  float cd = sqrt((world_pos.x-pc[j].x)*(world_pos.x-pc[j].x)+(world_pos.y-pc[j].y)*(world_pos.y-pc[j].y)
      +(world_pos.z-pc[j].z)*(world_pos.z-pc[j].z));
    if(abs(cd) < minnum){
      minnum = abs(cd);
      pointIdxNKNSearch = j;
      pointNKNSquaedDistance = abs(cd);
    }
  }
  // printf("sdf:%f %d\n",pointNKNSquaedDistance, pointIdxNKNSearch);
  // knn methods
  // float cd = knn_dist[blockIdx.x * blockDim.x + threadIdx.x];
  // pointIdxNKNSearch = knn_index[blockIdx.x * blockDim.x + threadIdx.x];
  // pointNKNSquaedDistance = cd;
  // printf("see:%d :%f\n", pointIdxNKNSearch, pointNKNSquaedDistance);

  float trun_param = 3;
  float trun_dis = trun_param * scale_table.GetScale(entry.pos).scale * voxel_size;


  float sdf = sqrt((world_pos.x-pc[pointIdxNKNSearch].x)*(world_pos.x-pc[pointIdxNKNSearch].x) + 
      (world_pos.y-pc[pointIdxNKNSearch].y) * (world_pos.y-pc[pointIdxNKNSearch].y) + 
      (world_pos.z-pc[pointIdxNKNSearch].z) * (world_pos.z-pc[pointIdxNKNSearch].z));
    // printf("sdf:%f from(%f %f %f)and(%f %f %f) trun:%f\n",sdf,world_pos.x,world_pos.y,world_pos.z,pc[pointIdxNKNSearch].x,pc[pointIdxNKNSearch].y,pc[pointIdxNKNSearch].z, 
   // trun_dis);
  // printf("trun:%f\n",trun_dis);
  
  //use camera pos
  // float cos = (world_pos.x * pc[pointIdxNKNSearch].x + world_pos.y * pc[pointIdxNKNSearch].y + world_pos.z * pc[pointIdxNKNSearch].z)/
  //   (sqrt(world_pos.x*world_pos.x+world_pos.y*world_pos.y+world_pos.z*world_pos.z) * sqrt(pc[pointIdxNKNSearch].x*pc[pointIdxNKNSearch].x+pc[pointIdxNKNSearch].y*pc[pointIdxNKNSearch].y+pc[pointIdxNKNSearch].z*pc[pointIdxNKNSearch].z));
  // float view2voxel = sqrt((world_pos.x-from_point.x)*(world_pos.x-from_point.x)+(world_pos.y-from_point.y)*(world_pos.y-from_point.y)+(world_pos.z-from_point.z)*(world_pos.z-from_point.z));
  // float view2pc = sqrt((pc[pointIdxNKNSearch].x-from_point.x)*(pc[pointIdxNKNSearch].x-from_point.x)+(pc[pointIdxNKNSearch].y-from_point.y)*(pc[pointIdxNKNSearch].y-from_point.y)+(pc[pointIdxNKNSearch].z-from_point.z)*(pc[pointIdxNKNSearch].z-from_point.z));
  // // printf("from:%f %f %f left:%f(%f %f %f) right:%f(%f %f %f) sdf:%f\n", from_point.x,from_point.y,from_point.z,
  // //   view2voxel*cos, world_pos.x,world_pos.y,world_pos.z,view2pc,
  // //   pc[pointIdxNKNSearch].x,pc[pointIdxNKNSearch].y,pc[pointIdxNKNSearch].z,sdf);
  // if(view2voxel*cos > view2pc)
  //   sdf = -sdf;

  // use normal
  float cosineX = (world_pos.x-pc[pointIdxNKNSearch].x)*(pc[pointIdxNKNSearch].normal_x) + 
      (world_pos.y-pc[pointIdxNKNSearch].y)*(pc[pointIdxNKNSearch].normal_y) + 
    (world_pos.z-pc[pointIdxNKNSearch].z)*(pc[pointIdxNKNSearch].normal_z);
  float cosineY = sdf*sqrt(pc[pointIdxNKNSearch].normal_x*pc[pointIdxNKNSearch].normal_x
    + pc[pointIdxNKNSearch].normal_y*pc[pointIdxNKNSearch].normal_y +
    pc[pointIdxNKNSearch].normal_z*pc[pointIdxNKNSearch].normal_z);
  float cosine = cosineX/cosineY;

  sdf = sdf * cosine;
  if(sdf > trun_dis)
    sdf = trun_dis;
  if(sdf < -trun_dis)
    sdf = -trun_dis;
  if(abs(sdf)>=trun_dis)
    return;
  // if(cosine<0)
  //   sdf = -sdf;
  // printf("sdf:%f cosine:%f trun:%f\n",sdf,cosine,trun_dis);

  // this_voxel.sdf = (this_voxel.sdf * this_voxel.inv_sigma2 + sdf * float(1.0)) / (this_voxel.inv_sigma2 + float(1.0));
  // this_voxel.inv_sigma2 = fminf(this_voxel.inv_sigma2 + float(1.0), 255.0f);

  uchar inv_sigma2 = 1;

  Voxel delta;
  delta.sdf = sdf;
  delta.inv_sigma2 = float(inv_sigma2);
  
  delta.color = make_uchar3(0, 255, 0);
  // this_voxel.sdf = (this_voxel.sdf * this_voxel.inv_sigma2 + delta.sdf * delta.inv_sigma2) / (this_voxelinv_sigma2 + delta.inv_sigma2);  //this sentence cost 0.1s???????
  float prev_sdf = this_voxel.sdf;
  
  this_voxel.Update(delta);
  
  //if(entry.pos.x==23&&entry.pos.y==9&&entry.pos.z==0)
  //if(this_voxel.inv_sigma2>=2&&prev_sdf<=0&&this_voxel.sdf>0)
    // printf("sdf:%f+%f->%f(%f) pos:%d %d %d->%d %d %d\n",prev_sdf,sdf,this_voxel.sdf,this_voxel.inv_sigma2,entry.pos.x,entry.pos.y,entry.pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z);
  //if(entry.pos.z==0){
  
  
  //if(scale_table.GetScale(entry.pos).scale>2&&blockIdx.x<50){
  //if(this_voxel.inv_sigma2>1&&voxel_pos.z==0&&sdf<0){
  
 /*
  if(scale_table.GetScale(entry.pos).scale>=4&&voxel_pos.x==0&&voxel_pos.y==-136&&voxel_pos.z==8){
  //if(entry.pos.x>7&&entry.pos.y>7&&entry.pos.z==0&&sdf-0.02*voxel_pos.z>0.001){
 
 // if(entry.pos.x==23 && entry.pos.z == 0&&entry.pos.y>=11&&entry.pos.y<=19&&threadIdx.x<=50){
  float3 block_world_pos = geometry_helper.VoxelToWorld(voxel_base_pos);
  float3 block_end_world_pos = block_world_pos + make_float3(geometry_helper.voxel_size*8);
  printf("scale:%d sdf:%f->%f pos:%d %d %d(%d %d %d) normal:%f %f %f weight:%f(%f %f %f)-(%f %f %f) = %f, sign:%f %f %f block:(%f %f %f)%d->(%f %f %f)%d ptr:%d\n",
    scale_table.GetScale(entry.pos).scale,prev_sdf,sdf, entry.pos.x, entry.pos.y, entry.pos.z, voxel_pos.x, voxel_pos.y, voxel_pos.z,
    pc[pointIdxNKNSearch].normal_x,pc[pointIdxNKNSearch].normal_y,pc[pointIdxNKNSearch].normal_z,
    this_voxel.inv_sigma2, world_pos.x, world_pos.y, world_pos.z, 
    pc[pointIdxNKNSearch].x, pc[pointIdxNKNSearch].y, pc[pointIdxNKNSearch].z, pointNKNSquaedDistance, 
    cosineX, cosineY, cosine,
    block_world_pos.x,block_world_pos.y,block_world_pos.z,blockIdx.x,
    block_end_world_pos.x,block_end_world_pos.y,block_end_world_pos.z,threadIdx.x, entry.ptr);
  }
*/

}



__global__
void checkSDFValueKernel(
    EntryArray &candidate_entries,
    BlockArray &blocks

    ){
        uint idx = blockIdx.x;
        HashEntry &entry = candidate_entries[idx];
        Block b = blocks[entry.ptr];
    }
/*
double UpdateBlocksSimple(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    pcl::PointCloud<pcl::PointXYZRGB> pc_,
    pcl::gpu::DeviceArray< pcl::PointXYZRGB >& pc,
    pcl::gpu::DeviceArray< pcl::Normal >& normals,
    HashTable &hash_table,
    ScaleTable& scale_table,
    float voxel_size,
    GeometryHelper &geometry_helper
) {

  Timer timer;
  timer.Tick();
  const uint threads_per_block = BLOCK_SIZE;

  uint candidate_entry_count = candidate_entries.count();
  // LOG(INFO)<<"return?:"<<candidate_entries.count();
  if (candidate_entry_count <= 0)
    return timer.Tock();

  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  float3 bbx_min = make_float3(999999,999999,999999);
  float3 bbx_max = make_float3(-999999,-999999,-999999);
  for(int bb=0;bb<pc.size();bb++){
      if(pc_[bb].x<bbx_min.x)
          bbx_min.x = pc_[bb].x;
      if(pc_[bb].y<bbx_min.y)
          bbx_min.y = pc_[bb].y;
      if(pc_[bb].z<bbx_min.z)
          bbx_min.z = pc_[bb].z;
      if(pc_[bb].x>bbx_min.x)
          bbx_max.x = pc_[bb].x;
      if(pc_[bb].y>bbx_min.y)
          bbx_max.y = pc_[bb].y;
      if(pc_[bb].z>bbx_min.z)
          bbx_max.z = pc_[bb].z;
  }
  timer.Tick();
  UpdateBlocksSimpleKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          pc,
          normals,
          pc.size(),
          bbx_min,
          bbx_max,
          hash_table,
          scale_table,
          voxel_size,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

    //checkSDFValueKernel<<<grid_size, block_size>>>(
     //   candidate_entries,

    //)
  return timer.Tock();
}
*/
double UpdateBlocksSimple(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    PointCloud& pc_gpu,
    HashTable &hash_table,
    ScaleTable& scale_table,
    float4x4& cTw,
    float voxel_size,
    GeometryHelper &geometry_helper
) {

  Timer timer;
  timer.Tick();
  // const uint threads_per_block = 32;
  const uint threads_per_block = BLOCK_SIZE; //8*8*8 = 512 (divide to 16 part)

  uint candidate_entry_count = candidate_entries.count();
  // LOG(INFO)<<"return?:"<<candidate_entries.count();
  if (candidate_entry_count <= 0)
    return timer.Tock();

  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  timer.Tick();

  /**
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 * Using cuBLAS, the computation of the distance matrix can be faster in some cases than other
 * implementations despite being more complex.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
// bool knn_cublas(const float * ref,
//                 int           ref_nb,
//                 const float * query,
//                 int           query_nb,
//                 int           dim,
//                 int           k,
//                 float *       knn_dist,
//                 int *         knn_index);


  // const int ref_nb = pc_gpu.count();
  // const int query_nb = candidate_entries.count();
  // const int dim = 3;
  // const int k = 1;
  // // Allocate input points and output k-NN distances / indexes
  // float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
  // float * query      = (float*) malloc(query_nb * threads_per_block * dim * sizeof(float));
  // float * knn_dist   = (float*) malloc(query_nb * threads_per_block * k   * sizeof(float));
  // int   * knn_index  = (int*)   malloc(query_nb * threads_per_block * k   * sizeof(int));

  // float* ref_gpu;  float* query_gpu; 
  // cudaMalloc(&ref_gpu, ref_nb   * dim * sizeof(float));
  // cudaMalloc(&query_gpu, query_nb * threads_per_block * dim * sizeof(float));
  // const dim3 grid_size_ref(ref_nb,1);
  // const dim3 block_size_ref(1,1);

  // const dim3 grid_size_query(query_nb, 1);
  // const dim3 block_size_query(threads_per_block,1);

  // PrepareKnnRefDataKernel << < grid_size_ref, block_size_ref >> >(
  //   pc_gpu.GetGPUPtr(),
  //   pc_gpu.count(),
  //   dim,
  //   ref_gpu
  //   );

  // PrepareKnnQueryDataKernel << < grid_size_query, block_size_query >> >(
  //   candidate_entries,
  //   candidate_entries.count(),
  //   dim,
  //   query_gpu,
  //   blocks,
  //   geometry_helper
  //   );

  // cudaMemcpy(ref, ref_gpu,
  //   sizeof(float) * ref_nb * dim, cudaMemcpyDeviceToHost);
  // cudaMemcpy(query, query_gpu,
  //   sizeof(float) * query_nb * threads_per_block * dim, cudaMemcpyDeviceToHost);

  // bool knnresult = knn_cuda_texture(
  //         ref,
  //         ref_nb,
  //         query,
  //         query_nb * threads_per_block,
  //         dim,
  //         k,
  //         knn_dist,
  //         knn_index);
  
  
  // float* knn_dist_gpu;  int* knn_index_gpu; 
  // cudaMalloc(&knn_dist_gpu, query_nb * threads_per_block * k   * sizeof(float));
  // cudaMalloc(&knn_index_gpu, query_nb * threads_per_block * k   * sizeof(int));
  // cudaMemcpy(knn_dist_gpu, knn_dist, query_nb * threads_per_block * k   * sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(knn_index_gpu, knn_index, query_nb * threads_per_block * k   * sizeof(int), cudaMemcpyHostToDevice);
  //end of knn-cuda methods


  UpdateBlocksSimpleKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          pc_gpu.GetGPUPtr(),
          pc_gpu.count(),
          hash_table,
          scale_table,
          // knn_dist_gpu,
          // knn_index_gpu,
          cTw,
          voxel_size,
          geometry_helper);

  // cudaFree(ref_gpu);
  // cudaFree(query_gpu);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  const dim3 block_size2(SHELL_SIZE,1);

  UpdateShellsSimpleKernel << < grid_size, block_size2>>> (
       candidate_entries,
          blocks,
          pc_gpu.GetGPUPtr(),
          pc_gpu.count(),
          hash_table,
          scale_table,
          // knn_dist_gpu,
          // knn_index_gpu,
          cTw,
          voxel_size,
          geometry_helper);

  return timer.Tock();
}
