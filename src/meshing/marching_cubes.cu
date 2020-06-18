#include <device_launch_parameters.h>
#include "meshing/marching_cubes.h"
#include "geometry/spatial_query.h"
#include "visualization/color_util.h"
//#define REDUCTION

////////////////////
/// class MappingEngine - meshing
////////////////////

////////////////////
/// Device code
////////////////////

/// Marching Cubes
__device__
float3  VertexIntersection(const float3 &p1, const float3 p2,
                          const float &v1, const float &v2,
                          const float &isolevel) {
  if (fabs(v1 - isolevel) < 0.008) return p1;
  if (fabs(v2 - isolevel) < 0.008) return p2;
  float mu = (isolevel - v1) / (v2 - v1);
  
  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}


__device__
inline bool IsInner(uint3 offset) {
  return (offset.x >= 1 && offset.y >= 1 && offset.z >= 1
          && offset.x < BLOCK_SIDE_LENGTH - 1
          && offset.y < BLOCK_SIDE_LENGTH - 1
          && offset.z < BLOCK_SIDE_LENGTH - 1);
}


__device__
inline int AllocateVertexWithMutex(
    MeshUnit &mesh_unit,
    uint  &vertex_idx,
    const float3 &vertex_pos,
    Mesh &mesh_m,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool color_type,
    bool enable_bayesian,
    bool enable_sdf_gradient
) {
  int ptr = mesh_unit.vertex_ptrs[vertex_idx];
  if (ptr == FREE_PTR) {
    int lock = atomicExch(&mesh_unit.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      ptr = mesh_m.AllocVertex();
    } /// Ensure that it is only allocated once
  }

  if (ptr >= 0) {
    Voxel voxel_query;
    //GetSpatialValue:get sdf value from pos(world_pos), store at voxel_query
    bool valid = GetSpatialValue(vertex_pos, blocks, hash_table,
                                 geometry_helper, &voxel_query);
    // printf("see vertex position:%f %f %f\n",vertex_pos.x,vertex_pos.y,vertex_pos.z);
    mesh_unit.vertex_ptrs[vertex_idx] = ptr;
    mesh_m.vertex(ptr).pos = vertex_pos;
    mesh_m.vertex(ptr).radius = sqrtf(1.0f / voxel_query.inv_sigma2);

    float3 grad;
    valid = GetSpatialSDFGradient(
        vertex_pos,
        blocks, hash_table,
        geometry_helper,
        &grad
    );
    float l = length(grad);
    mesh_m.vertex(ptr).normal = l > 0 && valid ? grad / l : make_float3(0);

    float rho = voxel_query.a/(voxel_query.a + voxel_query.b);
    //printf("%f %f\n", voxel_query.a, voxel_query.b);
    if (enable_bayesian) {
      if (color_type == 1) {
        mesh_m.vertex(ptr).color = make_float3(voxel_query.color.x,voxel_query.color.y,voxel_query.color.z) / 255.0f;
        //rgb vis
      }
      else if (color_type == 0) {mesh_m.vertex(ptr).color = ValToRGB(rho, 0.4f, 1.0f);}// rho vis
      //
    } else {
      mesh_m.vertex(ptr).color = ValToRGB(voxel_query.inv_sigma2/4000.0f, 0.2f, 1.0f);
    }
    // printf("if mesh update:%f %f %f - %f %f %f - %f(%d)\n",mesh_m.vertex(ptr).pos.x,mesh_m.vertex(ptr).pos.y,mesh_m.vertex(ptr).pos.z,
    //   mesh_m.vertex(ptr).normal.x,mesh_m.vertex(ptr).normal.y,mesh_m.vertex(ptr).normal.z,mesh_m.vertex(ptr).radius, ptr);
  }
  return ptr;
}


__device__
inline int AllocateVertexWithMutexScale(
    MeshUnit &mesh_unit,
    uint  &vertex_idx,
    const float3 &vertex_pos, int3 entry, int3 v,
    Mesh &mesh_m,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool color_type,
    int curr_scale,
    int query_scale,
    bool enable_bayesian,
    bool enable_sdf_gradient
) {
  int ptr = mesh_unit.vertex_ptrs[vertex_idx];
   //if(ptr>=4000000&&entry.x==4&&entry.y==2&&entry.z==1)
    // printf("out3:%d(%d %d %d) entry:%d %d %d->%d %d %d vertex_idx:%d pos:%f %f %f\n",ptr,mesh_unit.GetVertex(0),mesh_unit.GetVertex(1),mesh_unit.GetVertex(2),entry.x,entry.y,entry.z,v.x,v.y,v.z,vertex_idx,vertex_pos.x,vertex_pos.y,vertex_pos.z);
 
  if (ptr == FREE_PTR) {
    int lock = atomicExch(&mesh_unit.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      ptr = mesh_m.AllocVertex();
   
    } /// Ensure that it is only allocated once
   
  }

  if (ptr >= 0) {
   // Voxel voxel_query;

    //GetSpatialValue:get sdf value from pos(world_pos), store at voxel_query
    //bool valid = GetSpatialValue(vertex_pos, blocks, hash_table,
     //                            geometry_helper, &voxel_query);
//printf("vertex_idx:%d ptr:%d\n",vertex_idx, ptr);
    mesh_unit.vertex_ptrs[vertex_idx] = ptr;
    mesh_m.vertex(ptr).pos = vertex_pos;

 
   // mesh_m.vertex(ptr).radius = sqrtf(1.0f / voxel_query.inv_sigma2);

      /*
    float3 grad;
    valid = GetSpatialSDFGradient(
        vertex_pos,
        blocks, hash_table,
        geometry_helper,
        &grad
    );
    float l = length(grad);
    mesh_m.vertex(ptr).normal = l > 0 && valid ? grad / l : make_float3(0);
*/

    //  mesh_m.vertex(ptr).color = ValToRGB(voxel_query.inv_sigma2/4000.0f, 0.2f, 1.0f);
    // printf("if mesh update:%f %f %f - %f %f %f - %f(%d)\n",mesh_m.vertex(ptr).pos.x,mesh_m.vertex(ptr).pos.y,mesh_m.vertex(ptr).pos.z,
    //   mesh_m.vertex(ptr).normal.x,mesh_m.vertex(ptr).normal.y,mesh_m.vertex(ptr).normal.z,mesh_m.vertex(ptr).radius, ptr);
  }
  return ptr;
}


__device__
bool IfAlignTwoVertex(float3 this_v, float3 query_v, int index, int index2, float voxel_size){
  int direction = -1, direction2 = -1;
  //x:0,2,4,6(1) y:8,9,10,11(2) z:1,3,5,7(3)
  if(index==0 || index==2 || index==4 || index==6)
    direction = 1;
  else if(index==8 || index==9 || index==10 || index==11)
    direction = 2;
  else if(index==1 || index==3 || index==5 || index==7)
    direction = 3;

  if(index2==0 || index2==2 || index2==4 || index2==6)
    direction2 = 1;
  else if(index2==8 || index2==9 || index2==10 || index2==11)
    direction2 = 2;
  else if(index2==1 || index2==3 || index2==5 || index2==7)
    direction2 = 3;

  if(direction<0 || direction2<0)
    return false;

  if(direction!=direction2)
    return false;

  if(direction==1){
    if(abs(this_v.y-query_v.y)<0.3*voxel_size&&abs(this_v.z-query_v.z)<0.3*voxel_size)
      return true;
    else
      return false;
  }
  else if(direction==2){
    if(abs(this_v.x-query_v.x)<0.3*voxel_size&&abs(this_v.z-query_v.z)<0.3*voxel_size)
      return true;
    else
      return false;
  }
  else if(direction==3){
    if(abs(this_v.x-query_v.x)<0.3*voxel_size&&abs(this_v.y-query_v.y)<0.3*voxel_size)
      return true;
    else
      return false;
  }
  else
    return false;

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
void SurfelExtractionKernel(
    ScaleTable scale_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh_m,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    bool color_type,
    bool enable_bayesian,
    bool enable_sdf_gradient
    ) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block& block = blocks[entry.ptr];

       /* 
        if(threadIdx.x==0){
            int3 aa = scale_table.GetAncestor(entry.pos);
            if(entry.pos.x!=aa.x||entry.pos.y!=aa.y||entry.pos.z!=aa.z)
                printf("WRONG ANCESTOR:%d %d %d->%d %d %d\n",entry.pos.x,entry.pos.y,entry.pos.z,aa.x,aa.y,aa.z);
        }
        */
  if (threadIdx.x == 0) {
    block.active = block.min_active_voxel_pi <= 0.95f;
  }
  const uint curr_scale = scale_table.GetScale(entry.pos).scale;
  // if(curr_scale==1)
  //   return;
  __syncthreads();

  int3   voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3  offset = geometry_helper.DevectorizeIndex(threadIdx.x);
    
  int3   voxel_pos = voxel_base_pos + make_int3(offset)*pow(2,curr_scale-1);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  Voxel& this_voxel = block.voxels[threadIdx.x];

  //this_voxel.Check(entry.pos,entry.ptr,threadIdx.x);
  //////////
  /// 1. Read the scalar values, see mc_tables.h
  const int kVertexCount = 8;
  const float kVoxelSize = geometry_helper.voxel_size;
  const float kThreshold = 10.0f;  //bigger for 10, smaller for 0.5
  const float kIsoLevel = 0;

  float  d[kVertexCount];
  float3 p[kVertexCount];

  short cube_index = 0;

  this_mesh_unit.prev_cube_idx = this_mesh_unit.curr_cube_idx;
  this_mesh_unit.curr_cube_idx = 0;

  if (enable_bayesian) {
    float rho = this_voxel.a / (this_voxel.a + this_voxel.b);
      //printf("%f %f\n",rho,this_voxel.inv_sigma2);
    //if (rho < 0.1f || this_voxel.inv_sigma2 < squaref(0.25f / kVoxelSize))
    if (rho < 0.3f || this_voxel.inv_sigma2 < squaref(0.5f / kVoxelSize))
      return;
  } 
  else {      
    if (this_voxel.inv_sigma2 == 0) return;
  }

  /// Check 8 corners of a cube: are they valid?
  Voxel voxel_query;
 // int SaveNoValueCorner[8][3] = {{},{},{},{},
   //                              {},{},{},{}};
  for (int i = 0; i < kVertexCount; ++i) {
    
    float3 query_corner;
    int check_flag = 0;
    bool GetVoxelValueSuccess = GetVoxelValue(entry, voxel_pos + kVtxOffset[i] * pow(2,curr_scale-1),
                                             blocks, hash_table, scale_table,
                                             geometry_helper, &voxel_query, voxel_pos, &query_corner, &check_flag);
    if (!GetVoxelValueSuccess){
        //if check_flag==2, the corner cannot find right voxel because the small neighbour scale. 
      /* 
      if(curr_scale>=4&&entry.pos.x==-8&&entry.pos.y==-24&&entry.pos.z==0){
        int3 v = voxel_pos + kVtxOffset[i] * pow(2,curr_scale-1);
        printf("entry:%d %d %d(%d) voxel:%d %d %d query:%d %d %d(%d) sdf:%f weight:%f\n",entry.pos.x,entry.pos.y,entry.pos.z,curr_scale,voxel_pos.x,voxel_pos.y,voxel_pos.z,v.x,v.y,v.z,scale_table.GetScale(scale_table.GetAncestor(v)).scale,voxel_query.sdf,voxel_query.inv_sigma2);
      }
      */  
      return;
    }
   
      /*
    if(curr_scale==3){
        int3 fu = geometry_helper.VoxelToBlock(voxel_pos+kVtxOffset[i]*pow(2,curr_scale-1));
      printf("pos:%d %d %d->corner:%d %d %d(%d)\n",entry.pos.x,entry.pos.y,entry.pos.z,fu.x,fu.y,fu.z,scale_table.GetScale(fu).scale);
    }
    */

   // if(scale_table.GetScale(geometry_helper.VoxelToBlock(voxel_pos + kVtxOffset[i]*pow(2,curr_scale-1))).scale<0)
    //  return;

    int3 pp = voxel_pos + kVtxOffset[i] * pow(2,curr_scale-1);
  //  if(curr_scale==3)
    //if(scale_table.GetScale(geometry_helper.VoxelToBlock(voxel_pos)).scale!=scale_table.GetScale(geometry_helper.VoxelToBlock(pp)).scale)
  //    printf("voxel:%d %d %d(%d)->sdf:%f-%d %d %d(%d)\n",voxel_pos.x,voxel_pos.y,voxel_pos.z,scale_table.GetScale(geometry_helper.VoxelToBlock(voxel_pos)).scale,voxel_query.sdf,pp.x,pp.y,pp.z,scale_table.GetScale(geometry_helper.VoxelToBlock(pp)).scale);
   
    d[i] = voxel_query.sdf;

    if (fabs(d[i]) > kThreshold) return;

    int3 vv = voxel_pos + kVtxOffset[i] * pow(2,curr_scale-1);
    
    if (enable_bayesian) {
      float rho = this_voxel.a / (this_voxel.a + this_voxel.b);
      //if (rho < 0.1f || voxel_query.inv_sigma2 < squaref(0.1f / kVoxelSize))
      if (rho < 0.3f || voxel_query.inv_sigma2 < squaref(0.5f / kVoxelSize))
      return;
    } else {
      if (voxel_query.inv_sigma2 == 0) return;
    }
   
    p[i] = query_corner;
    //p[i] = world_pos + kVoxelSize * make_float3(kVtxOffset[i]) * pow(2,curr_scale-1);
   
    // if(this_mesh_unit.prev_cube_idx!=0)
    //   printf("this:%d index:%d i:%d\n",prev_situation, this_mesh_unit.prev_cube_idx, i);
    if (d[i] <= kIsoLevel) cube_index |= (1 << i); 
    // printf("cube_index:%f\n",cube_index);
    // if(curr_scale>1)
    //   printf("kVoxelSize:%f %d\n",kVoxelSize, curr_scale);

  }


  /*
  if(scale_table.GetScale(entry.pos).scale>=4&&(cube_index==0||cube_index==255)&&entry.pos.x==-8&&entry.pos.y==-24&&entry.pos.z==0){
 // if((voxel_pos.z>0 || abs(voxel_pos.z) <= 0.08)&&scale_table.GetScale(entry.pos).scale>1){
  // if(this_voxel.inv_sigma2>10){
 // if(threadIdx.x==191){
    uint3 oo = geometry_helper.DevectorizeIndex(threadIdx.x);
    printf("voxel_pos:(%d %d %d) block_pos:%d %d %d, sdf:%f %f %f %f %f %f %f %f(%f) -> (%d->%d) scale:%d block:%d idx:%d(%d %d %d)\n",
      voxel_pos.x,voxel_pos.y,voxel_pos.z,entry.pos.x,entry.pos.y,entry.pos.z,
       d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],this_voxel.inv_sigma2,this_mesh_unit.prev_cube_idx,cube_index, scale_table.GetScale(entry.pos).scale,blockIdx.x,threadIdx.x,oo.x,oo.y,oo.z);
  }
  */
  this_mesh_unit.curr_cube_idx = cube_index;
  //printf("pos:%d %d %d idx:%d\n",entry.pos.x,entry.pos.y,entry.pos.z,cube_index);
  /*
  // if((this_mesh_unit.curr_cube_idx==0||this_mesh_unit.curr_cube_idx==255) && (this_mesh_unit.prev_cube_idx!=0)&&(this_mesh_unit.prev_cube_idx!=255) && entry.pos.z==0&&entry.pos.x>8&&entry.pos.y>8){  
   //if(entry.pos.z==0&&(entry.pos.x<-10||entry.pos.x>10)&&(entry.pos.y<-10||entry.pos.y>10)&&(this_mesh_unit.curr_cube_idx==0||this_mesh_unit.curr_cube_idx==255)){
   if(curr_scale==3){
    uint3 xyz = geometry_helper.DevectorizeIndex(threadIdx.x);
       if(xyz.z!=0)
         return;
     printf("index tran:%d -> %d(sdf:%f %f %f %f %f %f %f %f) entry:%d %d %d, threadIdx.x:%d %d %d scale:%d\n",
            this_mesh_unit.prev_cube_idx,this_mesh_unit.curr_cube_idx,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],
           entry.pos.x,entry.pos.y,entry.pos.z,xyz.x,xyz.y,xyz.z,curr_scale);
   }
   */
   /*
   if((this_mesh_unit.curr_cube_idx==0||this_mesh_unit.curr_cube_idx==255)&&(this_mesh_unit.prev_cube_idx!=0&&this_mesh_unit.prev_cube_idx!=255)){
  //   // this_voxel
     for(int pp=0;pp<kVertexCount;pp++){
       printf("  sdf:%f %f %f %f %f %f %f %f\n",d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
     }
     printf("before:%d after:%d\nsdf:%f inv_sigma2:%f\n",this_mesh_unit.prev_cube_idx,this_mesh_unit.curr_cube_idx,this_voxel.sdf,this_voxel.inv_sigma2);
   }
   */
  if (cube_index == 0 || cube_index == 255) return;

  const int kEdgeCount = 12;
#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    // printf("cube_index:%f\n",cube_index);
    if (kCubeEdges[cube_index] & (1 << i)) {
      int2 edge_endpoint_vertices = kEdgeEndpointVertices[i];
      uint4 edge_cube_owner_offset = kEdgeOwnerCubeOffset[i];

      // Special noise-bit interpolation here: extrapolation
      float3 vertex_pos = VertexIntersection(
          p[edge_endpoint_vertices.x],
          p[edge_endpoint_vertices.y],
          d[edge_endpoint_vertices.x],
          d[edge_endpoint_vertices.y],
          kIsoLevel);


     // if(vertex_pos.z==0)
      // printf("vertex:%f %f %f\n",vertex_pos.x,vertex_pos.y,vertex_pos.z);
      int3 query_block_catch = geometry_helper.VoxelToBlock(voxel_pos + make_int3(edge_cube_owner_offset.x,
                                                                 edge_cube_owner_offset.y,
                                                                edge_cube_owner_offset.z)*pow(2,curr_scale-1));
      int3 query_block_result =scale_table.GetAncestor(query_block_catch);
     // if(hash_table.GetEntry(query_block_result).ptr==FREE_ENTRY)
       // printf("scale:%d ptr:%d\n",scale_table.GetScale(query_block_result).scale, hash_table.GetEntry(query_block_result).ptr);
      int query_scale = scale_table.GetScale(query_block_result).scale;

        /*
     if(entry.pos.x>7&&entry.pos.y>7&&vertex_pos.z!=0&&entry.pos.z==0){
       printf("pos:%d %d %d curr_scale:%d(p:%f %f %f d:%f) query_scale:%d(p:%f %f %f d:%f) vertex:%f %f %f\n",entry.pos.x,entry.pos.y,entry.pos.z,curr_scale,p[edge_endpoint_vertices.x].x,p[edge_endpoint_vertices.x].y,p[edge_endpoint_vertices.x].z,d[edge_endpoint_vertices.x],query_scale,p[edge_endpoint_vertices.y].x,p[edge_endpoint_vertices.y].y,p[edge_endpoint_vertices.y].z,d[edge_endpoint_vertices.y],vertex_pos.x,vertex_pos.y,vertex_pos.z);
     }
*/
      //if(entry.pos.x==26&&entry.pos.y==9&&entry.pos.z==56)
      //  printf("see voxel:%d %d %d visit:%d %d %d query:%d %d %d\n",voxel_pos.x,voxel_pos.y,voxel_pos.z,query_block_catch.x,query_block_catch.y,query_block_catch.z, query_block_result.x,query_block_result.y,query_block_result.z);

      MeshUnit &mesh_unit = GetMeshUnitRefScale(
          entry,
          voxel_pos + make_int3(edge_cube_owner_offset.x,
                                edge_cube_owner_offset.y,
                                edge_cube_owner_offset.z) * pow(2,curr_scale-1),
          query_block_result,
          blocks, hash_table, scale_table,
          geometry_helper,
          curr_scale,
          query_scale);
//printf("22:%d %d %d voxel_pos:%d %d %d scale:%d->%d\n",entry.pos.x,entry.pos.y,entry.pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z,curr_scale,query_scale);

      AllocateVertexWithMutexScale(
          mesh_unit,
          edge_cube_owner_offset.w,
          vertex_pos,
          entry.pos, voxel_pos + make_int3(edge_cube_owner_offset.x,edge_cube_owner_offset.y,edge_cube_owner_offset.z)*pow(2,curr_scale-1),
          mesh_m,
          blocks,
          hash_table,
          geometry_helper,
          color_type,
          curr_scale,
          query_scale,
          enable_bayesian,
          enable_sdf_gradient);  
//printf("33:%d %d %d voxel_pos:%d %d %d scale:%d->%d\n",entry.pos.x,entry.pos.y,entry.pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z,curr_scale,query_scale);

    }
  }
}

//modify surfel position between different block scales.
//19.10.27
// __global__
// void AlignMeshesKernel(
//     EntryArray candidate_entries,
//     BlockArray blocks,
//     Mesh mesh_m,
//     HashTable hash_table,
//     ScaleTable scale_table,
//     GeometryHelper geometry_helper,
//     float global_voxel_size
// ){
//   const HashEntry &entry = candidate_entries[blockIdx.x];
//   Block& block = blocks[entry.ptr];

//   // if (threadIdx.x == 0) {
//   //   block.active = block.min_active_voxel_pi <= 0.95f;
//   // }
//   __syncthreads();

//   //1.find neighbour from 3 direction
//   //2.choose the one has biggest scale
//   //3.start align
//   //3.1.align vertex inside edge
//   //3.2.align vertex outside edge

//   int3   voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
//   uint3  offset = geometry_helper.DevectorizeIndex(threadIdx.x);

//   if(offset.x!=0 && offset.x!=BLOCK_SIDE_LENGTH 
//     && offset.y!=0 && offset.y!=BLOCK_SIDE_LENGTH
//     && offset.z!=0 && offset.z!=BLOCK_SIDE_LENGTH)
//     return;

//   int3   voxel_pos = voxel_base_pos + make_int3(offset);
//   float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);
//   int curr_scale = scale_table.GetScale(voxel_pos).scale;

//   MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
//   Voxel& this_voxel = block.voxels[threadIdx.x];

// //Get surfel at 12 edges
//   const int kEdgeCount = 12;
//   int vertex_ptrs[kEdgeCount];

// #pragma unroll 1
//   for (int i = 0; i < kEdgeCount; ++i) {
//     if (kCubeEdges[this_mesh_unit.curr_cube_idx] & (1 << i)) {
//       uint4 edge_owner_cube_offset = kEdgeOwnerCubeOffset[i];

//       MeshUnit &mesh_unit  = GetMeshUnitRef(
//           entry,
//           voxel_pos + make_int3(edge_owner_cube_offset.x,
//                                 edge_owner_cube_offset.y,
//                                 edge_owner_cube_offset.z),
//           blocks,
//           hash_table,
//           geometry_helper);

//       vertex_ptrs[i] = mesh_unit.GetVertex(edge_owner_cube_offset.w);
//       mesh_unit.ResetMutexes();
//     }
//   }
  
//   int FindNotNeighborOffset[2][3][12] = {{{0,1,0,-1,0,1,0,-1,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},{1,0,-1,0,1,0,-1,0,1,1,-1,-1}},
//                                          {{0,0,0,0,0,0,0,0,-1,1,1,-1},{1,1,1,1,-1,-1,-1,-1,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}}};
//   // int EdgeIndexOffset[3][12] = {{},
//   //                               {},
//   //                               {}};
//   //Align surfel's position. (vertex_ptr[12])
//   // for(int i = 0;i < kEdgeCount; ++i) {
//   //   Vertex this_v = mesh_m.vertex(vertex_ptrs[i]);
//   //   printf("idx:%d\n",vertex_ptrs[i]); 
//   //   // printf("surfel pos:%f %f %f %d\n",this_v.pos.x,this_v.pos.y,this_v.pos.z, i);
//   // }

//   bool HasAligned[12] = {0};
//   for (int t = 0;
//        kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t] != -1;
//        t++) 
//   {
//      Vertex this_v = mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t]]);
//      //this vertex's owner is (int3)voxel_pos, its index is kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t]
//      int edge_index = kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t];
//      if(HasAligned[edge_index]==1)
//       continue;
//      for(int j=0;j<2;j++){
//       //should change to 
//       int3 query_pos = voxel_pos + make_int3(FindNotNeighborOffset[j][0][edge_index],
//                                              FindNotNeighborOffset[j][1][edge_index],
//                                              FindNotNeighborOffset[j][2][edge_index]) * 1;
//       // float3 query_pos = make_float3(this_v.pos.x,this_v.pos.y,this_v.pos.z) + 
//       //                    make_float3(FindNotNeighborOffset[j][0][edge_index],
//       //                                FindNotNeighborOffset[j][1][edge_index],
//       //                                FindNotNeighborOffset[j][2][edge_index]) * 0.5 * global_voxel_size;
//       // int3 query_pos_int = geometry_helper.WorldToVoxeli(query_pos);
//       int query_scale = scale_table.GetScale(query_pos).scale;
//       // if(query_scale!=1)
//       //   printf("sacle:%d\n",query_scale);
      
//       if(query_scale > curr_scale){
//         // int3 ancestor_pos = query_pos_int;
//         int3 ancestor_pos = geometry_helper.WorldToBlock(geometry_helper.BlockToWorld(query_pos), query_scale*geometry_helper.voxel_size);
//         // if(query_pos.x!=ancestor_pos.x||query_pos.y!=ancestor_pos.y||query_pos.z!=ancestor_pos.z){
//         //   printf("son:%d %d %d->%d %d %d(%d)\n",query_pos.x,query_pos.y,query_pos.z,ancestor_pos.x,ancestor_pos.y,ancestor_pos.z,j);
//         // }
//       }
      
      
//      }

//      // printf("surfel pos:%f %f %f %d\n",this_v.pos.x,this_v.pos.y,this_v.pos.z, vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t]]);
//   }

// }
__global__
void AlignMeshesKernel(
  EntryArray candidate_entries,
  BlockArray blocks,
  Mesh mesh_m,
  HashTable hash_table,
  ScaleTable scale_table,
  GeometryHelper geometry_helper,
  float global_voxel_size
){
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block& block = blocks[entry.ptr];

  // if (threadIdx.x == 0) {
  //   block.active = block.min_active_voxel_pi <= 0.95f;
  // }
  __syncthreads();

  int3   voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3  offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int    curr_scale = scale_table.GetScale(entry.pos).scale;

  if(offset.x!=0 && offset.x!=BLOCK_SIDE_LENGTH 
  && offset.y!=0 && offset.y!=BLOCK_SIDE_LENGTH
  && offset.z!=0 && offset.z!=BLOCK_SIDE_LENGTH)
  return;

  int3   voxel_pos = voxel_base_pos + make_int3(offset) * pow(2,curr_scale-1);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  Voxel& this_voxel = block.voxels[threadIdx.x];

  //Get surfel at 12 edges
  const int kEdgeCount = 12;
  int vertex_ptrs[kEdgeCount];

  #pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kCubeEdges[this_mesh_unit.curr_cube_idx] & (1 << i)) {
      uint4 edge_owner_cube_offset = kEdgeOwnerCubeOffset[i];

      MeshUnit &mesh_unit  = GetMeshUnitRef(
          entry,
          voxel_pos + make_int3(edge_owner_cube_offset.x,
                                edge_owner_cube_offset.y,
                                edge_owner_cube_offset.z)*pow(2,curr_scale-1),
          blocks,
          hash_table,
          geometry_helper);

      vertex_ptrs[i] = mesh_unit.GetVertex(edge_owner_cube_offset.w);
      mesh_unit.ResetMutexes();
    }
  }

  //FindNotNeighborOffset[x][y][z]:for vertex in edge z, find neighbor along direction x 
  /*    z
         /                        
        4-----5              ---4---
       /     /|             /      /|9
      7-----6-1--->x        ---6---
      |     |/             | 3    |/1
      3-----2               ---2--
      |
      y  
      */
  // int FindNotNeighborOffset[2][3][12] = {{{0,1,0,-1,0,1,0,-1,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0},{1,0,-1,0,1,0,-1,0,1,1,-1,-1}},
  //                                      {{0,0,0,0,0,0,0,0,-1,1,1,-1},{1,1,1,1,-1,-1,-1,-1,0,0,0,0},{0,0,0,0,0,0,0,0,0,0,0,0}}};
  int FindNotNeighborOffset[3][3][12] = 
    {{{0,1,0,1,0,1,0,1,1,1,1,1},{1,1,1,1,1,1,1,-1,0,0,0,0},{1,0,1,0,1,0,-1,0,1,1,1,-1}},
     {{0,1,0,-1,0,1,0,-1,-1,1,1,-1},{1,-1,1,1,-1,-1,-1,1,0,0,0,0},{-1,0,-1,0,-1,0,1,0,1,-1,-1,1}},
     {{0,-1,0,-1,0,-1,0,-1,-1,-1,-1,-1},{-1,1,-1,-1,-1,-1,-1,-1,0,0,0,0},{1,0,-1,0,-1,0,-1,0,-1,1,-1,-1}}};

    //direction:(1,1)(1,-1)(-1,1)(-1,-1)(except itself) from edge y
    int OppositeEdge[3][12] = {{6,7,6,7,6,7,4,5,11,11,11,8},
                 {4,3,4,5,2,3,2,3,10,8,8,10},
                 {2,5,0,1,0,1,0,1,9,10,9,9}};
  // int EdgeIndexOffset[3][12] = {{},
  //                               {},
  //                               {}};
  //Align surfel's position. (vertex_ptr[12])
  // for(int i = 0;i < kEdgeCount; ++i) {
  //   Vertex this_v = mesh_m.vertex(vertex_ptrs[i]);
  //   printf("idx:%d\n",vertex_ptrs[i]); 
  //   // printf("surfel pos:%f %f %f %d\n",this_v.pos.x,this_v.pos.y,this_v.pos.z, i);
  // }

  bool HasAligned[12] = {0};
  for (int t = 0;
     kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t] != -1;
     t++) 
  {
    Vertex& this_v = mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t]]);
    int biggest_neighbor_index = -1;
    int biggest_neighbor_scale = -1;
    //this vertex's owner is (int3)voxel_pos, its index is kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t]
    int edge_index = kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t];
    float3 corner1 = geometry_helper.VoxelToWorld(voxel_pos + kVtxOffset[kEdgeEndpointVertices[edge_index].x] * 1);
    float3 corner2 = geometry_helper.VoxelToWorld(voxel_pos + kVtxOffset[kEdgeEndpointVertices[edge_index].y] * 1);
    float3 edge_midpoint = 0.5*(corner1 + corner2);

    if(HasAligned[edge_index]==1)
      continue;
    for(int j=0;j<3;j++){
      float3 query_pos = edge_midpoint + make_float3(FindNotNeighborOffset[j][0][edge_index],
                                           FindNotNeighborOffset[j][1][edge_index],
                                           FindNotNeighborOffset[j][2][edge_index]) * geometry_helper.voxel_size;
      // float3 query_pos = make_float3(this_v.pos.x,this_v.pos.y,this_v.pos.z) + 
      //                    make_float3(FindNotNeighborOffset[j][0][edge_index],
      //                                FindNotNeighborOffset[j][1][edge_index],
      //                                FindNotNeighborOffset[j][2][edge_index]) * 0.5 * global_voxel_size;
      int3 query_pos_voxel = geometry_helper.WorldToVoxeli(query_pos);
      int query_scale = scale_table.GetScale(query_pos_voxel).scale;
      if(query_scale > biggest_neighbor_scale){
        biggest_neighbor_scale = query_scale;
        biggest_neighbor_index = j;
      }
    // if(query_scale!=1)
    //   printf("sacle:%d\n",query_scale);

    // if(query_scale > curr_scale){
    // int3 ancestor_pos = query_pos_int;
    // int3 ancestor_pos = geometry_helper.WorldToBlock(geometry_helper.BlockToWorld(query_pos), query_scale*geometry_helper.voxel_size);
    // if(query_pos.x!=ancestor_pos.x||query_pos.y!=ancestor_pos.y||query_pos.z!=ancestor_pos.z){
    //   printf("son:%d %d %d->%d %d %d(%d)\n",query_pos.x,query_pos.y,query_pos.z,ancestor_pos.x,ancestor_pos.y,ancestor_pos.z,j);
    // }
    } 
    //adjust vertex in edge 'kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t]' to biggest neighbor
    if(biggest_neighbor_scale > curr_scale){
      //0.get the biggest neighbor
      float3 query_pos = edge_midpoint + make_float3(FindNotNeighborOffset[biggest_neighbor_index][0][edge_index],
                                                   FindNotNeighborOffset[biggest_neighbor_index][1][edge_index],
                                                   FindNotNeighborOffset[biggest_neighbor_index][2][edge_index]) * geometry_helper.voxel_size;
      int3 query_pos_block = geometry_helper.WorldToBlock(query_pos, geometry_helper.voxel_size); //finest 
        int query_scale = scale_table.GetScale(query_pos_block).scale;
        int3 query_block_pos = 
            geometry_helper.WorldToBlock(query_pos, query_scale * geometry_helper.voxel_size) * query_scale;
        int3 query_voxel_pos = 
            geometry_helper.WorldToVoxeli(query_pos, query_scale * geometry_helper.voxel_size) * query_scale;

        // printf("pos:%d %d %d <-> block:%d %d %d(%d)\n",voxel_pos.x,voxel_pos.y,voxel_pos.z,
        //                                            block_pos.x,block_pos.y,block_pos.z,curr_scale);
        uint3 this_offset = geometry_helper.VoxelToOffset(query_block_pos, query_voxel_pos, query_scale);
        // printf("offset:%d %d %d\n",this_offset.x,this_offset.y,this_offset.z);
        uint voxel_index = geometry_helper.VectorizeOffset(this_offset);
        HashEntry query_entry = hash_table.GetEntry(query_block_pos);
        if(query_entry.ptr == FREE_ENTRY)
          return;
        //So the neighbor voxel is blocks[query_entry.ptr].xxx[voxel_index]
      //1.align vertex inside the edge
            MeshUnit &query_mesh_unit = blocks[query_entry.ptr].mesh_units[voxel_index];
            for (int t1 = 0;
              kTriangleVertexEdge[query_mesh_unit.curr_cube_idx][t1] != -1;
              t1++) 
            {
              //
              float3 query_v = mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[query_mesh_unit.curr_cube_idx][t1]]).pos;
              
              if(IfAlignTwoVertex(this_v.pos, query_v, edge_index, 
                kTriangleVertexEdge[query_mesh_unit.curr_cube_idx][t1], geometry_helper.voxel_size)){
                this_v.pos = query_v;
                //a vertex only need align once
                break;
              }
            }
      //2.align vertex outside the edge(TODO)
      // __syncthreads();
    }   
  }
   // printf("surfel pos:%f %f %f %d\n",this_v.pos.x,this_v.pos.y,this_v.pos.z, vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t]]);
}

__global__
void TriangleExtractionKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh_m,
    HashTable hash_table,
    ScaleTable scale_table,
    GeometryHelper geometry_helper,
    bool enable_sdf_gradient
) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block& block = blocks[entry.ptr];
  if (threadIdx.x == 0) {
    block.boundary_surfel_count = 0;
    block.inner_surfel_count = 0;
  }
  __syncthreads();

  int curr_scale = scale_table.GetScale(entry.pos).scale;

  int3   voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3  offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3   voxel_pos = voxel_base_pos + make_int3(offset)*pow(2,curr_scale-1);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  bool is_inner = IsInner(offset);
  for (int i = 0; i < 3; ++i) {
    if (this_mesh_unit.vertex_ptrs[i] >= 0) {
      if (is_inner) {
        atomicAdd(&block.inner_surfel_count, 1);
      } else {
        atomicAdd(&block.boundary_surfel_count, 1);
      }
    }
  }
  /// Cube type unchanged: NO need to update triangles
//  if (this_cube.curr_cube_idx == this_cube.prev_cube_idx) {
//    blocks[entry.ptr].voxels[local_idx].stats.duration += 1.0f;
//    return;
//  }
//  blocks[entry.ptr].voxels[local_idx].stats.duration = 0;
//printf("pos:%d %d %d, idx:%d\n",entry.pos.x,entry.pos.y,entry.pos.z,this_mesh_unit.curr_cube_idx);
  if(curr_scale>=4&&voxel_pos.z==0&&(this_mesh_unit.curr_cube_idx==0||this_mesh_unit.curr_cube_idx==255))
    printf("voxel:%d %d %d entry:%d %d %d(%d %d %d) idx:%d\n",voxel_pos.x,voxel_pos.y,voxel_pos.z,entry.pos.x,entry.pos.y,entry.pos.z,offset.x,offset.y,offset.z,this_mesh_unit.curr_cube_idx);
  if (this_mesh_unit.curr_cube_idx == 0
      || this_mesh_unit.curr_cube_idx == 255) {

     return;
  }
  
  //////////
  /// 2. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  /// This operation is in 2-pass
  /// pass2: Assign
  const int kEdgeCount = 12;
  int vertex_ptrs[kEdgeCount];

#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kCubeEdges[this_mesh_unit.curr_cube_idx] & (1 << i)) {
      uint4 edge_owner_cube_offset = kEdgeOwnerCubeOffset[i];
      int3 query_block_catch = geometry_helper.VoxelToBlock(voxel_pos + make_int3(edge_owner_cube_offset.x,
                                edge_owner_cube_offset.y,
                                edge_owner_cube_offset.z)*pow(2,curr_scale-1));
      int3 query_block_result =scale_table.GetAncestor(query_block_catch);
      int query_scale = scale_table.GetScale(query_block_result).scale;

     // if(entry.pos.x==26&&entry.pos.y==9&&entry.pos.z==56)
      //  printf("see voxel:%d %d %d visit:%d %d %d query:%d %d %d\n",voxel_pos.x,voxel_pos.y,voxel_pos.z,query_block_catch.x,query_block_catch.y,query_block_catch.z, query_block_result.x,query_block_result.y,query_block_result.z);

      MeshUnit &mesh_unit  = GetMeshUnitRefScale(
          entry,
          voxel_pos + make_int3(edge_owner_cube_offset.x,
                                edge_owner_cube_offset.y,
                                edge_owner_cube_offset.z) * pow(2,curr_scale-1),
          query_block_result,
          blocks,
          hash_table, scale_table,
          geometry_helper, curr_scale, query_scale);

      vertex_ptrs[i] = mesh_unit.GetVertex(edge_owner_cube_offset.w);
      mesh_unit.ResetMutexes();
    }
  }

  //////////
  /// 3. Assign triangles
  int i = 0;
  for (int t = 0;
       kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t] != -1;
       t += 3, ++i) {
    if(i>=5)
      printf("i:%d\n",i);
    int triangle_ptr = this_mesh_unit.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) {
      triangle_ptr = mesh_m.AllocTriangle();
    } else {
      mesh_m.ReleaseTriangle(mesh_m.triangle(triangle_ptr));
    }
    this_mesh_unit.triangle_ptrs[i] = triangle_ptr;
    if(this_mesh_unit.curr_cube_idx<0||this_mesh_unit.curr_cube_idx>=256||t<0||t+2>=16)
      printf("see:%d %d\n",this_mesh_unit.curr_cube_idx,t);
    
    
    mesh_m.AssignTriangle(
        mesh_m.triangle(triangle_ptr),
        make_int3(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 0]],
                  vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 1]],
                  vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 2]]),
        curr_scale);

     /*    
    if(curr_scale>0&&blockIdx.x<100&&threadIdx.x==191){
        printf("show vertex:%d(%f %f %f)-%d(%f %f %f)-%d(%f %f %f) scale:%d index:%d (block:%d idx:%d t:%d)\n",
                                    vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 0]],
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 0]]).pos.x,
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 0]]).pos.y,
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 0]]).pos.z,
                                    vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 1]],
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 1]]).pos.x,
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 1]]).pos.y,
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 1]]).pos.z,
                                    vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 2]],
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 2]]).pos.x,
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 2]]).pos.y,
                                    mesh_m.vertex(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 2]]).pos.z,
                                    curr_scale,this_mesh_unit.curr_cube_idx,blockIdx.x, threadIdx.x,t);
       }
      */
    if (!enable_sdf_gradient) {
      mesh_m.ComputeTriangleNormal(mesh_m.triangle(triangle_ptr));
    }
  }
}

/// Garbage collection (ref count)
__global__
void RecycleTrianglesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh_m) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[threadIdx.x];

  int i = 0;
  for (int t = 0;
       kTriangleVertexEdge[mesh_unit.curr_cube_idx][t] != -1;
       t += 3, ++i);

  // printf("clear:%d %d(%d)",i,N_TRIANGLE,mesh_unit.curr_cube_idx);
  for (; i < N_TRIANGLE; ++i) {
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
void RecycleVerticesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh_m
) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[threadIdx.x];

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (mesh_unit.vertex_ptrs[i] != FREE_PTR &&
        mesh_m.vertex(mesh_unit.vertex_ptrs[i]).ref_count == 0) {
      mesh_m.vertex(mesh_unit.vertex_ptrs[i]).Clear();
      mesh_m.FreeVertex(mesh_unit.vertex_ptrs[i]);
      mesh_unit.vertex_ptrs[i] = FREE_PTR;
    }
  }
}

////////////////////
/// Host code
////////////////////
float MarchingCubes(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh_m,
    HashTable &hash_table,
    ScaleTable &scale_table,
    GeometryHelper &geometry_helper,
    bool color_type,
    bool enable_bayesian,
    bool enable_sdf_gradient,
    float global_voxel_size
) {
  uint occupied_block_count = candidate_entries.count();
  LOG(INFO) << "Marching cubes block count: " << occupied_block_count;
  if (occupied_block_count == 0)
    return -1;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  /// Use divide and conquer to avoid read-write conflict
  Timer timer;
  timer.Tick();
    /*
    c<<<grid_size, dim3(1,1)>>>(hash_table, candidate_entries, blocks);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    */
  SurfelExtractionKernel << < grid_size, block_size >> > (
          scale_table,
          candidate_entries,
          blocks,
          mesh_m,
          hash_table,
          geometry_helper,
          color_type,
          enable_bayesian,
          enable_sdf_gradient);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass1_seconds = timer.Tock();
  LOG(INFO) << "Pass1 duration: " << pass1_seconds;

  timer.Tick();
  // AlignMeshesKernel << < grid_size, block_size >> >(
  //   candidate_entries,
  //   blocks,
  //   mesh_m,
  //   hash_table,
  //   scale_table,
  //   geometry_helper,
  //   global_voxel_size);
  // checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaGetLastError());
  double align_seconds = timer.Tock();
  LOG(INFO) << "Align meshes: " << align_seconds;

  timer.Tick();
  TriangleExtractionKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          mesh_m,
          hash_table,
          scale_table,
          geometry_helper,
          enable_sdf_gradient);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass2_seconds = timer.Tock();
  LOG(INFO) << "Pass2 duration: " << pass2_seconds;

  // Think about how to deal with this recycle operation;.p
  RecycleTrianglesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh_m);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleVerticesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh_m);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return (float)(pass1_seconds + pass2_seconds);
}

// float AlignMeshes(
//   EntryArray& candidate_entries,
//   BlockArray& blocks,
//   Mesh& mesh_m,
//   HashTable& hash_table,
//   GeometryHelper& geometry_helper
// ){
//   Timer timer;
//   timer.Tick();
//   uint occupied_block_count = candidate_entries.count();
//   if(occupied_block_count==0)
//     return -1;
//   const uint threads_per_block = BLOCK_SIZE;
//   const dim3 grid_size(occupied_block_count, 1);
//   const dim3 block_size(threads_per_block, 1);

//   AlignMeshesKernel <<<grid_size, block_size>>>(
//     candidate_entries,
//     blocks,
//     mesh_m,
//     hash_table,
//     geometry_helper);
//   checkCudaErrors(cudaDeviceSynchronize());
//   checkCudaErrors(cudaGetLastError());

//   return timer.Tock();
// }
