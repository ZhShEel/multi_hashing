//
// Created by wei on 17-10-25.
//

#ifndef GEOMETRY_VOXEL_QUERY_H
#define GEOMETRY_VOXEL_QUERY_H

#include <matrix.h>
#include "geometry_helper.h"

#include "core/hash_table.h"
#include "core/block_array.h"


// TODO(wei): refine it
// function:
// at @param world_pos
// get Voxel in @param blocks
// with the help of @param hash_table and geometry_helper

// function:
// block-pos @param curr_entry -> voxel-pos @param voxel_local_pos
// get Voxel in @param blocks
// with the help of @param hash_table and geometry_helper

__device__
inline Voxel &GetVoxelRefScale(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const int3 query_block_pos,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper, int curr_scale, int query_scale
) {
  //int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(query_block_pos, voxel_pos, query_scale);

  //if (curr_entry.pos == block_pos) {
    if(curr_entry.pos == query_block_pos) {
    uint i = geometry_helper.VectorizeOffset(offset);
 // if(query_block_pos.x!=block_pos.x||query_block_pos.y!=block_pos.y||query_block_pos.z!=block_pos.z)
     // printf("wrong:%d %d %d-%d %d %d\n",query_block_pos.x,query_block_pos.y,query_block_pos.z,block_pos.x,block_pos.y,block_pos.z);

    return blocks[curr_entry.ptr].voxels[i];
  } else {
      
    HashEntry entry = hash_table.GetEntry(query_block_pos);
    //printf("curr_scale:%d query_scale:%d\n",curr_scale,query_scale);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetVoxelRef-(%d %d %d)scale:(%d->%d): should never reach here!\n",entry.pos.x,entry.pos.y,entry.pos.z,curr_scale,query_scale);
    }
    uint i = geometry_helper.VectorizeOffset(offset);
  
    return blocks[entry.ptr].voxels[i];
  }
}

__device__
inline Voxel &GetVoxelRef(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

  if (curr_entry.pos == block_pos) {
    uint i = geometry_helper.VectorizeOffset(offset);
    return blocks[curr_entry.ptr].voxels[i];
  } else {
    HashEntry entry = hash_table.GetEntry(block_pos);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetVoxelRef-(%d %d %d): should never reach here!\n",entry.pos.x,entry.pos.y,entry.pos.z);
    }
    uint i = geometry_helper.VectorizeOffset(offset);
    return blocks[entry.ptr].voxels[i];
  }
}

__device__
inline MeshUnit &GetMeshUnitRefScale(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const int3 query_block_pos,
    BlockArray &blocks,
    const HashTable &hash_table,
    ScaleTable &scale_table,
    GeometryHelper &geometry_helper, int curr_scale, int query_scale
) {
    //int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
    //uint3 offset = geometry_helper.VoxelToOffset(query_block_pos, voxel_pos, query_scale);
/*
    if(curr_entry.pos.x==4&&curr_entry.pos.y==2&&curr_entry.pos.z==1&&query_scale>0&&voxel_pos.x==38&&voxel_pos.y==20&&voxel_pos.z==24){
        int3 origin_voxel=geometry_helper.BlockToVoxel(curr_entry.pos);
        int3 shell_voxel = voxel_pos;
        uint3 shell_offset = geometry_helper.VoxelToOffset(curr_entry.pos, shell_voxel, scale_table.GetScale(curr_entry.pos).scale);

      printf("fill:%d %d %d voxel:%d %d %d query_block_pos:%d %d %d(%d) scale:%d ptr:%d  vertex:%d %d %d\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z,query_block_pos.x,query_block_pos.y,query_block_pos.z,query_scale,scale_table.GetScale(curr_entry.pos).scale,curr_entry.ptr,blocks[curr_entry.ptr].shells_meshes[0].vertex_ptrs[0],blocks[curr_entry.ptr].shells_meshes[0].vertex_ptrs[1],blocks[curr_entry.ptr].shells_meshes[0].vertex_ptrs[2]);
    }
      */  
  
    if(query_scale<0){
        int3 origin_voxel = geometry_helper.BlockToVoxel(curr_entry.pos);
        int3 shell_voxel = voxel_pos;
        uint3 shell_offset = geometry_helper.VoxelToOffset(curr_entry.pos, shell_voxel, scale_table.GetScale(curr_entry.pos).scale);
        if(geometry_helper.IsInShell(shell_offset)){
            uint i = geometry_helper.VectorizeShellOffset(shell_offset);

            return blocks[curr_entry.ptr].shells_meshes[i];
        }
    }
    
    int3 ancest_pos = scale_table.GetAncestor(curr_entry.pos);
    int this_scale = scale_table.GetScale(ancest_pos).scale;
 
   
  //if (curr_entry.pos == block_pos) {
    if(curr_entry.pos == query_block_pos) {
      uint3 offset = geometry_helper.VoxelToOffset(ancest_pos, voxel_pos, this_scale);
      uint i = geometry_helper.VectorizeOffset(offset);
   // if(query_block_pos.x!=block_pos.x||query_block_pos.y!=block_pos.y||query_block_pos.z!=block_pos.z)
     // printf("wrong:%d %d %d-%d %d %d\n",query_block_pos.x,query_block_pos.y,query_block_pos.z,block_pos.x,block_pos.y,block_pos.z);
    //printf("same:%d %d %d ptr:%d i:%d\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,curr_entry.ptr,i);
      return blocks[curr_entry.ptr].mesh_units[i];
    } else {
      uint3 offset = geometry_helper.VoxelToOffset(query_block_pos, voxel_pos, query_scale);
      HashEntry entry = hash_table.GetEntry(query_block_pos);
      //printf("curr_scale:%d query_scale:%d\n",curr_scale,query_scale);
      uint i = geometry_helper.VectorizeOffset(offset);
      //if(curr_entry.pos.x==4&&curr_entry.pos.y==2&&curr_entry.pos.z==1&&voxel_pos.x==38&&voxel_pos.y==20&&voxel_pos.z==24){
        //   printf("fill:%d %d %d voxel:%d %d %d query_block_pos:%d %d %d(%d) scale:%d ptr:%d i:%d offset:%d %d %d vertex:%d %d %d\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z,query_block_pos.x,query_block_pos.y,query_block_pos.z,query_scale,scale_table.GetScale(curr_entry.pos).scale,curr_entry.ptr,i,offset.x,offset.y,offset.z,blocks[curr_entry.ptr].shells_meshes[i].vertex_ptrs[0],blocks[curr_entry.ptr].shells_meshes[i].vertex_ptrs[1],blocks[curr_entry.ptr].shells_meshes[i].vertex_ptrs[2]);
   
     // }
      if (entry.ptr == FREE_ENTRY){
        printf("GetMeshRef2-(%d %d %d)->(%d %d %d) offset:%d %d %d scale:(%d->%d) compare pos:%d %d %d<->%d %d %d: should never reach here!\n",entry.pos.x,entry.pos.y,entry.pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z,offset.x,offset.y,offset.z,curr_scale,query_scale,curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,query_block_pos.x,query_block_pos.y,query_block_pos.z);
        int3 origin_voxel = geometry_helper.BlockToVoxel(curr_entry.pos);
        int3 shell_voxel = voxel_pos;
        uint3 shell_offset = geometry_helper.VoxelToOffset(curr_entry.pos,shell_voxel,scale_table.GetScale(curr_entry.pos).scale);
        if(geometry_helper.IsInShell(shell_offset)){
            uint i = geometry_helper.VectorizeOffset(shell_offset);
            printf("out shell:%d %d %d\n",shell_offset.x,shell_offset.y,shell_offset.z);
        }
    }
    return blocks[entry.ptr].mesh_units[i];
  }
}

__device__
inline MeshUnit &GetMeshUnitRef(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

  if (curr_entry.pos == block_pos) {
    uint i = geometry_helper.VectorizeOffset(offset);
    return blocks[curr_entry.ptr].mesh_units[i];
  } else {
    HashEntry entry = hash_table.GetEntry(block_pos);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetMeshRef-(%d %d %d): should never reach here!\n",entry.pos.x,entry.pos.y,entry.pos.z);
    }
    uint i = geometry_helper.VectorizeOffset(offset);
    return blocks[entry.ptr].mesh_units[i];
  }
}

// function:
// block-pos @param curr_entry -> voxel-pos @param voxel_local_pos
// get SDF in @param blocks
// with the help of @param hash_table and geometry_helper
//

__device__
inline bool GetVoxelValue(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    ScaleTable &scale_table,
    GeometryHelper &geometry_helper,
    Voxel* voxel,
    int3 origin_pos,
    float3* corner,
    int* check) {
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  ScaleInfo scaleinfo = scale_table.GetScale(block_pos);
  int3 ancest_pos = scaleinfo.ancestor;
  int this_scale = scale_table.GetScale(ancest_pos).scale;

  uint3 offset = geometry_helper.VoxelToOffset(ancest_pos, voxel_pos, this_scale);
    
  *corner = geometry_helper.VoxelToWorld(geometry_helper.BlockToVoxel(ancest_pos) + make_int3(offset) * pow(2,this_scale-1));

 // if(geometry_helper.VectorizeOffset(offset)>=512&&geometry_helper.VectorizeOffset(offset)<0)
  //      printf("offset:%d %d %d(%d)\n",offset.x,offset.y,offset.z,geometry_helper.VectorizeOffset(offset));
   // if(scale_table.GetScale(curr_entry.pos).scale!=this_scale&&scale_table.GetScale(curr_entry.pos).scale>=4)
    //    printf("here:%d %d %d(%d)->pos:%d %d %d(%d)[%d %d %d(%d)] ptr:%d\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,scale_table.GetScale(curr_entry.pos).scale,ancest_pos.x,ancest_pos.y,ancest_pos.z,this_scale,block_pos.x,block_pos.y,block_pos.z,scale_table.GetScale(block_pos).scale,curr_entry.ptr);
   
  if(this_scale<0){
      *check = 2;
      int3 origin_voxel = geometry_helper.BlockToVoxel(curr_entry.pos);
      int3 shell_voxel = voxel_pos;
      uint3 shell_offset = geometry_helper.VoxelToOffset(curr_entry.pos, shell_voxel, scale_table.GetScale(curr_entry.pos).scale);
      if(geometry_helper.IsInShell(shell_offset)){
          uint i = geometry_helper.VectorizeShellOffset(shell_offset);
          *voxel = blocks[curr_entry.ptr].shells[i];
          //if(scale_table.GetScale(curr_entry.pos).scale>=4&&voxel_pos.x==0&&voxel_pos.y==-136&&voxel_pos.z==8)
           // printf("fill:%d %d %d offset:%d %d %d(%d) scale:%d(%d) sdf:%f weight:%f\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,shell_offset.x,shell_offset.y,shell_offset.z,i,scale_table.GetScale(curr_entry.pos).scale,this_scale,voxel->sdf,voxel->inv_sigma2);
          return true;
      }
     // if(scale_table.GetScale(curr_entry.pos).scale>=4)
     //   printf("wrong1:%d %d %d(%d %d %d)->visit:%d %d %d->%d %d %d(%d)\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,shell_offset.x,shell_offset.y,shell_offset.z,block_pos.x,block_pos.y,block_pos.z,ancest_pos.x,ancest_pos.y, ancest_pos.z,this_scale);
      return false;
  }
   
  if (curr_entry.pos == block_pos) {
      if(curr_entry.ptr<0)
        printf("ptr1:%d i:%d\n",curr_entry.ptr);
    uint i = geometry_helper.VectorizeOffset(offset);
    *voxel = blocks[curr_entry.ptr].voxels[i];
    // if(block_pos.x==21&&block_pos.y==25&&block_pos.z==0&&voxel->sdf!=0){
     // if(scale_table.GetScale(curr_entry.pos).scale>1)
      //  printf("scale:%d(%d %d %d) -> %d(%d %d %d)\n",scale_table.GetScale(curr_entry.pos).scale,curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,this_scale,ancest_pos.x,ancest_pos.y,ancest_pos.z);
    // if(block_pos.z==0&&block_pos.x>7&&block_pos.y>7&&scale_table.GetScale(curr_entry.pos).scale>this_scale){
      //  printf("pos:%d %d %d voxel:%d %d %d i:%d sdf:%f weight:%f ptr:%d scale:%d ancest:%d %d %d\n",block_pos.x,block_pos.y,block_pos.z,voxel_pos.x,voxel_pos.y,voxel_pos.z,i,voxel->sdf,voxel->inv_sigma2,curr_entry.ptr,this_scale, ancest_pos.x,ancest_pos.y,ancest_pos.z);
       // *check = 1;
    // }

  } else {
    HashEntry entry = hash_table.GetEntry(ancest_pos);
    //if(scale_table.GetScale(curr_entry.pos).scale!=this_scale&&scale_table.GetScale(curr_entry.pos).scale>=4)
    //    printf("here:%d %d %d(%d)->pos:%d %d %d(%d)[%d %d %d(%d)] ptr:%d\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,scale_table.GetScale(curr_entry.pos).scale,ancest_pos.x,ancest_pos.y,ancest_pos.z,this_scale,block_pos.x,block_pos.y,block_pos.z,scale_table.GetScale(block_pos).scale,entry.ptr);
    if (entry.ptr == FREE_ENTRY) {
       // if(scale_table.GetScale(curr_entry.pos).scale>=4)
          //printf("wrong2:%d %d %d\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z);
        return false;
    }
    uint i = geometry_helper.VectorizeOffset(offset);
    //if(i<0||i>=512||entry.ptr<0)
     //  printf("ptr2:%d i:%d(%d %d %d:%d %d %d->%d %d %d[%d])\n",entry.ptr,i,offset.x,offset.y,offset.z,voxel_pos.x,voxel_pos.y,voxel_pos.z,ancest_pos.x,ancest_pos.y,ancest_pos.z,this_scale);
    *voxel = blocks[entry.ptr].voxels[i];
  }
  if(voxel->inv_sigma2<=0){
    int3 origin_voxel = geometry_helper.BlockToVoxel(curr_entry.pos);
    int3 shell_voxel = voxel_pos;
    uint3 shell_offset = geometry_helper.VoxelToOffset(curr_entry.pos, shell_voxel, scale_table.GetScale(curr_entry.pos).scale);
    if(geometry_helper.IsInShell(shell_offset)){
        uint i = geometry_helper.VectorizeShellOffset(shell_offset);
        *voxel = blocks[curr_entry.ptr].shells[i];
        //if(scale_table.GetScale(curr_entry.pos).scale>=4&&voxel_pos.x==0&&voxel_pos.y==-136&&voxel_pos.z==8)
       //   printf("fill:%d %d %d offset:%d %d %d(%d) scale:%d(%d) sdf:%f weight:%f\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,shell_offset.x,shell_offset.y,shell_offset.z,i,scale_table.GetScale(curr_entry.pos).scale,this_scale,voxel->sdf,voxel->inv_sigma2);
        return true;
    }
     // if(scale_table.GetScale(curr_entry.pos).scale>=4)
     //   printf("wrong1:%d %d %d(%d %d %d)->visit:%d %d %d->%d %d %d(%d)\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,shell_offset.x,shell_offset.y,shell_offset.z,block_pos.x,block_pos.y,block_pos.z,ancest_pos.x,ancest_pos.y, ancest_pos.z,this_scale);
    return false;
  }
  // if(scale_table.GetScale(curr_entry.pos).scale>=4&&voxel_pos.x==0&&voxel_pos.y==-136&&voxel_pos.z==8)
     //   printf("fill2:%d %d %d(from %d %d %d) scale:%d(%d) sdf:%f weight:%f\n",curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,origin_pos.x,origin_pos.y,origin_pos.z,scale_table.GetScale(curr_entry.pos).scale,this_scale,voxel->sdf,voxel->inv_sigma2);
       
  return true;
}

__device__
inline bool GetVoxelValue(
    const float3 world_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    Voxel* voxel
) {
  int3 voxel_pos = geometry_helper.WorldToVoxeli(world_pos);
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

  HashEntry entry = hash_table.GetEntry(block_pos);
  if (entry.ptr == FREE_ENTRY) {
    voxel->sdf = 0;
    voxel->inv_sigma2 = 0;
    voxel->color = make_uchar3(0,0,0);
    return false;
  } else {
    uint i = geometry_helper.VectorizeOffset(offset);
    const Voxel& v = blocks[entry.ptr].voxels[i];
    voxel->sdf = v.sdf;
    voxel->inv_sigma2 = v.inv_sigma2;
    voxel->color = v.color;
    voxel->a = v.a;
    voxel->b = v.b;
    return true;
  }
}

#endif //MESH_HASHING_SPATIAL_QUERY_H
