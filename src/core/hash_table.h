//
// Created by wei on 17-4-28.
//

#ifndef CORE_HASH_TABLE_H
#define CORE_HASH_TABLE_H

#include "helper_cuda.h"
#include "helper_math.h"

#include "core/common.h"
#include "core/params.h"
#include "core/hash_entry.h"
#include "geometry/geometry_helper.h"

class ScaleTable {
public:
  uint bucket_count;
  uint bucket_size;
  uint entry_count;
  uint value_capacity;
  uint linked_list_size;
  float curvature_th;
  float cd_th;

  __host__ ScaleTable() = default;

  __host__ explicit ScaleTable(const ScaleParams &params);

  __host__ void Alloc(const ScaleParams &params);

  __host__ void Free();

  __host__ void Resize(const ScaleParams &params);

  __host__ void Reset();

  __host__ void ResetMutexes();

  __host__ __device__ ScaleInfo &scale(uint i){
    return scale_[i];
  }

ScaleInfo* scale_;
private:
  bool is_allocated_on_gpu_ = false;
  
  uint *heap_;

  uint *heap_counter_;

  // ScaleInfo *scale_;

  int *bucket_mutexes_;

  #ifdef __CUDACC__
    public:

    __device__ 
    void UpdateCurvature(const int3 &pos, float s){
      uint bucket_idx = HashBucketForBlockPos(pos);
      uint bucket_first_scale_idx = bucket_idx * bucket_size;

      for(int i=0;i<bucket_size;++i){
        ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
        if(IsPosAllocated(pos,curr_scale)){
          float tmp = curr_scale.curvature;
          scale_[i+bucket_first_scale_idx].curvature = max(tmp,s);
          return;
        }
      }
      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
          if(IsPosAllocated(pos,curr_scale)){
            float tmp = curr_scale.curvature;
            scale_[i].curvature = max(tmp,s);
            return;
          } 
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
        return;
    }

    __device__ 
    void AddScale(const int3 &pos){
      uint bucket_idx = HashBucketForBlockPos(pos);
      uint bucket_first_scale_idx = bucket_idx * bucket_size;

      for(int i=0;i<bucket_size;++i){
        ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
        // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          // printf("add1:%d %d %d <-> %d %d %d(%d+%d=%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i,bucket_first_scale_idx,i+bucket_first_scale_idx);
        if(IsPosAllocated(pos,curr_scale)){
          scale_[i+bucket_first_scale_idx].scale++;
          return;
        }
      }
      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
          // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          //   printf("add2:%d %d %d <-> %d %d %d(%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i);

          if(IsPosAllocated(pos,curr_scale)){
            scale_[i].scale++;
            return;
          } 
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
        return;
    }


    __device__
    void AllocScale(const int3& pos) {
      uint bucket_idx             = HashBucketForBlockPos(pos);   //hash bucket
      uint bucket_first_scale_idx = bucket_idx * bucket_size; //hash position
      /// 1. Try GetEntry, meanwhile collect an empty entry potentially suitable
      int empty_scale_idx = -1;
      for (uint j = 0; j < bucket_size; j++) {
        uint i = j + bucket_first_scale_idx;
        const ScaleInfo curr_scale = scale_[i];
        if (IsPosAllocated(pos, curr_scale)) {
          //  printf("alloc pos:%d %d %d->curr:%d %d %d scale:%d\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,curr_scale.scale);
          return;
        }

        if (empty_scale_idx == -1 && curr_scale.scale == -1) {
          empty_scale_idx = i;
        }
      }

      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
      uint i = bucket_last_scale_idx;
      uint iter = 0;
      for (iter = 0; iter < linked_list_size; ++iter) {
        ScaleInfo curr_scale = scale_[i];

        if (IsPosAllocated(pos, curr_scale)) {
          return;
        }
        if (curr_scale.offset == 0) {
          break;
        }
        i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
      }
      const uint existing_linked_list_size = iter + 1;
 
      /// 2. NOT FOUND, Allocate
      // printf("alloc1:%d %d %d\n",pos.x,pos.y,pos.z);

      if (empty_scale_idx != -1) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) { 
          ScaleInfo& scale_info = scale_[empty_scale_idx];
          scale_info.pos = pos;
          scale_info.ancestor = pos;
          // scale_info.scale = Alloc();
          scale_info.scale = 1;
          scale_info.offset = NO_OFFSET;
        }
        
        return;
      }

        if (existing_linked_list_size == linked_list_size){
            return;
        }


#pragma unroll 1
      for (uint linked_list_offset = 1; linked_list_offset < linked_list_size; ++linked_list_offset) {
        if ((linked_list_offset % bucket_size) == 0) continue;

        i = (bucket_last_scale_idx + linked_list_offset) % (entry_count);

        ScaleInfo& curr_scale = scale_[i];
        if (curr_scale.scale == -1) {
          int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            uint alloc_bucket_idx = i / bucket_size;
            lock = atomicExch(&bucket_mutexes_[alloc_bucket_idx], LOCK_ENTRY);
            if (lock != LOCK_ENTRY) {
              ScaleInfo& bucket_last_scale = scale_[bucket_last_scale_idx];
              ScaleInfo& scale_info = scale_[i];
              scale_info.pos = pos;
              scale_info.ancestor = pos;
              scale_info.offset = bucket_last_scale.offset; // pointer assignment in linked list
              // scale_info.scale = Alloc(); //memory alloc
              scale_info.scale = 1;

              // Not sure if it is ok to directly assign to reference
              bucket_last_scale.offset = linked_list_offset;
              scale_[bucket_last_scale_idx] = bucket_last_scale;
            }
          }
          return; //bucket was already locked
        }
      }
    }

    __device__
    void AllocAncestedScale(const int3& pos, int s, int3 ancest) {
      uint bucket_idx             = HashBucketForBlockPos(pos);   //hash bucket
      uint bucket_first_scale_idx = bucket_idx * bucket_size; //hash position
      /// 1. Try GetEntry, meanwhile collect an empty entry potentially suitable
      int empty_scale_idx = -1;
      for (uint j = 0; j < bucket_size; j++) {
        uint i = j + bucket_first_scale_idx;
        const ScaleInfo curr_scale = scale_[i];
        if (IsPosAllocated(pos, curr_scale)) {
          //  printf("alloc pos:%d %d %d->curr:%d %d %d scale:%d\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,curr_scale.scale);
          return;
        }

        if (empty_scale_idx == -1 && curr_scale.scale == -1) {
          empty_scale_idx = i;
        }
      }

      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
      uint i = bucket_last_scale_idx;
      uint iter = 0;
      for (iter = 0; iter < linked_list_size; ++iter) {
        ScaleInfo curr_scale = scale_[i];

        if (IsPosAllocated(pos, curr_scale)) {
          return;
        }
        if (curr_scale.offset == 0) {
          break;
        }
        i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
      }
      const uint existing_linked_list_size = iter + 1;
 
      /// 2. NOT FOUND, Allocate
      // printf("alloc1:%d %d %d\n",pos.x,pos.y,pos.z);

      if (empty_scale_idx != -1) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) { 
          ScaleInfo& scale_info = scale_[empty_scale_idx];
          scale_info.pos = pos;
          scale_info.ancestor = ancest;
          // scale_info.scale = Alloc();
          scale_info.scale = s;
          scale_info.offset = NO_OFFSET;
        }
       // printf("1\n");
        return;
      }

        if (existing_linked_list_size == linked_list_size){
            return;
        }


#pragma unroll 1
      for (uint linked_list_offset = 1; linked_list_offset < linked_list_size; ++linked_list_offset) {
        if ((linked_list_offset % bucket_size) == 0) continue;

        i = (bucket_last_scale_idx + linked_list_offset) % (entry_count);

        ScaleInfo& curr_scale = scale_[i];
        if (curr_scale.scale == -1) {
          int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            uint alloc_bucket_idx = i / bucket_size;
            lock = atomicExch(&bucket_mutexes_[alloc_bucket_idx], LOCK_ENTRY);
            if (lock != LOCK_ENTRY) {
              ScaleInfo& bucket_last_scale = scale_[bucket_last_scale_idx];
              ScaleInfo& scale_info = scale_[i];
              scale_info.pos = pos;
              scale_info.ancestor = ancest;
              scale_info.offset = bucket_last_scale.offset; // pointer assignment in linked list
              // scale_info.scale = Alloc(); //memory alloc
              scale_info.scale = s;

              // Not sure if it is ok to directly assign to reference
              bucket_last_scale.offset = linked_list_offset;
              scale_[bucket_last_scale_idx] = bucket_last_scale;
            }
          }
            printf("2\n");
          return; //bucket was already locked
        }
      }
    }

    __device__
    int3 GetAncestorUnit(const int3& pos){
      uint bucket_idx = HashBucketForBlockPos(pos);
      uint bucket_first_scale_idx = bucket_idx * bucket_size;

      for(int i=0;i<bucket_size;++i){
        ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
        // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          // printf("add1:%d %d %d <-> %d %d %d(%d+%d=%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i,bucket_first_scale_idx,i+bucket_first_scale_idx);
        if(IsPosAllocated(pos,curr_scale)){
         // if(curr_scale.scale==2)
          //  printf("curr:%d %d %d->%d %d %d\n",pos.x,pos.y,pos.z,curr_scale.ancestor.x,curr_scale.ancestor.y,curr_scale.ancestor.z);
          return curr_scale.ancestor;
        }
      }
      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
          // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          //   printf("add2:%d %d %d <-> %d %d %d(%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i);

          if(IsPosAllocated(pos,curr_scale)){
            //if(curr_scale.scale==2)
            //  printf("curr:%d %d %d->%d %d %d\n",pos.x,pos.y,pos.z,curr_scale.ancestor.x,curr_scale.ancestor.y,curr_scale.ancestor.z);
            return curr_scale.ancestor;
          } 
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
        //If not found
        return pos;
    }

    __device__
    int3 GetAncestor(const int3& pos) {
        int3 curr_pos = pos;
        int travel_cnt=0;
        while(1){
            int3 prev_ancestor = GetAncestorUnit(curr_pos);
            //int3 prev_prev_ancestor = GetAncestorUnit(prev_ancestor);
            // if(prev_ancestor == make_int3(0))
            //     return prev_ancestor;
            //if(prev_ancestor == GetAncestorUnit(prev_ancestor))
            if(prev_ancestor == curr_pos){
                //if(travel_cnt>=GetScale(pos).scale&&GetScale(pos).scale>0)
                //if(travel_cnt>0)
                //  printf("end:%d pos:%d %d %d->%d %d %d scale:%d\n",travel_cnt,pos.x,pos.y,pos.z,prev_ancestor.x,prev_ancestor.y,prev_ancestor.z,GetScale(pos).scale);
                return curr_pos;
            }
            //if(pos.x==15&&pos.y==19&&pos.z==0)

             // printf("travel %d:%d %d %d->%d %d %d(%d %d %d)\n",
             //      travel_cnt,curr_pos.x,curr_pos.y,curr_pos.z,prev_ancestor.x,prev_ancestor.y,prev_ancestor.z,
             //           pos.x,pos.y,pos.z);
            travel_cnt++;
            curr_pos = prev_ancestor;
        }
    }

    __device__
    void SetAncestor(const int3& pos, int3 ancest){
      uint bucket_idx = HashBucketForBlockPos(pos);
      uint bucket_first_scale_idx = bucket_idx * bucket_size;

      for(int i=0;i<bucket_size;++i){
        ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
         //if(pos.x!=0&&pos.y!=0&&pos.z!=0)
         //  printf("add1:%d %d %d <-> %d %d %d(%d+%d=%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i,bucket_first_scale_idx,i+bucket_first_scale_idx);
        if(IsPosAllocated(pos,curr_scale)){
          //if(pos.x!=curr_scale.ancestor.x||pos.y!=curr_scale.ancestor.y||pos.z!=curr_scale.ancestor.z)          
            //printf("wrong1:%d %d %d->%d %d %d\n",pos.x,pos.y,pos.z,curr_scale.ancestor.x,curr_scale.ancestor.y,curr_scale.ancestor.z);
          scale_[i+bucket_first_scale_idx].ancestor = ancest;
          return;
        }
      }
      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
          // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          //   printf("add2:%d %d %d <-> %d %d %d(%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i);

          if(IsPosAllocated(pos,curr_scale)){
            //if(pos.x!=curr_scale.ancestor.x||pos.y!=curr_scale.ancestor.y||pos.z!=curr_scale.ancestor.z)
              // printf("wrong2:%d %d %d->%d %d %d\n",pos.x,pos.y,pos.z,curr_scale.ancestor.x,curr_scale.ancestor.y,curr_scale.ancestor.z);
            scale_[i].ancestor = ancest;
            return;
          } 
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
       // AllocAncestedScale(pos,1,ancest);
       //printf("no:%d %d %d->%d %d %d\n",pos.x,pos.y,pos.z,ancest.x,ancest.y,ancest.z);
        return;
    }
    __device__
      ScaleInfo GetScale(const int3& pos) {
        // int3 ancest_pos = GetAncestor(pos);
        int3 ancest_pos = pos;
        while(1){
            int3 prev_ancestor = GetAncestorUnit(ancest_pos);
            // if(prev_ancestor == make_int3(0))
            //     return prev_ancestor;
            if(prev_ancestor == GetAncestorUnit(prev_ancestor)){
                ancest_pos = prev_ancestor;
                break;
            }
            ancest_pos = prev_ancestor;
        }

        uint bucket_idx = HashBucketForBlockPos(ancest_pos);
        uint bucket_first_scale_idx = bucket_idx * bucket_size;
        
        for(int i=0;i<bucket_size;++i){
          ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
          if(IsPosAllocated(ancest_pos,curr_scale)){
            // printf("find %d times(bucket_size:%d).\n",i,bucket_size);
            return curr_scale;
          }
          // if(i>6)
          //   printf("see:pos:%d %d %d curr_scale:%d %d %d ptr:%d(%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,curr_scale.scale,i);
        }

        const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
          if(IsPosAllocated(ancest_pos,curr_scale)){ 
            return curr_scale;
          }
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
        // printf("not find\n");
        ScaleInfo not_find_info;
        not_find_info.pos = pos;
        not_find_info.ancestor = pos;
        not_find_info.scale = -1;   //need to rethink(scale:from 1 to upper)
        not_find_info.offset = 0;
        return not_find_info;
      }
    __device__ 
    void SetScale(const int3 &pos, const int number){

      AllocScale(pos);

      uint bucket_idx = HashBucketForBlockPos(pos);
      uint bucket_first_scale_idx = bucket_idx * bucket_size;
      for(int i=0;i<bucket_size;++i){
        ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
        // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          // printf("add1:%d %d %d <-> %d %d %d(%d+%d=%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i,bucket_first_scale_idx,i+bucket_first_scale_idx);
        // printf("%d %d %d - %d %d %d\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z);
        if(IsPosAllocated(pos,curr_scale)){
          // printf("found1:%d %d %d(%d %d %d)\n",pos.x,pos.y,pos.z,i,bucket_first_scale_idx,i+bucket_first_scale_idx);
          // scale_[i+bucket_first_scale_idx].scale = number;
          // printf("arrive1:%d %d %d(%d)\n",pos.x,pos.y,pos.z,number);
          atomicExch(&(scale_[i+bucket_first_scale_idx].scale), number);
          // printf("check:%d (%d %d %d)%d\n",scale_[i+bucket_first_scale_idx].scale,pos.x,pos.y,pos.z,number);
          return;
        }
      }
      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
          // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          //   printf("add2:%d %d %d <-> %d %d %d(%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i);

          if(IsPosAllocated(pos,curr_scale)){
            // printf("found2:%d %d %d\n",pos.x,pos.y,pos.z);
            // printf("arrive2:%d %d %d(%d)\n",pos.x,pos.y,pos.z,number);
            atomicExch(&(scale_[i].scale), number);
            return;
          } 
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
        // printf("never found.\n");
        return;
    }
     __device__ 
    bool SetScaleAndAncestor(const int3 &pos, const int number, int3 ancest){

      AllocScale(pos);

      uint bucket_idx = HashBucketForBlockPos(pos);
      uint bucket_first_scale_idx = bucket_idx * bucket_size;
      for(int i=0;i<bucket_size;++i){
        ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
        if(IsPosAllocated(pos,curr_scale)){
           atomicExch(&(scale_[i+bucket_first_scale_idx].scale), number);
           scale_[i+bucket_first_scale_idx].ancestor = ancest;
           return;
        }
      }
      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
      
          if(IsPosAllocated(pos,curr_scale)){
            atomicExch(&(scale_[i].scale), number);
            scale_[i].ancestor = ancest; 
            return;
          } 
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
        // printf("never found.\n");
        return;
    }

    __device__
    bool FreeEntry(const int3& pos) {
      uint bucket_idx = HashBucketForBlockPos(pos); //hash bucket
      uint bucket_first_scale_idx = bucket_idx * bucket_size;   //hash position

      for (uint j = 0; j < (bucket_size - 1); j++) {
        uint i = j + bucket_first_scale_idx;
        const ScaleInfo& curr = scale_[i];

        if (IsPosAllocated(pos, curr)) {
            int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
            if (lock != LOCK_ENTRY) {
              scale_[i].Clear();
              return true;
            } else {
              return false;
            }
          }
        }

      // Init with linked list traverse
      const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
      int i = bucket_last_scale_idx;
      ScaleInfo& head_scale = scale_[i];

      if (IsPosAllocated(pos, head_scale)) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          head_scale.pos = make_int3(0);
          head_scale.scale = -1;
          // DO NOT RESET OFFSET
          return true;
        } else {
          return false;
        }
      }

      int prev_idx = i;
      i = (bucket_last_scale_idx + head_scale.offset) % (entry_count);
#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        ScaleInfo &curr = scale_[i];

        if (IsPosAllocated(pos, curr)) {
          int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            int prev_bucket_idx = prev_idx / bucket_size;
            lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
            if (prev_bucket_idx != bucket_idx || lock != LOCK_ENTRY) {
              scale_[prev_idx].offset = curr.offset;
              curr.Clear();
              return true;
            } else {
            return false;
            }
          }
        }

        if (curr.offset == 0) { //we have found the end of the list
          return false; //should actually never happen because we need to find that guy before
        }

        prev_idx = i;
        i = (bucket_last_scale_idx + curr.offset) % (entry_count);
      }
      return false;
    }



    private:
    //! see Teschner et al. (but with correct prime values)
      __device__
      uint HashBucketForBlockPos(const int3& pos) const {
        const int p0 = 73856093;
        const int p1 = 19349669;
        const int p2 = 83492791;

        int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2))
                  % bucket_count;
        if (res < 0) res += bucket_count;
        return (uint) res;
      }

      __device__
      bool IsPosAllocated(const int3& pos, const ScaleInfo& scale_info) const {
        return pos.x == scale_info.pos.x
            && pos.y == scale_info.pos.y
            && pos.z == scale_info.pos.z
            && scale_info.scale != -1;
      }

      __device__
      uint Alloc() {
        uint addr = atomicSub(&heap_counter_[0], 1);
        if (addr < MEMORY_LIMIT) {
          printf("Scale Table Memory nearly exhausted! %d -> %d\n", addr, heap_[addr]);
        }
        return heap_[addr];
      }

      __device__
      void Free(uint ptr) {
        uint addr = atomicAdd(&heap_counter_[0], 1);
        heap_[addr + 1] = ptr;
      }
  #endif
};

class HashTable {
public:
  /// Parameters
  uint bucket_count;
  uint bucket_size;
  uint entry_count;
  uint value_capacity;
  uint linked_list_size;

  __host__ HashTable() = default;

  __host__ explicit HashTable(const HashParams &params);
  // ~HashTable();
  __host__ void Alloc(const HashParams &params);

  __host__ void Free();

  __host__ void Resize(const HashParams &params);

  __host__ void Reset();

  __host__ void ResetMutexes();

  __host__ void ResetNeighborMutexes();

  __host__ __device__ HashEntry &entry(uint i) {
    return entries_[i];
  }
  //__host__ void Debug();

  /////////////////
  // Device part //

private:
  bool is_allocated_on_gpu_ = false;
  // @param array
  uint *heap_;             /// index to free values
  // @param read-write element
  uint *heap_counter_;     /// single element; used as an atomic counter (points to the next free block)

  // @param array
  HashEntry *entries_;          /// hash entries that stores pointers to sdf values
  // @param array
  int *bucket_mutexes_;   /// binary flag per hash bucket; used for allocation to atomically lock a bucket

#ifdef __CUDACC__
  public:
    __device__
    HashEntry GetEntry(const int3& pos) const {
      uint bucket_idx             = HashBucketForBlockPos(pos);
      uint bucket_first_entry_idx = bucket_idx * bucket_size;

      HashEntry entry;
      entry.pos    = pos;
      entry.offset = 0;
      entry.ptr    = FREE_ENTRY;

      for (uint i = 0; i < bucket_size; ++i) {
        HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
        if (IsPosAllocated(pos, curr_entry)) {
          return curr_entry;
        }
        // if(i>6)
        //     printf("see:pos:%d %d %d curr_scale:%d %d %d ptr:%d(%d)\n",pos.x,pos.y,pos.z,curr_entry.pos.x,curr_entry.pos.y,curr_entry.pos.z,curr_entry.ptr,i);
      }
      /// The last entry is visted twice, but it's OK
      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      int i = bucket_last_entry_idx;

#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          return curr_entry;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      return entry;
    }


    __device__ 
    void WillDelete(const int3 &pos){
      uint bucket_idx = HashBucketForBlockPos(pos);
      uint bucket_first_entry_idx = bucket_idx * bucket_size;

      for(int i=0;i<bucket_size;++i){
        HashEntry curr_entry = entries_[i+bucket_first_entry_idx];
        // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          // printf("add1:%d %d %d <-> %d %d %d(%d+%d=%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i,bucket_first_scale_idx,i+bucket_first_scale_idx);
        if(IsPosAllocated(pos,curr_entry)){
          entries_[i+bucket_first_entry_idx].will_delete = 1;
          return;
        }
      }
      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
        int i = bucket_last_entry_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          HashEntry curr_entry = entries_[i];
          // if(pos.x!=0&&pos.y!=0&&pos.z!=0)
          //   printf("add2:%d %d %d <-> %d %d %d(%d)\n",pos.x,pos.y,pos.z,curr_scale.pos.x,curr_scale.pos.y,curr_scale.pos.z,i);

          if(IsPosAllocated(pos,curr_entry)){
            entries_[i].will_delete = 1;
            return;
          } 
          if(curr_entry.offset == 0)
            break;
          i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
        }
        return;
    }

    //pos in SDF block coordinates
    __device__
    void AllocEntry(const int3& pos) {
      uint bucket_idx             = HashBucketForBlockPos(pos);		//hash bucket
      uint bucket_first_entry_idx = bucket_idx * bucket_size;	//hash position

      /// 1. Try GetEntry, meanwhile collect an empty entry potentially suitable
      int empty_entry_idx = -1;
      for (uint j = 0; j < bucket_size; j++) {
        uint i = j + bucket_first_entry_idx;
        const HashEntry& curr_entry = entries_[i];
        if (IsPosAllocated(pos, curr_entry)) {
          return;
        }


        /// wei: should not break and alloc before a thorough searching is over:
        if (empty_entry_idx == -1 && curr_entry.ptr == FREE_ENTRY) {
          empty_entry_idx = i;
        }
      }

      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      uint i = bucket_last_entry_idx;
      uint iter = 0;
      for (iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          return;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      const uint existing_linked_list_size = iter + 1;
      
      /// 2. NOT FOUND, Allocate
      if (empty_entry_idx != -1) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          HashEntry& entry = entries_[empty_entry_idx];
          entry.pos    = pos;
          entry.ptr    = Alloc();
          entry.offset = NO_OFFSET;
        }
        return;
      }

      if (existing_linked_list_size == linked_list_size)
        return;


#pragma unroll 1
      for (uint linked_list_offset = 1; linked_list_offset < linked_list_size; ++linked_list_offset) {
        if ((linked_list_offset % bucket_size) == 0) continue;

        i = (bucket_last_entry_idx + linked_list_offset) % (entry_count);

        HashEntry& curr_entry = entries_[i];

        if (curr_entry.ptr == FREE_ENTRY) {
          int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            uint alloc_bucket_idx = i / bucket_size;
            lock = atomicExch(&bucket_mutexes_[alloc_bucket_idx], LOCK_ENTRY);
            if (lock != LOCK_ENTRY) {
              HashEntry& bucket_last_entry = entries_[bucket_last_entry_idx];
              HashEntry& entry = entries_[i];
              entry.pos    = pos;
              entry.offset = bucket_last_entry.offset; // pointer assignment in linked list
              entry.ptr    = Alloc();	//memory alloc

              // Not sure if it is ok to directly assign to reference
              bucket_last_entry.offset = linked_list_offset;
              entries_[bucket_last_entry_idx] = bucket_last_entry;
            }
          }
          return;	//bucket was already locked
        }
      }
    }
     __device__
    void AllocLockedEntry(const int3& pos) {
      uint bucket_idx             = HashBucketForBlockPos(pos);		//hash bucket
      uint bucket_first_entry_idx = bucket_idx * bucket_size;	//hash position

      /// 1. Try GetEntry, meanwhile collect an empty entry potentially suitable
      int empty_entry_idx = -1;
      for (uint j = 0; j < bucket_size; j++) {
        uint i = j + bucket_first_entry_idx;
        const HashEntry& curr_entry = entries_[i];
        if (IsPosAllocated(pos, curr_entry)) {
          return;
        }


        /// wei: should not break and alloc before a thorough searching is over:
        if (empty_entry_idx == -1 && curr_entry.ptr == FREE_ENTRY) {
          empty_entry_idx = i;
        }
      }

      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      uint i = bucket_last_entry_idx;
      uint iter = 0;
      for (iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          return;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      const uint existing_linked_list_size = iter + 1;
      
      /// 2. NOT FOUND, Allocatae
      if (empty_entry_idx != -1) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          HashEntry& entry = entries_[empty_entry_idx];
          entry.pos    = pos;
          entry.ptr    = Alloc();
          entry.offset = NO_OFFSET;
          entry.mutex = 1;
        }
        return;
      }

      if (existing_linked_list_size == linked_list_size)
        return;


#pragma unroll 1
      for (uint linked_list_offset = 1; linked_list_offset < linked_list_size; ++linked_list_offset) {
        if ((linked_list_offset % bucket_size) == 0) continue;

        i = (bucket_last_entry_idx + linked_list_offset) % (entry_count);

        HashEntry& curr_entry = entries_[i];

        if (curr_entry.ptr == FREE_ENTRY) {
          int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            uint alloc_bucket_idx = i / bucket_size;
            lock = atomicExch(&bucket_mutexes_[alloc_bucket_idx], LOCK_ENTRY);
            if (lock != LOCK_ENTRY) {
              HashEntry& bucket_last_entry = entries_[bucket_last_entry_idx];
              HashEntry& entry = entries_[i];
              entry.pos    = pos;
              entry.offset = bucket_last_entry.offset; // pointer assignment in linked list
              entry.ptr    = Alloc();	//memory alloc
              entry.mutex   = 1;

              // Not sure if it is ok to directly assign to reference
              bucket_last_entry.offset = linked_list_offset;
              entries_[bucket_last_entry_idx] = bucket_last_entry;
            }
          }
          return;	//bucket was already locked
        }
      }
    }

    //! deletes a hash entry position for a given pos index
    // returns true uppon successful deletion; otherwise returns false
    __device__
    bool FreeEntry(const int3& pos) {
      uint bucket_idx = HashBucketForBlockPos(pos);	//hash bucket
      uint bucket_first_entry_idx = bucket_idx * bucket_size;		//hash position

      for (uint j = 0; j < (bucket_size - 1); j++) {
        uint i = j + bucket_first_entry_idx;
        const HashEntry& curr = entries_[i];

        // printf("pos(%d %d):%d %d %d - %d %d %d\n",j,i,pos.x,pos.y,pos.z,curr.pos.x,curr.pos.y,curr.pos.z);
        if (IsPosAllocated(pos, curr)) {
            int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
            if (lock != LOCK_ENTRY) {
              // printf("free:%d\n",curr.ptr);
              Free(curr.ptr);
              entries_[i].Clear();
              return true;
            } else {
              return false;
            }
          }
        }

      // Init with linked list traverse
      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      int i = bucket_last_entry_idx;
      HashEntry& head_entry = entries_[i];

      if (IsPosAllocated(pos, head_entry)) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          Free(head_entry.ptr);
          head_entry.pos = make_int3(0);
          head_entry.ptr = FREE_ENTRY;
          // DO NOT RESET OFFSET
          return true;
        } else {
          return false;
        }
      }

      int prev_idx = i;
      i = (bucket_last_entry_idx + head_entry.offset) % (entry_count);
#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        HashEntry &curr = entries_[i];

        if (IsPosAllocated(pos, curr)) {
          int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            int prev_bucket_idx = prev_idx / bucket_size;
            lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
            if (prev_bucket_idx != bucket_idx || lock != LOCK_ENTRY) {
              entries_[prev_idx].offset = curr.offset;
              // printf("free:%d\n",curr.ptr);
              Free(curr.ptr);
              curr.Clear();
              return true;
            } else {
            return false;
            }
          }
        }

        if (curr.offset == 0) {	//we have found the end of the list
          return false;	//should actually never happen because we need to find that guy before
        }

        prev_idx = i;
        i = (bucket_last_entry_idx + curr.offset) % (entry_count);
      }
      return false;
    }

    __device__
    void SetWillJoinTrue(const int3& pos) const{
      uint bucket_idx             = HashBucketForBlockPos(pos);
      uint bucket_first_entry_idx = bucket_idx * bucket_size;
      for (uint i = 0; i < bucket_size; ++i) {
        HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i+bucket_first_entry_idx].will_join = 1;
          return;
        }
      }

      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      int i = bucket_last_entry_idx;

#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i].will_join = 1;
          return;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      return;

    }
    __device__
    void SetWillJoinFalse(const int3& pos) const{
      uint bucket_idx             = HashBucketForBlockPos(pos);
      uint bucket_first_entry_idx = bucket_idx * bucket_size;
      for (uint i = 0; i < bucket_size; ++i) {
        HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i+bucket_first_entry_idx].will_join = 0;
          return;
        }
      }

      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      int i = bucket_last_entry_idx;

#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i].will_join = 0;
          return;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      return;

    }

    __device__
    void SetJoinSignalTrue(const int3& pos) const{
      uint bucket_idx             = HashBucketForBlockPos(pos);
      uint bucket_first_entry_idx = bucket_idx * bucket_size;
      for (uint i = 0; i < bucket_size; ++i) {
        HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i+bucket_first_entry_idx].join_signal = 1;
          return;
        }
      }

      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      int i = bucket_last_entry_idx;

#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i].join_signal = 1;
          return;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      return;

    }
    __device__
    void SetJoinSignalFalse(const int3& pos) const{
      uint bucket_idx             = HashBucketForBlockPos(pos);
      uint bucket_first_entry_idx = bucket_idx * bucket_size;
      for (uint i = 0; i < bucket_size; ++i) {
        HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i+bucket_first_entry_idx].join_signal = 0;
          return;
        }
      }

      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      int i = bucket_last_entry_idx;

#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i].join_signal = 0;
          return;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      return;
    }

     __device__
    void Lock(const int3& pos) {
      uint bucket_idx             = HashBucketForBlockPos(pos);
      uint bucket_first_entry_idx = bucket_idx * bucket_size;
      for (uint i = 0; i < bucket_size; ++i) {
        HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i+bucket_first_entry_idx].mutex = 1;
          return;
        }
      }

      const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
      int i = bucket_last_entry_idx;

#pragma unroll 1
      for (uint iter = 0; iter < linked_list_size; ++iter) {
        HashEntry curr_entry = entries_[i];

        if (IsPosAllocated(pos, curr_entry)) {
          entries_[i].mutex = 1;
          return;
        }
        if (curr_entry.offset == 0) {
          break;
        }
        i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
      }
      //NOT FOUND, ALLOC TMP BLOCK AND LOCK IT, SAME AS FUNCTION "AllocLockedEntry".
      AllocLockedEntry(pos);
      //printf("no found\n");
      return;
    }
    

   
  private:
    //! see Teschner et al. (but with correct prime values)
    __device__
    uint HashBucketForBlockPos(const int3& pos) const {
      const int p0 = 73856093;
      const int p1 = 19349669;
      const int p2 = 83492791;

      int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2))
                % bucket_count;
      if (res < 0) res += bucket_count;
      return (uint) res;
    }

    __device__
    bool IsPosAllocated(const int3& pos, const HashEntry& hash_entry) const {
      return pos.x == hash_entry.pos.x
          && pos.y == hash_entry.pos.y
          && pos.z == hash_entry.pos.z
          && hash_entry.ptr != FREE_ENTRY;
    }

    __device__
    uint Alloc() {
      uint addr = atomicSub(&heap_counter_[0], 1);
      if (addr < MEMORY_LIMIT) {
        printf("HashTable Memory nearly exhausted! %d -> %d\n", addr, heap_[addr]);
      }
      return heap_[addr];
    }

    __device__
    void Free(uint ptr) {
      uint addr = atomicAdd(&heap_counter_[0], 1);
      heap_[addr + 1] = ptr;
    }
#endif
};

#endif //VH_HASH_TABLE_H
