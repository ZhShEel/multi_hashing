//
// Created by wei on 17-10-21.
//

#ifndef CORE_HASH_ENTRY_H
#define CORE_HASH_ENTRY_H

#include "core/common.h"
#include "helper_math.h"
#include <stdio.h>

struct __ALIGN__(8) HashEntry {
  int3	pos;		   // block position (lower left corner of SDFBlock))
  int		ptr;	     // pointer into heap to SDFBlock
  uint	offset;		 // offset for linked lists
  float join_value; //roughness + measurement
  bool  will_join;
  bool  join_signal; 
  bool  will_delete;
  bool  mutex;

  // __device__
  // void operator=(const struct HashEntry& e) {
  //   ((long long*)this)[0] = ((const long long*)&e)[0];
  //   ((long long*)this)[1] = ((const long long*)&e)[1];
  //   ((int*)this)[4]       = ((const int*)&e)[4];
  // }

  __device__
  void operator=(const struct HashEntry& e){
    pos = e.pos;
    ptr = e.ptr;
    offset = e.offset;
    join_value = e.join_value;
    will_join = e.will_join;
    join_signal = e.join_signal;
    will_delete = e.will_delete;
    mutex = e.mutex;
  }

  __device__
  void Clear() {
    pos    = make_int3(0);
    ptr    = FREE_ENTRY;
    offset = 0;
    join_value = -1;  //<0, not join
    will_join = 0;
    join_signal = 0;
    will_delete = 0;
    mutex = 0;
  }

};

struct __ALIGN__(8) ScaleInfo
{
  int3 pos;
  int3 ancestor;
  int scale;
  uint offset;
  float curvature;

  // __device__
  // void operator=(const struct ScaleInfo& s){
  //   ((long long*)this)[0] = ((const long long*)&s)[0];
  //   ((long long*)this)[1] = ((const long long*)&s)[1];
  //   ((int*)this)[4]       = ((const int*)&s)[4];
  // }

  __device__
  void operator=(const struct ScaleInfo& s){
    pos = s.pos;
    ancestor = s.ancestor;
    scale = s.scale;
    offset = s.offset;
    curvature = s.curvature;
  }

  __device__
  void Clear() {
    pos = make_int3(0);
    ancestor = make_int3(0);
    scale = -1;
    offset = 0;
    curvature = -999999;
  }
};

#endif //MESH_HASHING_HASH_ENTRY_H
