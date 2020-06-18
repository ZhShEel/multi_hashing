//
// Created by wei on 17-10-22.
//

#ifndef CORE_COLLECT_H
#define CORE_COLLECT_H

#include "core/entry_array.h"
#include "core/hash_table.h"
#include "core/mesh.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"
#include "mapping/allocate.h"
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>

//@function
//spread curvatures of local regions to each voxel in scale hashtable
double SpreadMeasurementTerm(
    ScaleTable &scale_table,
    HashTable &hash_table,
    EntryArray &candidate_entries,
    PointCloud &pc_gpu,
    Mesh&,
    BlockArray&,
    GeometryHelper geometry_helper
);

double ComputeRoughnessTerm(
    ScaleTable &scale_table,
    HashTable &hash_table,
    EntryArray &candidate_entries,
    Mesh &mesh_m,
    BlockArray &blocks,
    GeometryHelper geometry_helper
);

void CountScaleNumberSerial(
    ScaleTable scale_table,
    HashTable hash_table
    );
//@function
//travel all entry in hash table and set join flags
double SetJoinFlags(
    ScaleTable &scale_table,
    HashTable &hash_table,
    EntryArray &candidate_entries,
    GeometryHelper geometry_helper
);

double JoinBlocks(
    ScaleTable &scale_table,
    HashTable &hash_table,
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh_,
    PointCloud,
    GeometryHelper geometry_helper
);

// @function
// Read the entries in @param hash_table
// Write to the @param candidate_entries (for parallel computation)
double CollectAllBlocks(
    HashTable &hash_table,
    ScaleTable &scale_table,
    EntryArray &candidate_entries
);

// @function
// Read the entries in @param hash_table
// Filter the positions with @param sensor info (pose and params),
//                       and @param geometry helper
// Write to the @param candidate_entries (for parallel computation)
double CollectBlocksInFrustum(
    HashTable &hash_table,
    Sensor &sensor,
    GeometryHelper &geometry_helper,
    EntryArray &candidate_entries
);

#endif //CORE_COLLECT_H
