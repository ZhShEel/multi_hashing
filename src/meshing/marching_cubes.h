//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_MARCHING_CUBES_H
#define MESH_HASHING_MARCHING_CUBES_H

#include <glog/logging.h>
#include <unordered_map>
#include <chrono>

#include <ctime>
#include "mc_tables.h"
#include "util/timer.h"
#include "engine/main_engine.h"
#include "core/collect_block_array.h"

float MarchingCubes(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    Mesh& mesh_m,
    HashTable& hash_table,
    ScaleTable& scale_table,
    GeometryHelper& geometry_helper,
    bool color_type,
    bool enable_bayesian,
    bool enable_sdf_gradient,
    float global_voxel_size);

float AlignMeshes(
	EntryArray& candidate_entries,
	BlockArray& blocks,
	Mesh& mesh_m,
	HashTable& hash_table,
	GeometryHelper& geometry_helper
	);
#endif //MESH_HASHING_MARCHING_CUBES_H
