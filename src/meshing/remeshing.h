#ifndef REMESHING_H
#define REMESHING_H

#include <glog/logging.h>
#include <unordered_map>
#include <chrono>

#include <ctime>
#include "mc_tables.h"
#include "util/timer.h"
#include "engine/main_engine.h"
#include "core/collect_block_array.h"
#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include "core/PointCloud.h"
#include "core/hash_table.h"
#include "core/entry_array.h"
#include "geometry/geometry_helper.h"
#include "sensor/rgbd_sensor.h"
#include "visualization/compact_mesh.h"

void Remeshing(
	ScaleTable& scale_table,
    HashTable& hash_table,
    PointCloud& pc_gpu,
    EntryArray& candidate_entries_,
    GeometryHelper& geometry_helper);


// void Remeshing(CompactMesh& mesh,PointCloud& pc_gpu);

#endif