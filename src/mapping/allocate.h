//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_ALLOCATE_H
#define MESH_HASHING_ALLOCATE_H
#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include "core/PointCloud.h"
#include "core/hash_table.h"
#include "core/entry_array.h"
#include "core/block_array.h"
#include "geometry/geometry_helper.h"
#include "sensor/rgbd_sensor.h"

// @function
// See what entries of @param hash_table
// was affected by @param sensor
// with the help of @param geometry_helper
double AllocBlockArray(
    HashTable& hash_table,
    Sensor& sensor,
    GeometryHelper& geometry_helper
);
double AllocBlockArray(
	ScaleTable& scale_table,
	HashTable& hash_table,
	pcl::PointCloud<pcl::PointXYZRGB> pc_,
	pcl::gpu::DeviceArray< pcl::PointXYZRGB >& pc,
	GeometryHelper& geometry_helper
);
double AllocBlockArray(
	ScaleTable& scale_table,
	HashTable& hash_table,
    BlockArray &blocks,
	PointCloud& pc_gpu,
	EntryArray& candidate_entries_,
	GeometryHelper& geometry_helper
);
double AllocScaleBlock(
    HashTable& hash_table,
    ScaleTable& scale_table,
    BlockArray& blocks,
    EntryArray& candidate_entries_
    );

#endif //MESH_HASHING_ALLOCATE_H
