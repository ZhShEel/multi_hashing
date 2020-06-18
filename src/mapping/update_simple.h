//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_FUSE_H
#define MESH_HASHING_FUSE_H

#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "core/PointCloud.h"
#include "core/hash_table.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "core/mesh.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"
#include "geometry/knncuda.h"

// @function
// Enumerate @param candidate_entries
// change the value of @param blocks
// according to the existing @param mesh
//                 and input @param sensor data
// with the help of hash_table and geometry_helper
double UpdateBlocksSimple(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    pcl::PointCloud<pcl::PointXYZRGB> pc_,
    pcl::gpu::DeviceArray< pcl::PointXYZRGB >& pc,
    pcl::gpu::DeviceArray< pcl::Normal >&, 
    HashTable& hash_table,
    ScaleTable& scale_table,
    float voxel_size,
    GeometryHelper& geometry_helper
);

double UpdateBlocksSimple(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    PointCloud& pc_gpu, 
    HashTable& hash_table,
    ScaleTable& scale_table,
    float4x4& cTw,
    float voxel_size,
    GeometryHelper& geometry_helper
);

#endif //MESH_HASHING_FUSE_H
