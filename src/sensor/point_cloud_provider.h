
#ifndef MESH_HASHING_PC_LOCAL_SEQUENCE_H
#define MESH_HASHING_PC_LOCAL_SEQUENCE_H

#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include "io/config_manager.h"

struct PointCloudDataProvider {
  /// Read from Disk
  size_t frame_id = 0;
  std::vector<std::string> pointcloud_list;


  void LoadDataset(std::string dataset_path);

  /// If read from disk, then provide mat at frame_id
  /// If read from network/USB, then wait until a mat comes;
  ///                           a while loop might be inside
  bool ProvideData(pcl::PointCloud<pcl::PointXYZRGB>::Ptr);
};

#endif //MESH_HASHING_RGBD_LOCAL_SEQUENCE_H
