//
// Created by wei on 17-10-21.
//

#include "rgbd_data_provider.h"
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
const std::string kConfigPaths[] = {
    "../config/ICL.yml",
    "../config/TUM1.yml",
    "../config/TUM2.yml",
    "../config/TUM3.yml",
    "../config/ZHOU.yml",
    "../config/SUN3D.yml",
    "../config/PKU.yml",
    "../config/BOE.yml"
};

void RGBDDataProvider::LoadDataset(
    DatasetType dataset_type
) {
  std::string config_path = kConfigPaths[dataset_type];
  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  std::string dataset_path = (std::string)fs["dataset_path"];
  LoadDataset(dataset_path, dataset_type);
}

void RGBDDataProvider::LoadDataset(Dataset dataset) {
  LoadDataset(dataset.path, dataset.type);
}

void RGBDDataProvider::LoadDataset(
    std::string dataset_path,
    DatasetType dataset_type
) {
  switch (dataset_type) {
    case ICL:
      LoadICL(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case ZHOU:
      LoadZHOU(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case SUN3D:
      LoadSUN3D(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case TUM1:
      LoadTUM(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case TUM2:
      LoadTUM(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case TUM3:
      LoadTUM(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case PKU:
      Load3DVCR(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
    case BOE:
      LoadZHOU(dataset_path, depth_image_list, color_image_list, wTcs);
      break;
  }
}

bool RGBDDataProvider::ProvideData(
    cv::Mat &depth,
    cv::Mat &color
) {
  if (frame_id > depth_image_list.size()) {
    LOG(ERROR) << "All images provided!";
    return false;
  }
  depth = cv::imread(depth_image_list[frame_id], cv::IMREAD_UNCHANGED);
  color = cv::imread(color_image_list[frame_id]);
  if (color.channels() == 3) {
    cv::cvtColor(color, color, cv::COLOR_BGR2BGRA);
  }
  ++frame_id;

  return true;
  // TODO: Network situation
}

bool RGBDDataProvider::ProvideData(cv::Mat &depth,
                              cv::Mat &color,
                              float4x4 &wTc) {
  if (frame_id >= depth_image_list.size()) {
    LOG(ERROR) << "All images provided!";
    return false;
  }

  {
    LOG(INFO) << frame_id << "/" << depth_image_list.size();
    depth = cv::imread(depth_image_list[frame_id], cv::IMREAD_UNCHANGED);

    //std::cout << ori_depth * 255 / 6 * 0.001 << std::endl;
    //cv::imshow("original depth", ori_depth * 255 / 6 * 0.001);
    color = cv::imread(color_image_list[frame_id]);
    //cv::Mat ori_depth = cv::imread(depth_image_list[frame_id], cv::IMREAD_UNCHANGED);
    //cv::resize(ori_depth, depth, cv::Size(1000, 750), cv::INTER_CUBIC);
    //cv::Mat ori_color = cv::imread(color_image_list[frame_id]);
    //cv::resize(ori_color, color, cv::Size(1000, 750), cv::INTER_NEAREST);

    if (color.channels() == 3) {
      cv::cvtColor(color, color, cv::COLOR_BGR2BGRA);
    }
    wTc = wTcs[0].getInverse() * wTcs[frame_id];
    ++frame_id;
  } ///while (frame_id >= 1960 && frame_id <= 1985);

  return true;
  // TODO: Network situation
}