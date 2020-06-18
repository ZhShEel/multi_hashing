
#include "point_cloud_provider.h"
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>

int scanFiles(std::vector<std::string> &fileList, std::string inputDirectory)    //io
{
    inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char* str = inputDirectory.c_str();

    p_dir = opendir(str);   
    if( p_dir == NULL)
    {
        std::cout<< "can't open :" << inputDirectory << std::endl;
    }

    struct dirent *p_dirent;

    while ( p_dirent = readdir(p_dir))
    {
        std::string tmpFileName = p_dirent->d_name;
        if( tmpFileName == "." || tmpFileName == "..")
        {
            continue;
        }
        else
        {
          std::string padleft(20-tmpFileName.size(),'0');
          std::string tmpFileNamepad = padleft+tmpFileName;
          rename((inputDirectory + tmpFileName).c_str(),(inputDirectory + tmpFileNamepad).c_str());
          tmpFileName = padleft + tmpFileName;
            fileList.push_back(inputDirectory+tmpFileName);
        }
    }
    closedir(p_dir);
    return fileList.size();
}

void PointCloudDataProvider::LoadDataset(
    std::string dataset_path
) {//pointcloud_list
    // string dir = "/home/zhangsheng/Research/recon/recon/data/data/ply1/";
    pcl::PLYReader reader;
    // MarchingCubes marchingcubes;
    std::cout<<"data folder:"<<dataset_path<<std::endl;
    int file_number = scanFiles(pointcloud_list,dataset_path);
    sort(pointcloud_list.begin(),pointcloud_list.end());
}

bool PointCloudDataProvider::ProvideData(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc
) {
  std::cout<<frame_id<<" "<<pointcloud_list.size()<<std::endl;
  if (frame_id >= pointcloud_list.size()) {
    LOG(ERROR) << "All images provided!";
    return false;
  }
  if(pcl::io::loadPLYFile<pcl::PointXYZRGB>(pointcloud_list[frame_id],*pc)==-1){
    PCL_ERROR("Couldn't read file test_pcd.pcd \n");
        system("PAUSE");
        return (-1);
  }

  ++frame_id;

  return true;
  // TODO: Network situation
}
