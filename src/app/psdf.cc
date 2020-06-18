//
// Created by Yan on 19-1-13.
//

#include "app/psdf.h"

psdf::psdf(const std::string& config_path) {
  /// argv parsing

    LoadRuntimeParams(config_path+"/args.yml", args);
    dataset_type = DatasetType(args.dataset_type);
    config.LoadConfig(dataset_type);
    sensor.init(config.sensor_params);
    lighting.Load(config_path+"/lights.yaml");
  main_engine.init(
      config.scale_params,
      config.hash_params,
      config.sdf_params,
      config.mesh_params,
      config.sensor_params,
      config.ray_caster_params
  );
  main_engine.ConfigMappingEngine(
      args.enable_bayesian_update
  );
    
  main_engine.ConfigVisualizingEngine(
      lighting,
      args.enable_navigation,
      args.enable_global_mesh,
      args.enable_bounding_box,
      args.enable_trajectory,
      args.enable_polygon_mode,
      args.enable_meshing,
      args.enable_ray_casting,
      args.enable_join_process,
      args.enable_color
  );
  
  main_engine.ConfigLoggingEngine(
      args.filename_prefix,
      args.enable_video_recording,
      args.enable_ply_saving
  );
  main_engine.enable_sdf_gradient() = args.enable_sdf_gradient;
  //printf("=================color type: %d====================", args.color_type);
  main_engine.color_type() = args.color_type;

  pc_last.Free();
  cTw_last.setIdentity();
}

psdf::~psdf() {}

void psdf::normalizeDepthImage(cv::Mat& depth, cv::Mat& disp)
{

  ushort* depthData = (ushort*)depth.data;
  int width = depth.cols;
  int height = depth.rows;

  static ushort max = *std::max_element(depthData, depthData + width*height);
  static ushort min = *std::min_element(depthData, depthData + width*height);

  disp = cv::Mat(depth.size(), CV_8U);
  uchar* dispData = disp.data;


  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
    {
      //if(depthData[i*width + j]<)
      dispData[i*width + j] = (((double)(depthData[i*width + j] - min)) / ((double)(max - min))) * 255;
    }

}
// int psdf::reconstruction(pcl::PointCloud<pcl::PointXYZRGB> pc_,
//                          pcl::gpu::DeviceArray< pcl::PointXYZRGB > pc, 
//                          pcl::gpu::DeviceArray< pcl::Normal > normal, 
//                          pcl::gpu::DeviceArray< pcl::PrincipalCurvatures > curvature){
//   // sensor.Process(pc);
//   // sensor.set_transform(wTc);
//   // float4x4 cTw = wTc.getInverse();

//   main_engine.Mapping(pc_,pc,normal,curvature);
//   main_engine.Meshing();
  
//   const float cTw_arr[16] = {
//        0.992902, 0.118918, -0.002181, 0.000000,
//        -0.000000, 0.018340, 0.999832, 0.000000,
//        0.118938, -0.992735, 0.018210, 0.000000,
//        -0.651026, -1.863275, -5.765015, 1.000000
//    };
//   float4x4 cTw = float4x4(cTw_arr).getTranspose();
//   // main_engine.Visualize(cTw);
//   //main_engine.Log(); //enable video
//   main_engine.Recycle();
// }

int psdf::reconstruction(PointCloud pc_gpu){

  // float4x4 cTw = main_engine.Icp(pc_gpu, pc_last, cTw_last);
  
  const float cTw_arr[16] = {
       0.992902, 0.118918, -0.002181, 0.000000,
       -0.000000, 0.018340, 0.999832, 0.000000,
       0.118938, -0.992735, 0.018210, 0.000000,
       -0.651026, -1.863275, -5.765015, 1.000000
   };
   float4x4 cTw = float4x4(cTw_arr).getTranspose();


  std::cout<<cTw.m11<<" "<<cTw.m12<<" "<<cTw.m13<<" "<<cTw.m14<<std::endl<<
            cTw.m21<<" "<<cTw.m22<<" "<<cTw.m23<<" "<<cTw.m24<<std::endl<<
            cTw.m31<<" "<<cTw.m32<<" "<<cTw.m33<<" "<<cTw.m34<<std::endl<<
            cTw.m41<<" "<<cTw.m42<<" "<<cTw.m43<<" "<<cTw.m44<<std::endl;
  main_engine.Mapping(pc_gpu, cTw);
  main_engine.Meshing();

  // main_engine.Visualize(cTw);
  //main_engine.Log(); //enable video
  main_engine.Recycle();
}
// int psdf::reconstruction(cv::Mat color, cv::Mat depth, float4x4 wTc){
//   // Preprocess data
//   // cv::imshow("rgb",color);
//   // cv::imshow("depth",depth);
//   // normalization is applicable for depth data from datasets using CV_16F
//   //cv::Mat depthshow;
//   //normalizeDepthImage(depth,depthshow);
//   //cv::imshow("depth",depthshow);
//   sensor.Process(depth, color, true);
//   sensor.set_transform(wTc);
//   float4x4 cTw = wTc.getInverse();
//   main_engine.Mapping(sensor);
//   main_engine.Meshing();
//   // main_engine.Visualize(cTw);
// //    float4x4 m; m.setIdentity(); m.m22 = m.m33 = -1;
// //    const float cTw_arr[16] = {
// //        0.992902, 0.118918, -0.002181, 0.000000,
// //        -0.000000, 0.018340, 0.999832, 0.000000,
// //        0.118938, -0.992735, 0.018210, 0.000000,
// //        -0.651026, -1.863275, -5.765015, 1.000000
// //    };
// //    cTw = float4x4(cTw_arr).getTranspose();
// //    cTw = m * cTw * m;
//   //if (main_engine.Visualize(cTw))
//     //break;

//   main_engine.Log();
//   //main_engine.RecordBlocks();
//   main_engine.Recycle();

// }
