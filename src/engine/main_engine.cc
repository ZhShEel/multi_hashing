//
// Created by wei on 17-10-22.
//

#include <Eigen/Eigen>
#include <mapping/update_psdf.h>
#include <optimize/primal_dual.h>
#include "engine/main_engine.h"

#include "core/collect_block_array.h"
#include "localizing/point_to_psdf.h"
#include "mapping/allocate.h"
#include "mapping/update_simple.h"
#include "mapping/recycle.h"
#include "meshing/marching_cubes.h"
#include "visualization/compress_mesh.h"
#include "visualization/extract_bounding_box.h"

////////////////////
/// Host code
////////////////////
/// GPU -> mat6x6, mat6x1
/// solve by Eigen -> Matrix<float, 6, 1> dxi
/// turn to Sophus -> Sophus::SE3f::Tangent
/// expf in Eigen  -> Matrix4f dT
/// Eigen::Matrix4f -> float4x4
/// cTw float4x4

float4x4 SolveAndConvertDeltaXi(const mat6x6& A, const mat6x1&b, float lambda) {
  Eigen::Matrix<float, 6, 6> eigen_A;
  Eigen::Matrix<float, 6, 1> eigen_b, eigen_dxi;

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      eigen_A.coeffRef(i, j) = A.entries2D[i][j];
    }
    eigen_b.coeffRef(i) = b.entries[i];
  }

  eigen_A = eigen_A + lambda * Eigen::Matrix<float, 6, 6>::Identity();
  eigen_dxi = eigen_A.ldlt().solve(-eigen_b);
  //LOG(INFO) << "\n" << eigen_A;
  //LOG(INFO) << "\n" << eigen_b.transpose();
  //eigen_dxi = eigen_A.inverse() * eigen_b;
  //LOG(INFO) << eigen_dxi.transpose();
  Eigen::Matrix4f eigen_dT = Sophus::SE3f::exp(eigen_dxi).matrix();
  float4x4 dT;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      dT.entries2[i][j] = eigen_dT.coeff(i, j);
    }
  }

  return dT;
}

Eigen::Matrix<float, 6, 1> SE3Tose3(float4x4 mat) {
  Eigen::Matrix4f eigen_mat;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      eigen_mat(i, j) = mat.entries2[i][j];
    }
  }





  
  /// [t ~ omega] in Sophus
  Sophus::SE3f SE3;
  SE3.setRotationMatrix(eigen_mat.topLeftCorner<3, 3>());
  SE3.translation() = eigen_mat.topRightCorner<3, 1>();
  return SE3.log();
};

void MainEngine::Localizing(Sensor &sensor, int iters, float4x4& gt) {
  for (int iter = 0; iter < iters; ++iter) {
//    Eigen::Matrix<float, 6, 1> pose = SE3Tose3(sensor.wTc());
//    Eigen::Matrix<float, 6, 1> gt_pose = SE3Tose3(gt);
//    std::stringstream ss;
//    for (int i = 0; i < 6; ++i) {
//      ss << pose(i) - gt_pose(i) << " ";
//    }
//    LOG(INFO) << "Delta pose: " << ss.str();

    mat6x6 A;
    mat6x1 b;
    int count;
    float error = PointToSurface(blocks_, sensor,
                                 hash_table_, geometry_helper_,
                                 A, b, count);
    if (count == 0) {
      LOG(INFO) << "Count equals 0!";
      return;
    };

    LOG(INFO) << "Localization error: " << error << " / " << count << " = "
              << error / count;
    log_engine_.WriteLocalizationError(error);

    float4x4 dT = SolveAndConvertDeltaXi(A, b, 100000);
    sensor.set_transform(dT * sensor.wTc());
  }
}

void GetPCBBX(pcl::gpu::DeviceArray<pcl:: PointXYZRGB>& pc,
              float3 & min_bbx,
              float3 & max_bbx){
  min_bbx = make_float3(999999,999999,999999);
  max_bbx = make_float3(-999999,-999999,-999999);
  for(int i=0;i<pc.size();i++){
    if(pc[i].x<min_bbx.x)
      min_bbx.x = pc[i].x;
    if(pc[i].y<min_bbx.y)
      min_bbx.y = pc[i].y;
    if(pc[i].z<min_bbx.z)
      min_bbx.z = pc[i].z;
    if(pc[i].x>max_bbx.x)
      max_bbx.x = pc[i].x;
    if(pc[i].y>max_bbx.y)
      max_bbx.y = pc[i].y;
    if(pc[i].z>max_bbx.z)
      max_bbx.z = pc[i].z;
  }
}

float4x4 MainEngine::Icp(PointCloud& pc_now, PointCloud& pc_last, float4x4& cTw_last){
  float4x4 ans;
  if (pc_now.count()==0 || pc_last.count()==0){
    return cTw_last;
  }
  else{
    ans = do_icp(pc_now, pc_last, cTw_last);
    pc_last = pc_now;
    cTw_last = ans;
    float3 center = make_float3(ans.m14,ans.m24,ans.m34);
    // printf("center:%f %f %f\n",center.x,center.y,center.z);
    return ans;
  }
}


//origin:pc_,pc_normals, curvatures
void MainEngine::Mapping(PointCloud pc_gpu, float4x4 cTw) {
  // float3 min_bbx,max_bbx;
  // GetPCBBX(pc,min_bbx,max_bbx);
  // double alloc_time = AllocBlockArray(
  //     scale_table_,
  //     hash_table_,
  //     pc_gpu,
  //     candidate_entries_,
  //     geometry_helper_
  // );
  // LOG(INFO) <<"alloc time:"<<alloc_time<<std::endl;
  // allocate_time_sum = (allocate_time_sum*integrated_frame_count_)/(1+integrated_frame_count_)
  //                   + alloc_time/(1+integrated_frame_count_);

  // double spread_time = SpreadCurvatureToHash(
  //     scale_table_,
  //     pc_,
  //     pc,
  //     normals,
  //     curvatures,
  //     geometry_helper_
  // );
  bool If_multiscale = false;
  
  double spread_time = SpreadMeasurementTerm(
      scale_table_,
      hash_table_,
      candidate_entries_,
      pc_gpu,
      mesh_,
      blocks_,
      geometry_helper_
  );
  LOG(INFO) <<"Compute measurement term:"<<spread_time;

  double roughness_time = ComputeRoughnessTerm(
      scale_table_,
      hash_table_,
      candidate_entries_,
      mesh_,
      blocks_,
      geometry_helper_
  );
  LOG(INFO) <<"Compute roughness term:"<<roughness_time;

  // double setflags_time = SetJoinFlags(
  //     scale_table_,
  //     hash_table_,
  //     candidate_entries_,
  //     geometry_helper_
  // );
  // LOG(INFO) <<"Set join flag:"<<setflags_time;

  // CountScaleNumberSerial(scale_table_,hash_table_);
  //update join process before join_time]
  double join_time = 0;
  if(If_multiscale){
    join_time = JoinBlocks(
        scale_table_,
        hash_table_,
        candidate_entries_,
        blocks_,
        mesh_,
        pc_gpu,
        geometry_helper_
    );
    LOG(INFO) << "Join time:"<<join_time;
   }
   //update join process after join

    double alloc_time = AllocBlockArray(
       scale_table_,
       hash_table_,
       blocks_,
       pc_gpu,
       candidate_entries_,
       geometry_helper_
   );
   LOG(INFO) <<"alloc time:"<<alloc_time<<std::endl;
   allocate_time_sum = (allocate_time_sum*integrated_frame_count_)/(1+integrated_frame_count_)
                     + alloc_time/(1+integrated_frame_count_);


  double collect_time = CollectAllBlocks(
      hash_table_,
      scale_table_,
      candidate_entries_
  );

  for(int i=0;i<2;i++){
    double alloc_scale_time = AllocScaleBlock(
      hash_table_,
      scale_table_,
      blocks_,
      candidate_entries_
      );
  }



  double update_time = 0;
  LOG(INFO) << "Simple update";

  //test
  // Remeshing(
  //   scale_table_,
  //   hash_table_,
  //   pc_gpu,
  //   candidate_entries_,
  //   geometry_helper_);

  update_time = UpdateBlocksSimple(candidate_entries_,
                                   blocks_,
                                   pc_gpu,
                                   hash_table_,
                                   scale_table_,
                                   cTw,
                                   volume_params_.voxel_size,
                                   geometry_helper_);
  LOG(INFO) << "Update time:"<<update_time;
  update_time_sum = (update_time_sum*integrated_frame_count_)/(1+integrated_frame_count_)
                    + update_time/(1+integrated_frame_count_);
  log_engine_.WriteMappingTimeStamp(
        alloc_time,
        join_time,
        update_time,
        integrated_frame_count_);
  

  integrated_frame_count_ ++;
}

// void MainEngine::Mapping(pcl::PointCloud<pcl::PointXYZRGB>& pc_,
//                          pcl::gpu::DeviceArray< pcl::PointXYZRGB >& pc,
//                          pcl::gpu::DeviceArray< pcl::Normal >& normals,
//                          pcl::gpu::DeviceArray< pcl::PrincipalCurvatures >& curvatures ) {
//   // float3 min_bbx,max_bbx;
//   // GetPCBBX(pc,min_bbx,max_bbx);
//   double alloc_time = AllocBlockArray(
//       scale_table_,
//       hash_table_,
//       pc_,
//       pc,
//       geometry_helper_
//   );
//   LOG(INFO) <<"alloc time:"<<alloc_time<<std::endl;
//   allocate_time_sum = (allocate_time_sum*integrated_frame_count_)/(1+integrated_frame_count_)
//                     + alloc_time/(1+integrated_frame_count_);

//   double spread_time = SpreadCurvatureToHash(
//       scale_table_,
//       pc_,
//       pc,
//       normals,
//       curvatures,
//       geometry_helper_
//   );

//   double setflags_time = SetJoinFlags(
//       scale_table_,
//       hash_table_,
//       geometry_helper_
//   );
//   LOG(INFO) <<"Set join flag:"<<setflags_time;

//   double join_time = JoinBlocks(
//       scale_table_,
//       hash_table_,
//       blocks_,
//       geometry_helper_
//   );
//   LOG(INFO) << "Join time:"<<join_time;

//   double collect_time = CollectAllBlocks(
//       hash_table_,
//       candidate_entries_
//   );

//   double join_process_time = spread_time + setflags_time + join_time + collect_time;
//   join_time_sum = (join_time_sum*integrated_frame_count_)/(1+integrated_frame_count_)
//                     + join_process_time/(1+integrated_frame_count_);

//   double update_time = 0;
//   LOG(INFO) << "Simple update";
//   update_time = UpdateBlocksSimple(candidate_entries_,
//                                      blocks_,
//                                      pc_,
//                                      pc,
//                                      normals,
//                                      hash_table_,
//                                      scale_table_,
//                                      volume_params_.voxel_size,
//                                      geometry_helper_);
//   LOG(INFO) << "Update time:"<<update_time;
//   update_time_sum = (update_time_sum*integrated_frame_count_)/(1+integrated_frame_count_)
//                     + update_time/(1+integrated_frame_count_);

//   log_engine_.WriteMappingTimeStamp(
//         alloc_time,
//         collect_time,
//         update_time,
//         integrated_frame_count_);
  

//   integrated_frame_count_ ++;
// }

void MainEngine::Meshing() {
  // if (! vis_engine_.enable_meshing_) return;
  float time = MarchingCubes(candidate_entries_,
                             blocks_,
                             mesh_,
                             hash_table_,
                             scale_table_,
                             geometry_helper_,
                             color_type_,
                             map_engine_.enable_bayesian_update(),
                             enable_sdf_gradient_,
                             volume_params_.voxel_size);

  LOG(INFO) << "MarchingCubes time:" <<time;

  marching_cube_time_sum = (marching_cube_time_sum*(integrated_frame_count_-1))/(integrated_frame_count_)
                    + time/(integrated_frame_count_);

  Timer timer;
  timer.Tick();
  CollectLowSurfelBlocks(candidate_entries_,
                         blocks_,
                         hash_table_,
                         geometry_helper_);
  if (integrated_frame_count_ % 10 == 0) {
    RecycleGarbageBlockArray(candidate_entries_,
                             blocks_,
                             mesh_,
                             hash_table_,
                             scale_table_);
  }
  double recycle_time = timer.Tock();
  LOG(INFO) << "Recycle time:" <<recycle_time;
  recycle_time_sum = (recycle_time_sum*(integrated_frame_count_-1))/(integrated_frame_count_)
                    + recycle_time/(integrated_frame_count_);
  log_engine_.WriteMeshingTimeStamp(time, recycle_time, integrated_frame_count_);
}

void MainEngine::Recycle() {
  // TODO(wei): change it via global parameters
  int kRecycleGap = 15;
  if (!map_engine_.enable_bayesian_update()
      && integrated_frame_count_ % kRecycleGap == kRecycleGap - 1) {
    StarveOccupiedBlockArray(candidate_entries_, blocks_);
    CollectGarbageBlockArray(candidate_entries_,
                             blocks_,
                             geometry_helper_);
    hash_table_.ResetMutexes();
    RecycleGarbageBlockArray(candidate_entries_,
                             blocks_,
                             mesh_,
                             hash_table_,
                             scale_table_);
  }
}

// view: world -> camera
int MainEngine::Visualize(float4x4 view, float4x4 view_gt) {
  if (vis_engine_.enable_interaction()) {
    vis_engine_.update_view_matrix();
  } else {
    glm::mat4 glm_view;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        glm_view[i][j] = view.entries2[i][j];
    glm_view = glm::transpose(glm_view);
    vis_engine_.set_view_matrix(glm_view);
  }

  if (vis_engine_.enable_global_mesh()) {
    CollectAllBlocks(hash_table_,scale_table_, candidate_entries_);
  } // else CollectBlocksInFrustum

  int3 stats;
  CompressMesh(candidate_entries_,
               blocks_,
               mesh_,
               vis_engine_.compact_mesh(),
               stats);

  if (vis_engine_.enable_bounding_box()) {
    vis_engine_.bounding_box().Reset();

    ExtractBoundingBox(candidate_entries_,
                       vis_engine_.bounding_box(),
                       geometry_helper_);
  }
  if (vis_engine_.enable_trajectory()) {
    vis_engine_.trajectory().AddPose(view.getInverse());
    vis_engine_.trajectory().AddPose(view_gt.getInverse());
  }


  if (vis_engine_.enable_ray_casting()) {
    Timer timer;
    timer.Tick();
    vis_engine_.RenderRayCaster(view,
                                blocks_,
                                hash_table_,
                                geometry_helper_);
    double raycasting_time = timer.Tock();
    LOG(INFO) << " Raycasting time: " << raycasting_time;
    log_engine_.WriteRayCastingTimeStamp(raycasting_time, integrated_frame_count_-1);
  }

  return  vis_engine_.Render();
}

int MainEngine::Visualize(float4x4 view) {
  if (vis_engine_.enable_meshing_) {
    if (vis_engine_.enable_interaction()) {
      vis_engine_.update_view_matrix();
    } else {
      glm::mat4 glm_view;
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
          glm_view[i][j] = view.entries2[i][j];
      glm_view = glm::transpose(glm_view);
      vis_engine_.set_view_matrix(glm_view);
    }

    if (vis_engine_.enable_global_mesh()) {
      CollectAllBlocks(hash_table_,scale_table_, candidate_entries_);
    } // else CollectBlocksInFrustum
    int3 stats;
    CompressMesh(candidate_entries_,
                 blocks_,
                 mesh_,
                 vis_engine_.compact_mesh(),
                 stats);
    log_engine_.WriteMeshStats(stats.x, stats.y, integrated_frame_count_ - 1);
    if (vis_engine_.enable_bounding_box()) {
      vis_engine_.bounding_box().Reset();

      ExtractBoundingBox(candidate_entries_,
                         vis_engine_.bounding_box(),
                         geometry_helper_);
    }
    if (vis_engine_.enable_trajectory()) {
      vis_engine_.trajectory().AddPose(view.getInverse());
    }
  }

  if (vis_engine_.enable_ray_casting()) {
    Timer timer;
    timer.Tick();
    vis_engine_.RenderRayCaster(view,
                                blocks_,
                                hash_table_,
                                geometry_helper_);
    double raycasting_time = timer.Tock();
    LOG(INFO) << " Raycasting time: " << raycasting_time;
    log_engine_.WriteRayCastingTimeStamp(raycasting_time, integrated_frame_count_-1);
  }
  return vis_engine_.Render();
}

void MainEngine::Log() {
  if (log_engine_.enable_video()) {
    cv::Mat capture = vis_engine_.Capture();
    log_engine_.WriteVideo(capture);
  }
}

void MainEngine::FinalLog() {
  CollectAllBlocks(hash_table_,scale_table_, candidate_entries_);
  Meshing();
  int3 timing;
  CompressMesh(candidate_entries_,
               blocks_,
               mesh_,
               vis_engine_.compact_mesh(), timing);
  if (log_engine_.enable_ply()) {
    log_engine_.WritePly(vis_engine_.compact_mesh());
  }
}

void MainEngine::FinalLog_without_meshing() {
  CollectAllBlocks(hash_table_,scale_table_, candidate_entries_);
  int3 timing;
  CompressMesh(candidate_entries_,
               blocks_,
               mesh_,
               vis_engine_.compact_mesh(), timing);
  // if (log_engine_.enable_ply()) {
  //   log_engine_.WritePly(vis_engine_.compact_mesh());
  // }
}

void MainEngine::TempLog(int frame){
    if(frame > 8)
        return;
    CollectAllBlocks(hash_table_,scale_table_,candidate_entries_);
    int3 timing;
    CompressMesh(candidate_entries_,blocks_,mesh_,vis_engine_.compact_mesh(),timing);
    if(log_engine_.enable_ply()){
        log_engine_.WritePly_Temp(vis_engine_.compact_mesh(),frame);
    }
}

/// Life cycle
//TODO here:
long MainEngine::init(
    const ScaleParams& scale_params,
    const HashParams& hash_params,
    const VolumeParams &volume_params,
    const MeshParams &mesh_params,
    const SensorParams &sensor_params,
    const RayCasterParams &ray_caster_params
) {
  scale_params_ = scale_params;
  hash_params_ = hash_params;
  volume_params_ = volume_params;
  mesh_params_ = mesh_params;
  sensor_params_ = sensor_params;
  ray_caster_params_ = ray_caster_params;

  scale_table_.Resize(scale_params);
  hash_table_.Resize(hash_params);
  candidate_entries_.Resize(hash_params.entry_count);
  blocks_.Resize(hash_params.value_capacity);

  mesh_.Resize(mesh_params);

  geometry_helper_.Init(volume_params);

  allocate_time_sum = 0;
  join_time_sum = 0;
  update_time_sum = 0;
  marching_cube_time_sum = 0;
  recycle_time_sum = 0;
}

MainEngine::~MainEngine() {
  hash_table_.Free();
  blocks_.Free();
  mesh_.Free();

  candidate_entries_.Free();
}

/// Reset
void MainEngine::Reset() {
  integrated_frame_count_ = 0;

  hash_table_.Reset();
  blocks_.Reset();
  mesh_.Reset();

  candidate_entries_.Reset();
}

void MainEngine::ConfigMappingEngine(
    bool enable_bayesian_update
) {
  map_engine_.Init(sensor_params_.width,
                   sensor_params_.height,
                   enable_bayesian_update);
}

void MainEngine::ConfigVisualizingEngine(
    gl::Lighting &light,
    bool enable_navigation,
    bool enable_global_mesh,
    bool enable_bounding_box,
    bool enable_trajectory,
    bool enable_polygon_mode,
    bool enable_meshing,
    bool enable_ray_caster,
    bool enable_join_process,
    bool enable_color
) {
  // the same resolution as sensor params
    /*
   // printf("11\n");
   vis_engine_.Init("VisEngine", sensor_params_.width, sensor_params_.height, sensor_params_);
   // printf("22\n");
   vis_engine_.set_interaction_mode(enable_navigation);
   //printf("33\n"); 
   vis_engine_.set_light(light);
   vis_engine_.BindMainProgram(mesh_params_.max_vertex_count,
                               mesh_params_.max_triangle_count,
                               enable_global_mesh,
                               enable_polygon_mode,
                               enable_color);
                               */
  vis_engine_.compact_mesh().Resize(mesh_params_);
/*
   if (enable_bounding_box || enable_trajectory) {
     vis_engine_.BuildHelperProgram();
   }

   if (enable_bounding_box) {
     vis_engine_.InitBoundingBoxData(hash_params_.value_capacity*24);
   }
   if (enable_trajectory) {
     vis_engine_.InitTrajectoryData(80000);
   }

   vis_engine_.enable_meshing_ = enable_meshing;
   if (enable_ray_caster) {
     vis_engine_.BuildRayCaster(ray_caster_params_);
   }
   */
}

void MainEngine::ConfigLoggingEngine(
    std::string path,
    bool enable_video,
    bool enable_ply
) {
  log_engine_.Init(path);
  if (enable_video) {
    log_engine_.ConfigVideoWriter(MainEngine::sensor_params_.width, MainEngine::sensor_params_.height);
  }
  if (enable_ply) {
    log_engine_.ConfigPlyWriter();
  }
}

void MainEngine::RecordBlocks(std::string prefix) {
  //CollectAllBlocks(hash_table_, candidate_entries_);
  BlockMap block_map = log_engine_.RecordBlockToMemory(
      blocks_.GetGPUPtr(), hash_params_.value_capacity,
      candidate_entries_.GetGPUPtr(), candidate_entries_.count());

  std::stringstream ss("");
  ss << integrated_frame_count_ - 1;
  log_engine_.WriteRawBlocks(block_map, prefix + ss.str());
}
