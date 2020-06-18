#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include "psdf.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>


#define DEBUG_

GLuint showFaceList, showallpointlist,showalloclist,showNodeList;

//shader part
CompactMesh Remesh_mesh;
std::mutex remeshing_mutex;
std::condition_variable remesh_cond;
PointCloud remesh_point;
bool remesh_signal = false;

//

void initGL()
{
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClearDepth(1.0);
  glShadeModel(GL_SMOOTH);
  GLfloat ambient[] = {0.3f,0.3f,0.3f,1.0f};
  GLfloat diffuse[] = {1.0f,1.0f,1.0f,1.0f};
  GLfloat specular[] = {1.0f,1.0f,1.0f,1.0f};
  GLfloat position[] = {0.5f,-0.5f,0.8f,1.0f};
  glLightfv(GL_LIGHT0,GL_POSITION,position);
  glLightfv(GL_LIGHT0,GL_AMBIENT,ambient);
  glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuse);
  glLightfv(GL_LIGHT0,GL_SPECULAR,specular);
  glEnable(GL_DEPTH_TEST);
  // ------------------- Lighting
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  // ------------------- Display List
  showFaceList = glGenLists(1);
  showallpointlist = glGenLists(1);
  showalloclist = glGenLists(1);
  showNodeList = glGenLists(1);

}

void ShowPointsPerFrame(pcl::PointCloud<pcl::PointXYZRGB> pc){
    float scale = 300;
  // float scale = 10;
    for(int i=0;i<pc.size();i++){
        glDisable(GL_LIGHTING);
        glBegin(GL_POINTS);
        glPointSize(7.f);   
        // glColor3d(1-curvature_vis_his[i],curvature_vis_his[i],0); 
        glColor3d(0,0,0);   
        // glVertex3f(pc[i].x/octree->lengthSide,pc[i].y/octree->lengthSide,pc[i].z/octree->lengthSide);
        glVertex3f(pc[i].x/scale,pc[i].y/scale,pc[i].z/scale);
        glEnd();
        glEnable(GL_LIGHTING);
    }
    glEndList();
    glCallList(showallpointlist);
}
void ShowAllocPerFrame(pcl::PointCloud<pcl::PointXYZRGB> pc){
    
    for(int x=0;x<pc.size();x++){
        int step = 0;
        float voxel_size_ = 5;
        for(float i=pc[x].x-step*voxel_size_;i<=pc[x].x+step*voxel_size_;i+=voxel_size_){
          for(float j=pc[x].y-step*voxel_size_;j<=pc[x].y+step*voxel_size_;j+=voxel_size_){
            for(float k=pc[x].z-step*voxel_size_;k<=pc[x].z+step*voxel_size_;k+=voxel_size_){
              glDisable(GL_LIGHTING);
              glBegin(GL_POINTS);
              glPointSize(7.f);   
              // glColor3d(1-curvature_vis_his[i],curvature_vis_his[i],0); 
              glColor3d(1,0,0);   
              // glVertex3f(pc[i].x/octree->lengthSide,pc[i].y/octree->lengthSide,pc[i].z/octree->lengthSide);
              glVertex3f(pc[x].x/500,pc[x].y/500,pc[x].z/500);
              glEnd();
              glEnable(GL_LIGHTING);
            }
          }
        }
        
    }
    glEndList();
    glCallList(showalloclist);
}

void drawSkelethonCube(float posX, float posY,float posZ,float dX, float dY,float dZ, float3 rgb)
{
    float vis_scale = 500;
    // float vis_scale = 1;
    posX /= vis_scale;
    posY /= vis_scale;
    posZ /= vis_scale;
    dX /= vis_scale;
    dY /= vis_scale;
    dZ /= vis_scale;
  // cout<<posX<<" "<<posY<<" "<<posZ<<" "<<dX<<" "<<dY<<" "<<dZ<<endl;
    glNewList(showNodeList,GL_COMPILE);
    glDisable(GL_LIGHTING);
    glLineWidth(1.f);
    glBegin(GL_LINE_LOOP);  
    glColor3d(rgb.x,rgb.y,rgb.z);
    glVertex3f(posX,posY,posZ);
    glVertex3f(posX+dX,posY,posZ);
    glVertex3f(posX+dX,posY+dY,posZ);
    glVertex3f(posX,posY+dY,posZ);
    glEnd();
    
    glBegin(GL_LINE_LOOP);  
    glColor3d(rgb.x,rgb.y,rgb.z);
    glVertex3f(posX,posY,posZ+dZ);
    glVertex3f(posX+dX,posY,posZ+dZ);
    glVertex3f(posX+dX,posY+dY,posZ+dZ);
    glVertex3f(posX,posY+dY,posZ+dZ);
    glEnd();
    
    glBegin(GL_LINES);
    glColor3d(rgb.x,rgb.y,rgb.z);
    glVertex3f(posX,posY,posZ);
    glVertex3f(posX,posY,posZ+dZ);
    glEnd();
    
    glBegin(GL_LINES);
    glColor3d(rgb.x,rgb.y,rgb.z);
    glVertex3f(posX+dX,posY,posZ);
    glVertex3f(posX+dX,posY,posZ+dZ);
    glEnd();

    glBegin(GL_LINES);
    glColor3d(rgb.x,rgb.y,rgb.z);
    glVertex3f(posX+dX,posY+dY,posZ);
    glVertex3f(posX+dX,posY+dY,posZ+dZ);
    glEnd();    

    glBegin(GL_LINES);
    glColor3d(rgb.x,rgb.y,rgb.z);
    glVertex3f(posX,posY+dY,posZ);
    glVertex3f(posX,posY+dY,posZ+dZ);
    glEnd();
    glEnable(GL_LIGHTING);
    glEndList();
    glCallList(showNodeList);
};

uint HashBucketForBlockPos(const int3& pos, int bucket_count) {
  const int p0 = 73856093;
  const int p1 = 19349669;
  const int p2 = 83492791;

  int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2))
            % bucket_count;
  if (res < 0) res += bucket_count;
  return (uint) res;
}

bool IsPosAllocated(const int3& pos, const ScaleInfo& scale_info) {
  return pos.x == scale_info.pos.x
      && pos.y == scale_info.pos.y
      && pos.z == scale_info.pos.z
      && scale_info.scale != -1;
}

ScaleInfo GetScale(const int3& pos, ScaleInfo* scale_, uint entry_count) {
        uint bucket_size = 10;
        uint linked_list_size = 7;
        uint bucket_idx = HashBucketForBlockPos(pos,100000);
        uint bucket_first_scale_idx = bucket_idx * 10;
        
        ScaleInfo scale_info;
        scale_info.pos = pos;
        scale_info.scale = 1;   //need to rethink(scale:from 1 to upper)
        scale_info.offset = 0;

        for(int i=0;i<bucket_size;++i){
          ScaleInfo curr_scale = scale_[i+bucket_first_scale_idx];
          if(IsPosAllocated(pos,curr_scale))
            return curr_scale;
        }

        const uint bucket_last_scale_idx = bucket_first_scale_idx + bucket_size - 1;
        int i = bucket_last_scale_idx;

/// The last entry is visted twice, but it's OK
  #pragma unroll 1
        for(uint iter = 0;iter < linked_list_size;++iter) {
          ScaleInfo curr_scale = scale_[i];
          if(IsPosAllocated(pos,curr_scale)) 
            return curr_scale;
          if(curr_scale.offset == 0)
            break;
          i = (bucket_last_scale_idx + curr_scale.offset) % (entry_count);
        }
        return scale_info;
}
void ShowJoiningNode(JoinProcessVis join_process_vis){
  // glClear(GL_COLOR_BUFFER_BIT);
  // uint node_count = entry_array.count();
  // for(int i=0;i<node_count;++i){
  //   printf("%d/%d\n",i,node_count);
  //   HashEntry &entry = entry_array[i];
  //   printf("count:%d\n",blocks.count());
  //   std::cout<<entry.Ptr()<<std::endl;
  //   Block& block = blocks[entry.ptr];
  //   printf("%d/%d\n",i,node_count);
  //   int3   voxel_pos = geometry_helper.BlockToVoxel(entry.pos);

  //   float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  //   int curr_scale = GetScale(voxel_pos, scale_table.scale_, scale_table.entry_count ).scale; 
  //   printf("tranf\n");
  //   drawSkelethonCube(entry.pos.x, entry.pos.y, entry.pos.z,
  //                     curr_scale*voxel_size, curr_scale*voxel_size, curr_scale*voxel_size,
  //                     make_float3(1,0,0));
  //   printf("drawed\n");

  // }
  for(int i=0;i<join_process_vis.entry_count;i++){
    float3 color;
    if(join_process_vis.join_state_toShow[i] == 0)
      color = make_float3(0,0,0);
    else if(join_process_vis.join_state_toShow[i] == 1)
      color = make_float3(1,0,0);
    drawSkelethonCube(join_process_vis.pos_toShow[i].x, join_process_vis.pos_toShow[i].y, join_process_vis.pos_toShow[i].z,
                      join_process_vis.length_toShow[i],join_process_vis.length_toShow[i],join_process_vis.length_toShow[i],
                      color);
  }
  // glCallList(showNodeList);
}


float3 getcolorfrompalette(int v, float3 p[], int num){
  //from 0-255
  if(v > 255)
    return make_float3(0,0,0);

  int w = v / (255/(num - 1));

  float3 color_min = p[w];
  float3 color_max = p[w+1];
  
  int v_min = w * (255/(num - 1));
  int v_max = (w + 1) * (255/(num - 1));

  return color_min + (v - v_min) * ((color_max - color_min)/(v_max - v_min));
}
void ShowMesh(bool IfShowLine, bool IfScale, CompactMesh &compact_mesh)
{
  uint compact_vertex_count = compact_mesh.vertex_count();
  uint compact_triangle_count = compact_mesh.triangle_count();
  // float scale = 500;
  float scale = 8;
  float3* vertices = new float3[compact_vertex_count];
  float3* normals  = new float3[compact_vertex_count];
  int3* triangle  = new int3  [compact_triangle_count];
  int* triangle_scale = new int[compact_triangle_count];

  cudaMemcpy(vertices, compact_mesh.vertices(),
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost);
  cudaMemcpy(triangle, compact_mesh.triangles(),
                             sizeof(int3) * compact_triangle_count,
                             cudaMemcpyDeviceToHost);
  cudaMemcpy(normals, compact_mesh.normals(),
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost);
  cudaMemcpy(triangle_scale, compact_mesh.scales(),
                             sizeof(int) * compact_triangle_count,
                             cudaMemcpyDeviceToHost);

        glNewList(showFaceList, GL_COMPILE);
        glDisable(GL_LIGHTING);
        if(IfShowLine){ 
            glLineWidth(1.0f); 
            glColor3f(0.5f,0.5f,0.5f);
            for(int i=0;i<compact_triangle_count;i++){
              int3 idx = triangle[i];
                        // glColor3f((float)mesh_unit.curr_cube_idx/255,0,0);
                        // glColor3f(triangle.color.x,triangle.color.y,triangle.color.z); 
                        glBegin(GL_LINES);
                        glVertex3f(vertices[idx.x].x/scale,
                            vertices[idx.x].y/scale,
                            vertices[idx.x].z/scale);
                        glVertex3f(vertices[idx.y].x/scale,
                            vertices[idx.y].y/scale,
                            vertices[idx.y].z/scale);
                        glEnd();
                        glBegin(GL_LINES);
                        glVertex3f(vertices[idx.x].x/scale,
                            vertices[idx.x].y/scale,
                            vertices[idx.x].z/scale);
                        glVertex3f(vertices[idx.z].x/scale,
                            vertices[idx.z].y/scale,
                            vertices[idx.z].z/scale);
                        glEnd();
                        glBegin(GL_LINES);
                        glVertex3f(vertices[idx.z].x/scale,
                            vertices[idx.z].y/scale,
                            vertices[idx.z].z/scale);
                        glVertex3f(vertices[idx.y].x/scale,
                            vertices[idx.y].y/scale,
                            vertices[idx.y].z/scale);
                        glEnd();
                        // std::cout<<triangle.normals[0].x<<" "<<triangle.normals[0].y<<" "<<triangle.normals[0].z<<std::endl;
                    }
        }
        // glEnable(GL_LIGHTING);
        for(int i=0;i<compact_triangle_count;i++){
                    int3 idx = triangle[i];
                    // glColor3f((float)mesh_unit.curr_cube_idx/255,0,0);  //cubeindex color
                    // glColor3f(triangle.confidence,1-triangle.confidence,0); //confidence color
                    if(IfScale){
                      int show_scale = triangle_scale[i];
                      //'RdYIGn_r'
                      float3 color_palette[6] = {
                        make_float3(0.22468281430219147, 0.6558246828143022, 0.3444059976931949),
                        make_float3(0.6165321030372932, 0.835909265667051, 0.41191849288735105),
                        make_float3(0.8918877354863514, 0.954479046520569, 0.6010765090349866),
                        make_float3(0.9971549404075356, 0.9118031526336026, 0.6010765090349867),
                        make_float3(0.9873125720876587, 0.6473663975394082, 0.36424452133794705),
                        make_float3(0.8899653979238754, 0.28673587081891583, 0.19815455594002307)};

                      float3 color_of_scale = getcolorfrompalette(show_scale,color_palette,6);
                      
                      glColor3f(color_of_scale.x,color_of_scale.y,color_of_scale.z);  //scale color

                      //simple method...
                      if(show_scale == 1)
                        glColor3f(0,0,1);
                      else if(show_scale == 2)
                        glColor3f(0,1,1);
                      else if(show_scale == 3)
                        glColor3f(0,1,0);
                      else if(show_scale == 4)
                        glColor3f(1,1,0);
                      else if(show_scale == 5)
                        glColor3f(1,0,0);
                      else
                        glColor3f(0,0,1);
                      //
                    }
                    else{
                      float3 this_normals = normals[idx.x]*0.5+0.5;
                      float3 lightpos = make_float3(1,1,1);
                      // printf("see:%f %f %f\n",this_normals.x,this_normals.y,this_normals.z);
                      float3 lightDir = make_float3(lightpos.x-vertices[idx.x].x/scale, 
                                                    lightpos.y-vertices[idx.x].y/scale,
                                                    lightpos.z-vertices[idx.x].z/scale);
                      float3 lightcolor = make_float3(1.0f,0.8f,0.6f);
                      float3 ambientcolor = make_float3(0.6f,0.6f,1.0f)*0.2;
                      float D = sqrt(lightDir.x*lightDir.x+lightDir.y*lightDir.y+lightDir.z*lightDir.z);
                      float3 falloff = make_float3(0.2f,0.3f,0.5f);
                      float Attenuation = 1.0 / ( falloff.x + (falloff.y*D) + (falloff.z*D*D) );

                      float3 Diffuse = lightcolor*max(dot(this_normals,lightDir),0.0);
                      float3 Intensity = ambientcolor + Diffuse * Attenuation;

                      // glColor3f(Intensity.x,Intensity.y,Intensity.z);
                      //former is base color(rgb)
                      // float3 FinalColor = make_float3(255/255,153/255,0/255) * Intensity;
                      float3 FinalColor = make_float3((255*Intensity.x/255),(153*Intensity.y/255),(33*Intensity.z/255));
                        // vec3 FinalColor = DiffuseColor.rgb * Intensity;
                         // gl_FragColor = vColor * vec4(FinalColor, DiffuseColor.a);
                      // printf("see:%f %f %f %f %f %f\n",Intensity.x,Intensity.y,Intensity.z,FinalColor.x,FinalColor.y,FinalColor.z);
                      glColor3f(FinalColor.x,FinalColor.y,FinalColor.z);

                      // glColor3f(normals[idx.x].x/2+0.5,nWormals[idx.x].y/2+0.5,normals[idx.x].z/2+0.5);  //normals color
                    }
                    // std::cout<<"color:"<<triangle.color.x<<" "<<triangle.color.y<<" "<<triangle.color.z<<std::endl;
                    glBegin(GL_TRIANGLES);
                    // printf("%f %f %f-%f %f %f-%f %f %f\n",vertices[idx.x].x,vertices[idx.x].y,vertices[idx.x].z,
                    //   vertices[idx.y].x,vertices[idx.y].y,vertices[idx.y].z,vertices[idx.z].x,vertices[idx.z].y,vertices[idx.z].z);
                    glVertex3f(vertices[idx.x].x/scale,vertices[idx.x].y/scale,vertices[idx.x].z/scale);
                    glVertex3f(vertices[idx.y].x/scale,vertices[idx.y].y/scale,vertices[idx.y].z/scale);
                    glVertex3f(vertices[idx.z].x/scale,vertices[idx.z].y/scale,vertices[idx.z].z/scale);
                    glEnd();
        }
        glEndList();
        delete[] vertices;
        delete[] triangle;
        delete[] normals;
        glCallList(showFaceList);
}

void UpdateDataSizeImage(cv::Mat &DataSizeImage, std::vector<cv::Point>& DataSizePoints, int frame_num, int pc_all_cnt){
    int max_pt_show = 1, max_frame_show = 1500;
    DataSizeImage.setTo(cv::Scalar(255, 255, 255));
    for (int i = 0; i < DataSizePoints.size(); i++)
    {
      if(max_pt_show<DataSizePoints[i].y)
        max_pt_show = DataSizePoints[i].y;
    }
    // max_pt_show = max(max_pt_show,pc_all_cnt);
    max_pt_show = 2000000;
    // std::cout<<"see:"<<pc_all->size()<<" "<<max_pt_show<<std::endl;
    DataSizePoints.push_back(cv::Point(frame_num,pc_all_cnt));
    // std::cout<<frame_num*480/max_frame_show<<" "<<pc_all->size()*640/max_pt_show<<"---"<<std::endl;
    std::vector<cv::Point> DataSizePoints_nor = DataSizePoints;
    std::vector<cv::Point> X_axis;
    std::vector<cv::Point> Y_axis;
    for (int i = 0; i < DataSizePoints.size(); i++)
    {
      DataSizePoints_nor[i].x = (DataSizePoints[i].x*640/max_frame_show)*550/640 + 50;  //[50,600]
      DataSizePoints_nor[i].y = 450-(DataSizePoints[i].y*480/(1.3*max_pt_show));
    }
    X_axis.push_back(cv::Point(610,450));
    X_axis.push_back(cv::Point(50,450));
    Y_axis.push_back(cv::Point(50,50));
    Y_axis.push_back(cv::Point(50,450));
    std::string origin_pt_show = "O";
    cv::putText(DataSizeImage,origin_pt_show,cv::Point(30,470),cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0),2, 8, 0);
    cv::polylines(DataSizeImage, X_axis, false, cv::Scalar(0,0,0),2, 8, 0);
    cv::polylines(DataSizeImage, Y_axis, false, cv::Scalar(0,0,0),2, 8, 0);
    cv::polylines(DataSizeImage, DataSizePoints_nor, false, cv::Scalar(255,0,0), 1, 8, 0);
    cv::imshow("data size",DataSizeImage);
    cv::waitKey(1);
}

int wait_time[2] = {-1, 10};
int wait_time_idx = 0;

void Send_to_remeshing_thread(CompactMesh &mesh_){
  std::cout<<"send to remesh thread."<<std::endl;
  std::lock_guard<std::mutex> guard(remeshing_mutex);
  Remesh_mesh = mesh_;
}

void Get_remeshing_data(CompactMesh &mesh_){
  std::lock_guard<std::mutex> guard(remeshing_mutex);
  mesh_ = Remesh_mesh;
}

int main_function(int argc, char **argv) {
  psdf process("../config/");
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_all(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
  float4x4 wTc;
  // change the datareader and loop below with your own IO
  // RGBDDataProvider rgbd_local_sequence; 
  PointCloudDataProvider pc_sequence;
  
  pc_sequence.LoadDataset(process.config.data_path);
  // pc_sequence.LoadDataset(process.args.dataset_type)
  // pc_sequence.LoadDataset("/home/zhangsheng/Research/recon/recon/data/rgbd_dataset_freiburg3_long_office_household/point");
  // pc_sequence.LoadDataset("/home/zhangsheng/ssd/Dataset/Point/data/clply"); //Engine
  // pc_sequence.LoadDataset("/home/zhangsheng/ssd/Dataset/ICL-NUIM/living_room_traj0_frei_png/point");
  // pc_sequence.LoadDataset("/home/zhangsheng/ssd/Dataset/Point/data/data/ply2"); //Door
  // pc_sequence.LoadDataset("/home/zhangsheng/Research/recon/recon/data/all/ply3");
  // pc_sequence.LoadDataset("/home/zhangsheng/Research/recon/recon/data/ball_n_plane");

  // std::string config_path = kConfigPaths[dataset_type];
  // pc_sequence.LoadDataset(config_path);
  int frame_num = 1;
  int all_frame_num = 0;
  uint sum_point_num = 0;

    //visualization

    // // Create OpenGL window
    pangolin::CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    // pangolin::Var<bool> menuShowNode("menu.Show Node",false,true);
    pangolin::Var<bool> menuShowMesh("menu.Show Mesh",false,true);
    pangolin::Var<bool> menuPoints("menu.Show Point",true,true);
    pangolin::Var<bool> menuAlloc("menu.Show Alloc",false,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
    pangolin::Var<bool> record_cube("menu.Record_Cube",true,false);
    //glEnable(GL_DEPTH_TEST);
    initGL();

    float eyez = -0.8, eye_angle = 45;
    float eyex = 0.5*1.414*cos(eye_angle*3.14159/180);
    float eyey = 0.5*1.414*sin(eye_angle*3.14159/180);

 
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
            // pangolin::ModelViewLookAt(eyex,eyey,eyez,0,0,0,0,1,1)
            pangolin::ModelViewLookAt(eyex,eyey,eyez,0,0,0,pangolin::AxisNegY)
    );
    pangolin::Handler3D handler(s_cam);    
    pangolin::View &d_cam = pangolin::CreateDisplay().SetBounds(0.0,1.0,pangolin::Attach::Pix(175),1.0,-640.0f/480.0f).SetHandler(&handler);


    // draw data size 
    // cv::Mat DataSizeImage = cv::Mat::zeros(480,640,CV_8UC3);
    // DataSizeImage.setTo(cv::Scalar(255, 255, 255));
    // std::vector<cv::Point> DataSizePoints;
    // DataSizePoints.clear();
    // end of draw data size
    // end of visualization

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointpool(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr reliablepoints(new pcl::PointCloud<pcl::PointXYZRGB>);
  while (pc_sequence.ProvideData(pc)) {
    (*pc_all) = (*pc_all) + (*pc);
    //
    // (*pointpool) = (*pointpool) + (*pc);
    // if(pointpool->size()<=0)
    //   continue;
    // for(int i=0;i<pc->size();i++){
    //   bool flag_found = 0;
    //   for(int j=0;j<reliablepoints->size();j++){
    //     float disx = (*pc)[i].x-(*reliablepoints)[j].x;
    //     float disy = (*pc)[i].y-(*reliablepoints)[j].y;
    //     float disz = (*pc)[i].z-(*reliablepoints)[j].z;
    //     if(sqrt(disx*disx+disy*disy+disz*disz)<5){
    //       pc_all->push_back((*pc)[i]);
    //       flag_found = 1;
    //       break;
    //     }
    //   }
    //   if(flag_found==1)
    //     break;
    //   pcl::KdTreeFLANN<pcl::PointXYZRGB> kd;
    //   kd.setInputCloud(pointpool);
    //   std::vector<int> nearsetOne(30);
    //   std::vector<float> nearsetDis(30);
    //   kd.nearestKSearch((*pc)[i],30,nearsetOne,nearsetDis);
    //   pcl::PointCloud<pcl::PointXYZRGB>::iterator index = pointpool->begin();
    //   // std::cout<<"why:"<<nearsetDis[29]<<std::endl;
    //   if(nearsetDis[29]<5){
    //     reliablepoints->push_back((*pc)[i]);
    //     for(int j=0;j<30;j++){
    //       pc_all->push_back((*pointpool)[nearsetOne[j]]);
    //     }
    //   // std::cout<<"dis:"<<nearsetDis[0]<<std::endl;
    //   }
    // }



    
    // std::cout<<"now the pc size is "<<pc_all->size()<<" "<<pointpool->size()<<" points in the pool."<<std::endl;
    //
    // all_frame_num++;
    if(pc_all->size()<=20000){
    // if(frame_num < 1 ){
      continue;
    }    
    // if(all_frame_num > 100){
    //   process.main_engine.FinalLog();
    //   break;
    // }

    // if(pc_sequence.frame_id>100)
    //   break;
    
    Timer timer;
    timer.Tick();
    std::cout<<frame_num<<std::endl;
    frame_num++;
    pcl::NormalEstimation<pcl::PointXYZRGB,pcl::Normal> normalEstimation;
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB,pcl::Normal,pcl::PrincipalCurvatures> curvaturesEstimation;

    normalEstimation.setInputCloud(pc_all);
    normalEstimation.setKSearch(30);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    // pcl::gpu::DeviceArray< pcl::PointXYZRGB > cloud_device;
    // pcl::gpu::DeviceArray< pcl::Normal > normals_device;

    // cloud_device.upload(pc_all->points);
    // cloud_device.download(pc_all->points);
    // normals_device.upload(normals->points);
    // normals_device.download(normals->points);

    curvaturesEstimation.setInputCloud(pc_all);
    curvaturesEstimation.setInputNormals(normals);
    curvaturesEstimation.setSearchMethod(kdtree);
    curvaturesEstimation.setKSearch(30);
    curvaturesEstimation.compute(*curvatures);

    // pcl::gpu::DeviceArray<pcl::PrincipalCurvatures> curvatures_device;  
    // curvatures_device.upload(curvatures->points);
    // curvatures_device.download(curvatures->points);
    
    PointCloud pc_gpu;
    pc_gpu.Alloc(pc_all->size());
    Point* pc_cpu = new Point[pc_all->size()];

    float3 bbx_min = make_float3(999999,999999,999999);
    float3 bbx_max = make_float3(-999999,-999999,-999999);
    for(uint p = 0;p < pc_all->size();p++){
      pc_cpu[p].x = (*pc_all)[p].x; pc_cpu[p].y = (*pc_all)[p].y; pc_cpu[p].z = (*pc_all)[p].z; 
      pc_cpu[p].normal_x = (*normals)[p].normal_x;
      pc_cpu[p].normal_y = (*normals)[p].normal_y;
      pc_cpu[p].normal_z = (*normals)[p].normal_z;
      if(pc_cpu[p].x<bbx_min.x)
          bbx_min.x = pc_cpu[p].x;
      if(pc_cpu[p].y<bbx_min.y)
          bbx_min.y = pc_cpu[p].y;
      if(pc_cpu[p].z<bbx_min.z)
          bbx_min.z = pc_cpu[p].z;
      if(pc_cpu[p].x>bbx_min.x)
          bbx_max.x = pc_cpu[p].x;
      if(pc_cpu[p].y>bbx_min.y)
          bbx_max.y = pc_cpu[p].y;
      if(pc_cpu[p].z>bbx_min.z)
          bbx_max.z = pc_cpu[p].z;
    }
    pc_gpu.bbx_min = bbx_min;
    pc_gpu.bbx_max = bbx_max;
    pc_gpu.TransferPtToGPU(pc_cpu);
    std::cout<<pc_all->size()<<" "<<pc_gpu.count()<<std::endl;
    delete[] pc_cpu;
    
    process.reconstruction(pc_gpu);
    
    std::cout<<"frame:"<<frame_num<<" "<<"save:"<<process.args.save_per_frame<<std::endl;
    if(frame_num%process.args.save_per_frame == 0){
      // Send_to_remeshing_thread(process.main_engine.vis_engine_.compact_mesh());
      std::cout<<"send to remesh thread."<<std::endl;
      std::lock_guard<std::mutex> guard(remeshing_mutex);
      Remesh_mesh = process.main_engine.vis_engine_.compact_mesh();
      remesh_point = pc_gpu;
      uint compact_vertex_count = Remesh_mesh.vertex_count();
      std::cout<<"mesh number:"<<Remesh_mesh.vertex_count()<<std::endl;
      remesh_cond.notify_one();
    }

    // process.reconstruction(*pc_all,cloud_device,normals_device,curvatures_device);

    //visualizarion
   
      glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
      //glOrtho(-1,1,-1,1,-1,1);
      if(process.args.enable_view_rotation){
        eye_angle++;
        eyex = 0.5*1.414*cos(eye_angle*3.14159/180);
        eyey = 0.5*1.414*sin(eye_angle*3.14159/180);
        s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
              pangolin::ModelViewLookAt(eyex,eyey,eyez,0,0,0,pangolin::AxisNegY));
      }
      

      // d_cam.RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");
      if( pangolin::Pushed(record_cube) )
        pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=20,bps=8388608,unique_filename]//screencap.avi");
      d_cam.Activate(s_cam);
      
      if(!pangolin::ShouldQuit())
      {
          // pangolin::glDrawAxis(3);
          // if(menuShowNode)
          //     ShowNode(process.main_engine.candidate_entries_, 
          //              process.main_engine.scale_table_,
          //              process.main_engine.blocks_,
          //              process.main_engine.geometry_helper_,
          //              process.main_engine.volume_params_.voxel_size);
          if(menuShowMesh)
              // ShowMesh_();
              ShowMesh(0, process.args.enable_join_process, process.main_engine.vis_engine_.compact_mesh());
          if(menuPoints)
              ShowPointsPerFrame(*pc_all); 
          if(menuAlloc)
              ShowAllocPerFrame(*pc_all); 


//           setImageData(imageArray,3*640*480);
//           for(int i = 0 ; i < size;i++) {
//     imageArray[i] = (unsigned char)(rand()/(RAND_MAX/255.0));
// 


          pangolin::FinishFrame();
  
      }

      //without pangolin
      // std::cout<<"up:"<<pc_all->size()<<std::endl;
      // // UpdateDataSizeImage(DataSizeImage, DataSizePoints,frame_num,pc_all->size());
      // sum_point_num+=pc_all->size();
      // UpdateDataSizeImage(DataSizeImage, DataSizePoints,frame_num,sum_point_num);

    //end of visualization
    if(process.args.save_result_all_the_time||frame_num%1000==0){
      process.main_engine.FinalLog();   //savemesh
     // return 0;
    }
    else
      process.main_engine.FinalLog_without_meshing();
    // return 0;
    process.main_engine.TempLog(frame_num);
    pc_all->clear();
    LOG(INFO)<<"now each frame costs "<<timer.Tock()<<" s.";
    // ----------------- Toggle keys ---------------------------------
    // int key = cv::waitKey(wait_time[wait_time_idx]);
    // if (key == 27) return 0;  //ESC
    // else if (key == 13) wait_time_idx = 1;  //Enter
    // else if (key == 32) wait_time_idx = 1 - wait_time_idx;  //Space /* Toggle */ 
  }
  process.main_engine.FinalLog();
  pc_all->clear();
  return 0;
  // return 0;
}

int remeshing_function(int argc, char **argv){
  while(1){
    std::unique_lock<std::mutex> guard(remeshing_mutex);
    remesh_cond.wait(guard,[]{return Remesh_mesh.vertex_count()>0;});

    CompactMesh mesh_origin = Remesh_mesh;
    // Get_remeshing_data(mesh_origin);
    std::cout<<"data got, start to remesh them."<<std::endl;
    guard.unlock();
    //remesh below
    Remesh_mesh.remeshing(remesh_point);
    std::cout<<"remeshed."<<std::endl;
    
    // std::cout<<"remeshing:"<<mesh_origin.triangle_count()<<std::endl;
  }
}


int main(int argc, char **argv){
  
  ConfigManager all_config;
  RuntimeParams all_args;
  LoadRuntimeParams("../config/args.yml", all_args);
  DatasetType all_dataset_type = DatasetType(all_args.dataset_type);
  all_config.LoadConfig(all_dataset_type);
  Remesh_mesh.Resize(all_config.mesh_params);

  std::thread thread1(main_function, argc, argv);
  // std::thread thread2(remeshing_function, argc, argv);
  thread1.join();
  // thread2.join();

  return 0;
}
