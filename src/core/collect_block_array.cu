#include "matrix.h"

#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include <unordered_set>
#include <vector>
#include <list>
#include <glog/logging.h>
#include <device_launch_parameters.h>
#include <util/timer.h>
#include "meshing/mc_tables.h"


#define PINF  __int_as_float(0x7f800000)

////////////////////
/// class MappingEngine - compress, recycle
////////////////////

uint count_num = 0;

__device__
float DisFromPtToMesh(
    float3 p,
    float3 v1,
    float3 v2,
    float3 v3
    ){
        //1.if point can project inside the triangle
        float3 p_v1 = v1 - p;
        float3 p_v2 = v2 - p;
        float3 p_v3 = v3 - p;
        // if(cross(p_v1,p_v2)*cross(p_v3,p_v1)<=0 || cross(p_v1,p_v2)*cross(p_v2,p_v3)<=0)
        //     return -1;
        if(dot(v2 + v3 - 2 * v1,p - v1) < 0 || dot(v1 + v3 - 2 * v2,p - v2) < 0 || dot(v2 + v1 - 2 * v3,p - v3) < 0)
        return -1;

        //2.compute normal of triangle
        float3 normal = cross(v2-v1, v3-v1);

        //3.compute distance
        float dis = dot(p_v1, normal) / sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        return abs(dis);
    }
__global__
void OutputBlockCornerKernel(EntryArray candidate_entries, int* cornerVec, int entry_count, GeometryHelper geometry_helper){
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= entry_count)
    return;
    HashEntry &entry = candidate_entries[idx];
    cornerVec[3 * idx] = entry.pos.x;
    cornerVec[3 * idx + 1] = entry.pos.y;
    cornerVec[3 * idx + 2] = entry.pos.z;
    return;
}

uint block_output_num = 0;
void OutputBlockCorner(
    EntryArray &candidate_entries,
    GeometryHelper geometry_helper){
        uint entry_count = candidate_entries.count();
        if(entry_count <= 0)
        return;
        int* cornerVec = new int[entry_count*3];
        int* cornerVec_gpu;
        cudaMalloc(&cornerVec_gpu, entry_count * sizeof(int) * 3);
        const dim3 grid_size(entry_count,1);
        const dim3 block_size(1,1);
        OutputBlockCornerKernel<<<grid_size, block_size>>>(candidate_entries, cornerVec_gpu, entry_count, geometry_helper);
        cudaMemcpy(cornerVec, cornerVec_gpu, entry_count * sizeof(int) * 3, cudaMemcpyDeviceToHost);
        cudaFree(cornerVec_gpu);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        printf("start writing corner\n");
        block_output_num++;
        std::string output_block_path = "BlockVis/block_corner_"+std::to_string(block_output_num)+".ply";
        std::ofstream out(output_block_path);
        std::stringstream ss;  // ss.str("");
        out << "ply\n"
        "format ascii 1.0\n";
        out << "element vertex " << entry_count << "\n";
        out << "property float x\n"
        "property float y\n"
        "property float z\n";
        // "property float nx\n"
        // "property float ny\n"
        // "property float nz\n"
        // "property uchar red\n"
        // "property uchar green\n"
        // "property uchar blue\n";
        out << "element face 0" << "\n";
        out << "property list uchar int vertex_index\n";
        out << "end_header\n";
        // out << ss.str();

        for(uint i = 0; i < entry_count; i++){
            out<<cornerVec[3*i] << " " <<cornerVec[3*i+1] <<" "<<cornerVec[3*i+2]<<"\n"; 
        }
        // cudaFree(cornerVec_gpu);
    }
__global__
void  ClearExistedJoinValueKernel(
    EntryArray candidate_entries,
    Mesh mesh_m,
    BlockArray blocks,
    GeometryHelper geometry_helper
    ){
        const HashEntry &entry = candidate_entries[blockIdx.x];
        Block& block = blocks[entry.ptr];

        MeshUnit &mesh_unit = block.mesh_units[threadIdx.x];

        mesh_unit.measurement_term = -1;

        return; 
    }


__global__
void SpreadMeasurementTermKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    Point* pc,
    uint pc_size,
    Mesh mesh_m,
    BlockArray blocks, 
    GeometryHelper geometry_helper
    ){
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx > pc_size)
        return;
        Point p = pc[idx];

        //1.Find the voxel point p belong
        int3 curr_pos = geometry_helper.WorldToBlock(make_float3(p.x,p.y,p.z), geometry_helper.voxel_size); //finest 
        int curr_scale = scale_table.GetScale(curr_pos).scale;
        if(curr_scale==-1)  //modify it later
        return;
        int3 block_pos = 
        geometry_helper.WorldToBlock(make_float3(p.x,p.y,p.z), pow(2,curr_scale-1) * geometry_helper.voxel_size) * pow(2,curr_scale-1);
        int3 voxel_pos = 
        geometry_helper.WorldToVoxeli(make_float3(p.x,p.y,p.z), pow(2,curr_scale-1) * geometry_helper.voxel_size) * pow(2,curr_scale-1);

        uint3 this_offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos, pow(2,curr_scale-1));
        uint voxel_index = geometry_helper.VectorizeOffset(this_offset);
        HashEntry entry = hash_table.GetEntry(block_pos);
        if(entry.ptr == FREE_ENTRY)
        return;

        //2.Compute distance between p and every triangle in the voxel
        MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[voxel_index];
        float min_dis = 999999;
        for(uint i=0;i<N_TRIANGLE;i++){
            if(mesh_unit.triangle_ptrs[i]==FREE_PTR){
                if(i==0)
                return;
                else
                break;
            }
            Triangle triangle = mesh_m.triangle(mesh_unit.triangle_ptrs[i]);

            float distance = 
            DisFromPtToMesh(make_float3(p.x,p.y,p.z), mesh_m.vertex(triangle.vertex_ptrs.x).pos, mesh_m.vertex(triangle.vertex_ptrs.y).pos, mesh_m.vertex(triangle.vertex_ptrs.z).pos);
            if(distance < min_dis)
            min_dis = distance;
        }

    //3. add all distace to hash entry
    if(min_dis<0)
    return;
    mesh_unit.Update_Measurement(min_dis);

}

__global__
void ComputeRoughnessTermKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    Mesh mesh_m,
    BlockArray blocks,
    GeometryHelper geometry_helper
    ){
        const HashEntry &entry = candidate_entries[blockIdx.x];
        Block& block = blocks[entry.ptr];
    
        MeshUnit &mesh_unit = block.mesh_units[threadIdx.x];

        float roughness = 0;

        int triangle_num = 0;
        for(uint i=0;i<N_TRIANGLE;i++){
            if(mesh_unit.triangle_ptrs[i]==FREE_PTR){
                triangle_num = i;
                break;
            }
        }
        for(int i=0;i<triangle_num;i++){
            for(int j=0;j<triangle_num;j++){
                if(i==j)
                continue;
                Triangle triangle_i = mesh_m.triangle(mesh_unit.triangle_ptrs[i]);
                Triangle triangle_j = mesh_m.triangle(mesh_unit.triangle_ptrs[j]);

                float3 normal_i = make_float3((mesh_m.vertex(triangle_i.vertex_ptrs.x).normal.x+mesh_m.vertex(triangle_i.vertex_ptrs.y).normal.x+mesh_m.vertex(triangle_i.vertex_ptrs.z).normal.x)/3,
                                              (mesh_m.vertex(triangle_i.vertex_ptrs.x).normal.y+mesh_m.vertex(triangle_i.vertex_ptrs.y).normal.y+mesh_m.vertex(triangle_i.vertex_ptrs.z).normal.y)/3,
                                              (mesh_m.vertex(triangle_i.vertex_ptrs.x).normal.z+mesh_m.vertex(triangle_i.vertex_ptrs.y).normal.z+mesh_m.vertex(triangle_i.vertex_ptrs.z).normal.z)/3);

                float3 normal_j = make_float3((mesh_m.vertex(triangle_j.vertex_ptrs.x).normal.x+mesh_m.vertex(triangle_j.vertex_ptrs.y).normal.x+mesh_m.vertex(triangle_j.vertex_ptrs.z).normal.x)/3,
                                              (mesh_m.vertex(triangle_j.vertex_ptrs.x).normal.y+mesh_m.vertex(triangle_j.vertex_ptrs.y).normal.y+mesh_m.vertex(triangle_j.vertex_ptrs.z).normal.y)/3,
                                              (mesh_m.vertex(triangle_j.vertex_ptrs.x).normal.z+mesh_m.vertex(triangle_j.vertex_ptrs.y).normal.z+mesh_m.vertex(triangle_j.vertex_ptrs.z).normal.z)/3);


                float3 d = normal_i - normal_j;
                roughness += (d.x*d.x+d.y*d.y+d.z*d.z);

            }
        }
        if(roughness==0)
        return;
        
        if(mesh_unit.measurement_term == -1)
            mesh_unit.measurement_term = 0;
        mesh_unit.measurement_term += geometry_helper.regularization_rough_measure * roughness;

    }

__global__
void GetMaxTermInAllVoxelsKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    Mesh mesh_m,
    BlockArray blocks,
   // CompactMesh compact_mesh,
    GeometryHelper geometry_helper
    ){
        const HashEntry &entry = candidate_entries[blockIdx.x];
        Block& block = blocks[entry.ptr];

        MeshUnit &mesh_unit = block.mesh_units[threadIdx.x];
        //printf("compare:%f < %f\n",mesh_unit.measurement_term, geometry_helper.join_th);
        uint rest_vertex = mesh_m.GetRestVertex();
        uint rest_triangle = mesh_m.GetRestTriangle();
        float rest_space = 0;
        if(rest_vertex < rest_triangle)
            rest_space = rest_vertex;
        else
            rest_space = rest_triangle;
       // printf("rest:%d\n",rest);

        //printf("compare:%f < %f\n",mesh_unit.measurement_term, geometry_helper.join_th/rest_space);
        //condition here
        if(mesh_unit.measurement_term < geometry_helper.join_th*(1/rest_space) && block.join_value==0){
            // if(mesh_unit.measurement_term>0)
            //   printf("compare:%f th:%f(%d %d %d)\n",mesh_unit.measurement_term,geometry_helper.join_th*(1/rest_space),entry.pos.x,entry.pos.y,entry.pos.z);
            block.Update_JoinValue(mesh_unit.measurement_term);
        }

    }

__global__
void CountScaleNumberSerialKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    int number,
    int* output,
    int large_size
    ){
        for(int i=0;i<number;i++){
            int3 curr_pos = hash_table.entry(i).pos;
// 
            if(hash_table.entry(i).ptr == FREE_ENTRY)
            continue;
            int curr_scale = scale_table.GetScale(curr_pos).scale;
            // printf("scale:%d\n",curr_scale);
            // if(curr_scale==2){
            //   printf("pos:%d %d %d(%d/%d)->%d %d(%d)\n",curr_pos.x,curr_pos.y,curr_pos.z,i,number,curr_scale,hash_table.entry(i).will_join,hash_table.entry(i).ptr);
            // }
            if(curr_scale < large_size)
            output[curr_scale]++;
        }
        // printf("\n\n\n\n");
        return;
    }

//join every 8 nodes now
__global__
void ClearWillJoinKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries
    ){
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        HashEntry &entry = candidate_entries[idx];
       // if(scale_table.GetScale(entry.pos).scale<0)
       //   printf("eee:%d %d %d->%d\n",entry.pos.x,entry.pos.y,entry.pos.z,scale_table.GetScale(entry.pos).scale);
        entry.will_join = 0;
        entry.join_signal = 0;
        entry.mutex = 0;

    }

__global__
void ClearJoinSignalKernel(
    ScaleTable scale_table,
    EntryArray candidate_entries,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    int entry_count
    ){
        const HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;
        if(threadIdx.x==0){
            curr_scale_[blockIdx.x] = scale_table.GetScale(curr_pos).scale;
            join_signal_[blockIdx.x] = 0;
            for(int i=0;i<8;i++){
                If_free[blockIdx.x * 8 + i] = 0;
            }
        }
    }

//0: empty:totally free-can join
//1: same scale: can join
//2: different scale:cannot join
__device__
double IsTotallyEmpty(
    ScaleTable scale_table,
    HashTable hash_table,
    int3 pos,
    int scale
    ){
        //if(scale_table.GetScale(pos).scale>0)
         // printf("pos:%d %d %d scale:%d ancest:%d %d %d\n",pos.x,pos.y,pos.z,scale_table.GetScale(pos).scale,scale_table.GetAncestor(pos).x,scale_table.GetAncestor(pos).y,scale_table.GetAncestor(pos).z);
        if(scale_table.GetScale(pos).scale == scale && scale_table.GetAncestor(pos)==pos)
            return 1;
        int s = pow(2,scale-1);
        for(int i=0;i<s;i++){
            for(int j=0;j<s;j++){
                for(int k=0;k<s;k++){
                    int3 this_p = pos + make_int3(i,j,k);
                    int this_scale = scale_table.GetScale(scale_table.GetAncestor(this_p)).scale;
                    //part of a joined cube may be empty.
                    if(hash_table.GetEntry(scale_table.GetAncestor(this_p)).ptr != FREE_ENTRY)
                        return 2;
                    //printf("judge:%d %d %d->(%d %d %d)(%d %d %d):ptr:%d scale:%d\n",pos.x,pos.y,pos.z,i,j,k,this_p.x,this_p.y,this_p.z,hash_table.GetEntry(this_p).ptr,this_scale);
                    //if(hash_table.GetEntry(this_p).ptr != FREE_ENTRY || (this_scale != scale && this_scale != -1))
                     //   return 2;
                }
            }
        }
        return 0;
    }


__global__
void SetJoinFlagsKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    bool* If_free,
    EntryArray candidate_entries
    ){
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        HashEntry &entry = candidate_entries[idx];
       // printf("curvature:%f - pos:%d %d %d\n",entry.join_value, entry.pos.x, entry.pos.y, entry.pos.z);
        const uint accept_free_node_number = 4; //for rough objects, can set 2.
        int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                           {0,0,1,1,0,0,1,1},
                           {0,1,0,1,0,1,0,1}};
        //for each block, search its 8-neighbor with its scale
        // int3 curr_pos = hash_table.entry(idx).pos;
        int3 curr_pos = candidate_entries[idx].pos;
        int curr_scale = scale_table.GetScale(curr_pos).scale;   //search

        if(curr_scale==-1){
            return;
        }
            
        uint NotJoinFlag = 0;
        bool NotJoinThisNode = 0;
        for(int i=0;i<8;i++){
            int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-1);
            int3 neigh_ancestor_pos = scale_table.GetAncestor(neigh_pos);

            //8-neighbor must exist and have same scale (add conditions)
            int neigh_scale = scale_table.GetScale(neigh_pos).scale;
            double empty_value = IsTotallyEmpty(scale_table,hash_table,neigh_pos,curr_scale);
            if(empty_value==2){
                NotJoinThisNode = 1;
            }
            else if(empty_value==0){
                If_free[idx*8+i] = 1;
                NotJoinFlag++;
            }
            if(NotJoinFlag > accept_free_node_number)
                NotJoinThisNode = 1;

        }
        if(NotJoinThisNode)
            return;

    hash_table.SetWillJoinTrue(curr_pos);
    entry.will_join = 1;

    }

__device__
float3 floor3(float3 a){
    return make_float3(floor(a.x),floor(a.y),floor(a.z));
}

__device__
void AllocScaleForEmptyBrother(ScaleTable scale_table, int3 pos, int scale){
    int s = pow(2,scale-1);
    int3 ancestor = pos;
    for(int i=0;i<s;i++){
        for(int j=0;j<s;j++){
            for(int k=0;k<s;k++){
                int3 n = pos + make_int3(i,j,k);
                scale_table.AllocAncestedScale(n,scale,ancestor);
            }
        }
    }
}

__global__
void AllocAllBrothersKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    uint group_id,
    GeometryHelper geometry_helper
    ){
        if(threadIdx.x!=0)
          return;
        const HashEntry &entry = candidate_entries[blockIdx.x];
        int curr_scale = curr_scale_[blockIdx.x];
        int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                           {0,0,1,1,0,0,1,1},
                           {0,1,0,1,0,1,0,1}};
        //if this cube belongs to group_id
        int3 curr_pos = entry.pos;
        int3 curr_pos_curr_scale = make_int3(floor3(make_float3(curr_pos) / pow(2,curr_scale-1)));
        int curr_pos_group_query = (curr_pos_curr_scale.x % 2) * 4 + 
        (curr_pos_curr_scale.y % 2) * 2 +  
        (curr_pos_curr_scale.z % 2);
        

        if(curr_pos_group_query != group_id)
          return;
        
        //if this cube's will_join flag is true
        if(entry.will_join == 0)
          return;
    
    
        for(int i=0;i<8;i++){
            int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-1);
            if(If_free[blockIdx.x*8+i]){
              hash_table.AllocEntry(neigh_pos);
            }
        }
        return;
    }

__global__
void SetJoinSignalEightTimeKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    uint group_id,
    GeometryHelper geometry_helper
    ){
        if(threadIdx.x!=0)
          return;
        const uint idx = blockIdx.x;
        //if this cube belongs to group_id
        const HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;
        int curr_scale = curr_scale_[idx];
        int3 curr_pos_curr_scale = make_int3(floor3(make_float3(curr_pos) / pow(2,curr_scale-1)));
        int curr_pos_group_query = (curr_pos_curr_scale.x % 2) * 4 + 
        (curr_pos_curr_scale.y % 2) * 2 +  
        (curr_pos_curr_scale.z % 2);
        
        if(curr_scale>3)
          return;
        if(curr_pos_group_query != group_id)
          return;
        
        //if this cube's will_join flag is true
        if(entry.will_join == 0)
          return;
    
        int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                           {0,0,1,1,0,0,1,1},
                           {0,1,0,1,0,1,0,1}};

        
        for(int i=0;i<8;i++){
            int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-1);

            if(hash_table.GetEntry(neigh_pos).mutex==1)
                return;
        }
        AllocScaleForEmptyBrother(scale_table, curr_pos, curr_scale);
        hash_table.Lock(curr_pos);
        //now, the cube here will be join, so we set its (never visited) neighbors' will_join flag to false
        for(int i = 1;i < 8;i++){
            int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-1);
            //if(neighbor has been join, return)
            HashEntry this_entry = hash_table.GetEntry(neigh_pos);

    
            AllocScaleForEmptyBrother(scale_table, neigh_pos, curr_scale);

            //if one cube's neighbor has a different scale, its will_join flag will never be set to true, 
            //so no need to judge again
            int before = this_entry.will_join;
            hash_table.SetWillJoinFalse(neigh_pos);
            this_entry.will_join = 0;
            hash_table.Lock(neigh_pos);

        }
       join_signal_[idx]=1;
        hash_table.SetJoinSignalTrue(curr_pos);
    }


__global__
void SetJoinSignalKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    GeometryHelper geometry_helper
    ){
        const uint idx = blockIdx.x;
        const HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;

        int curr_scale = curr_scale_[idx];
      
        if(entry.ptr!=FREE_ENTRY && entry.will_join){
            int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                               {0,0,1,1,0,0,1,1},
                               {0,1,0,1,0,1,0,1}};
            /*      
            0-----4----->x
            /     /|
            2-----6 5
            /|     |/
            y 3-----7
            */

            //choose safe blocks
            HashEntry curr_entry;
            uint free_num = 0;
            if(threadIdx.x==0){
                for(int i=0;i<8;i++){
                    int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-1);
                    HashEntry this_entry = hash_table.GetEntry(neigh_pos);
                    if(i==0)
                      curr_entry = this_entry;
                    if(this_entry.ptr == FREE_ENTRY || (this_entry.will_join == 1 && i!=0))
                      return;

                }
                join_signal_[idx]=1;

            }
        }
    }

__global__
void CheckScaleTable(
    ScaleTable scale_table
    ){
        for(int i=0;i<800000;i++){
            ScaleInfo s = scale_table.scale(i);
            if(s.scale>0)
              printf("i:%d pos:%d %d %d scale:%d\n",i,s.pos.x,s.pos.y,s.pos.z,s.scale);
        }
    }
__global__
void CheckHashTable(HashTable hash_table){

        for(int i=0;i<8000000;i++){
            HashEntry e = hash_table.entry(i);
            //hash_table.entry(i).mutex = 0;
            if(e.ptr!=FREE_ENTRY)
              printf("i:%d pos:%d %d %d mutex:%d ptr:%d\n",i,e.pos.x,e.pos.y,e.pos.z,e.mutex,e.ptr);
        }
    }
__global__
void CheckJoinSeedValid(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    GeometryHelper geometry_helper
    ){
        uint idx = blockIdx.x;
        HashEntry &entry = candidate_entries[blockIdx.x];
        int curr_scale = curr_scale_[idx];
        bool join_signal = join_signal_[idx];
        int3 curr_pos = entry.pos;
        if(join_signal == 0)
          return;
       // if(threadIdx.x!=0)
         // return;
        int n[8];
        if(entry.ptr==FREE_ENTRY)
            printf("no entry here:%d %d %d\n",entry.pos.x,entry.pos.y,entry.pos.z);
        if(entry.will_join==0)
            printf("should not join here:%d %d %d\n",entry.pos.x,entry.pos.y,entry.pos.z);
        if(entry.ptr!=FREE_ENTRY && entry.will_join){
            int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                               {0,0,1,1,0,0,1,1},
                               {0,1,0,1,0,1,0,1}};
            for(int i=0;i<8;i++){
                int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i])*pow(2,curr_scale-1);
                int neigh_scale = scale_table.GetScale(neigh_pos).scale;
                n[i] = neigh_scale;
             }
            uint3 xyz = geometry_helper.DevectorizeIndex(threadIdx.x);
            //if(curr_pos.x==4&&curr_pos.y==4&&curr_pos.z==1)
            if(n[0]==n[1]&&n[0]==n[3])
              printf("pos:%d %d %d sdf:%f(%d %d %d)->scale:%d %d %d %d %d %d %d %d\n",entry.pos.x,entry.pos.y,entry.pos.z,blocks[entry.ptr].voxels[threadIdx.x].sdf,xyz.x,xyz.y,xyz.z,n[0],n[1],n[2],n[3],n[4],n[5],n[6],n[7]);
        }
    }
__global__
void GatherValueKernel(
    BlockArray blocks,
    float* tmp_sdf,
    float* tmp_inv_sigma2
    ){
        const uint idx = blockIdx.x * 512 + threadIdx.x;
        tmp_sdf[idx] = blocks[blockIdx.x].voxels[threadIdx.x].sdf;
        tmp_inv_sigma2[idx] = blocks[blockIdx.x].voxels[threadIdx.x].inv_sigma2;
        return;
    }


__global__
void JoinBlocksKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    float* tmp_sdf,
    float* tmp_inv_sigma2,
    GeometryHelper geometry_helper
    ){
        const uint idx = blockIdx.x;

        HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;

        int curr_scale = curr_scale_[idx];
        bool join_signal = join_signal_[idx];
        
  
        if(entry.ptr!=FREE_ENTRY && entry.will_join){
            int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                               {0,0,1,1,0,0,1,1},
                               {0,1,0,1,0,1,0,1}};
            HashEntry curr_entry = hash_table.GetEntry(curr_pos);
            //start join 
            //1.add one in scale table in curr pos
            if(join_signal==0)
              return;
            if(threadIdx.x==0){
                scale_table.AddScale(curr_pos);
                curr_scale_[idx]++;
            

            }
            //2.change upper_left block to a bigger block

            HashEntry& this_entry = entry;
            Voxel &this_voxel = blocks[this_entry.ptr].voxels[threadIdx.x];

            uint3 this_xyz = geometry_helper.DevectorizeIndex(threadIdx.x);
            uint which_son = 0;
            which_son += (this_xyz.x>=4)?4:0;
            which_son += (this_xyz.y>=4)?2:0;
            which_son += (this_xyz.z>=4)?1:0;
            
            if(If_free[idx*8 + which_son] == 1){
                this_voxel.Clear();
                return;
            }
            uint3 new_xyz;
            new_xyz.x = (2*this_xyz.x)%BLOCK_SIDE_LENGTH;
            new_xyz.y = (2*this_xyz.y)%BLOCK_SIDE_LENGTH;
            new_xyz.z = (2*this_xyz.z)%BLOCK_SIDE_LENGTH;
            uint new_index = geometry_helper.VectorizeOffset(new_xyz);

            int3 son_pos = curr_pos + make_int3(neigh[0][which_son], neigh[1][which_son], neigh[2][which_son]) * pow(2,curr_scale-1);  

            HashEntry son_entry = hash_table.GetEntry(son_pos);
            Block son_block = blocks[son_entry.ptr];

           if(son_entry.ptr<0){
               printf("pos:%d %d %d son:%d %d %d from:%d %d %d->%d %d %d free:%d %d %d %d %d %d %d %d scale:%d block:%d\n",entry.pos.x,entry.pos.y,entry.pos.z,son_entry.pos.x,son_entry.pos.y,son_entry.pos.z,this_xyz.x,this_xyz.y,this_xyz.z,new_xyz.x,new_xyz.y,new_xyz.z,If_free[0],If_free[1],If_free[2],If_free[3],If_free[4],If_free[5],If_free[6],If_free[7],scale_table.GetScale(son_entry.pos).scale,son_entry.ptr);
               return;
           }
           this_voxel.sdf = tmp_sdf[son_entry.ptr * 512 + new_index];
           this_voxel.inv_sigma2 = tmp_inv_sigma2[son_entry.ptr * 512 + new_index];

          __syncthreads();

          if(threadIdx.x==0){
           for(int i=1;i<8;i++){
               int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i])*pow(2,curr_scale-1);
               scale_table.SetScaleAndAncestor(neigh_pos, curr_scale+1 ,curr_pos);

           }
          }

        }
    
    }


__global__
void AddBrothersScaleKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    GeometryHelper geometry_helper
    ){

        const uint idx = blockIdx.x;

        const HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;

        int curr_scale = curr_scale_[idx];
        bool join_signal = join_signal_[idx];
        if(entry.ptr!=FREE_ENTRY && entry.will_join){
            int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                               {0,0,1,1,0,0,1,1},
                               {0,1,0,1,0,1,0,1}};
            if(join_signal==0)
            return;

            if(threadIdx.x==0){
                //check valid

                for(int i=1;i<8;i++){
                    //   continue;
                    int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-2);
                    int new_scale = scale_table.GetScale(neigh_pos).scale;
                    bool success = scale_table.SetScaleAndAncestor(neigh_pos,curr_scale, curr_pos);
                    int3 ancest_pos = scale_table.GetAncestor(neigh_pos);
                   
                }
            }

        }
    }

__global__
void RemovePreNodeKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    GeometryHelper geometry_helper
    ){
        const uint idx = blockIdx.x;

        HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;

        int curr_scale = curr_scale_[idx];
        bool join_signal = join_signal_[idx];

       
        if(entry.ptr!=FREE_ENTRY && entry.will_join){
            int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                               {0,0,1,1,0,0,1,1},
                               {0,1,0,1,0,1,0,1}};

            if(join_signal==0)
            return;
            //3.remove the node alloced in SetJoinFlagsKernel
            if(threadIdx.x==0){
                for(int i=1;i<8;i++){
                    if(If_free[idx*8+i]==1){
                        int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-1);
                        int3 ancest_pos = scale_table.GetAncestor(neigh_pos);
                        printf("check:%d %d %d neigh_pos:%d %d %d -> ancest:%d %d %d scale:%d i:%d\n",
                              curr_pos.x,curr_pos.y,curr_pos.z,neigh_pos.x,neigh_pos.y,neigh_pos.z,ancest_pos.x,ancest_pos.y,ancest_pos.z,curr_scale,i);
                        hash_table.FreeEntry(neigh_pos);
                    }
                }
            }

        }
    }

__global__
void RemoveBrothersKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    GeometryHelper geometry_helper
    ){
        const uint idx = blockIdx.x;

        HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;

        int curr_scale = curr_scale_[idx];
        bool join_signal = join_signal_[idx];
        
        int3 ancest_pos = scale_table.GetAncestor(entry.pos);
        
        if(entry.ptr!=FREE_ENTRY && entry.will_join){
            int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                               {0,0,1,1,0,0,1,1},
                               {0,1,0,1,0,1,0,1}};

            if(join_signal==0){
                return;
            }
            //4.remove left 7 neighbors
            if(threadIdx.x==0){
                blocks[entry.ptr].ClearShell();
                for(int i=1;i<8;i++){
                    int3 neigh_pos = curr_pos + make_int3(neigh[0][i],neigh[1][i],neigh[2][i]) * pow(2,curr_scale-2);
                    int3 ancest_pos = scale_table.GetAncestor(neigh_pos);
                    int before = hash_table.GetEntry(neigh_pos).ptr;
                    
                    hash_table.WillDelete(neigh_pos);
                  
                }
              }
        }
    }

__global__
void FreeKernel(
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh_
    ){
        uint idx = blockIdx.x;

        HashEntry &entry = candidate_entries[idx];
        bool W_D = hash_table.GetEntry(entry.pos).will_delete;
        if(W_D){
            //clear vertex and triangle
            MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[threadIdx.x];
            
              //#pragma unroll 1
            for(int i=0;i<3;i++){
                if(mesh_unit.vertex_ptrs[i]!=FREE_PTR){
                    mesh_.vertex(mesh_unit.vertex_ptrs[i]).Clear();
                    mesh_.FreeVertex(mesh_unit.vertex_ptrs[i]);
                }
            }
            
             for(int i=0;i<5;i++){
                if(mesh_unit.triangle_ptrs[i]!=FREE_PTR) {
                    mesh_.triangle(mesh_unit.triangle_ptrs[i]).Clear();
                    mesh_.FreeTriangle(mesh_unit.triangle_ptrs[i]);
                }
            }
           
        }
        __syncthreads();
        if(W_D&&threadIdx.x==0){
            blocks[entry.ptr].Clear();
            hash_table.FreeEntry(entry.pos);
        }
        return;
    }

__global__
void CheckKernel(
    ScaleTable scale_table,
    HashTable hash_table,
    EntryArray candidate_entries,
    BlockArray blocks,
    int* curr_scale_,
    bool* join_signal_,
    bool* If_free,
    GeometryHelper geometry_helper
    ){
        const uint idx = blockIdx.x;
        int neigh[3][8] = {{0,0,0,0,1,1,1,1},
                           {0,0,1,1,0,0,1,1},
                           {0,1,0,1,0,1,0,1}};

        const HashEntry &entry = candidate_entries[blockIdx.x];
        int3 curr_pos = entry.pos;
        int curr_scale = curr_scale_[idx];
        bool join_signal = join_signal_[idx];
        Block block = blocks[entry.ptr];
        Voxel voxel = block.voxels[threadIdx.x];
        MeshUnit mesh_unit = block.mesh_units[threadIdx.x];
        if(entry.pos.z==0&&(mesh_unit.curr_cube_idx==0||mesh_unit.curr_cube_idx==255)){
           // printf("value:%f weight:%f pos:%d %d %d(%d)\n",voxel.sdf,voxel.inv_sigma2,entry.pos.x,entry.pos.y,entry.pos.z,threadIdx.x);
        }

    }


__global__
void SpreadCurvatureToHashKernel(
    ScaleTable scale_table,
    pcl::gpu::PtrSz<pcl::PointXYZRGB> pc,
    pcl::gpu::PtrSz<pcl::PrincipalCurvatures> cu,
    GeometryHelper geometry_helper
    ) {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        pcl::PointXYZRGB p = pc[idx];

        int3 block_pos = geometry_helper.WorldToBlock(make_float3(p.x,p.y,p.z), geometry_helper.voxel_size);
        scale_table.UpdateCurvature(block_pos, cu[idx].pc1*cu[idx].pc2);

    }

/// Condition: IsBlockInCameraFrustum
__global__
void CollectBlocksInFrustumKernel(
    HashTable hash_table,
    SensorParams sensor_params,
    float4x4     c_T_w,
    GeometryHelper geometry_helper,
    EntryArray candidate_entries
    ) {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ int local_counter;
        if (threadIdx.x == 0) local_counter = 0;
        __syncthreads();

        int addr_local = -1;
        if (idx < hash_table.entry_count
            && hash_table.entry(idx).ptr != FREE_ENTRY
            && geometry_helper.IsBlockInCameraFrustum(c_T_w, hash_table.entry(idx).pos,
                                                      sensor_params)) {
                                                          addr_local = atomicAdd(&local_counter, 1);
                                                      }
        __syncthreads();

        __shared__ int addr_global;
        if (threadIdx.x == 0 && local_counter > 0) {
            addr_global = atomicAdd(&candidate_entries.counter(),
                                    local_counter);
        }
        __syncthreads();

        if (addr_local != -1) {
            const uint addr = addr_global + addr_local;
            candidate_entries[addr] = hash_table.entry(idx);
        }
    }

__global__
void CollectAllBlocksKernel(
    HashTable hash_table,
    ScaleTable scale_table,
    EntryArray candidate_entries
    ) {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

           
        __shared__ int local_counter;
        if (threadIdx.x == 0) local_counter = 0;
        __syncthreads();

        int addr_local = -1;
        if (idx < hash_table.entry_count
            && hash_table.entry(idx).ptr != FREE_ENTRY) {
                addr_local = atomicAdd(&local_counter, 1);
            }

        __syncthreads();

        __shared__ int addr_global;
        if (threadIdx.x == 0 && local_counter > 0) {
            addr_global = atomicAdd(&candidate_entries.counter(),
                                    local_counter);
        }
        __syncthreads();

        if (addr_local != -1) {
            const uint addr = addr_global + addr_local;
            candidate_entries[addr] = hash_table.entry(idx);

            int3 aa = scale_table.GetAncestor(hash_table.entry(idx).pos);
            int3 o = hash_table.entry(idx).pos;
            if(hash_table.GetEntry(aa).ptr == FREE_ENTRY)
              printf("FREE ANCESTOR:%d %d %d->%d %d %d, ptr:%d\n",o.x,o.y,o.z,aa.x,aa.y,aa.z,hash_table.GetEntry(aa).ptr);
            if((aa.x!=o.x||aa.y!=o.y||aa.z!=o.z)&&scale_table.GetScale(aa).scale<=1)
              printf("WRONG ANCESTOR:%d %d %d->%d %d %d\n",o.x,o.y,o.z,aa.x,aa.y,aa.z);
              
        }
    }
__global__
void CheckBlockKernel(
    HashTable& hash_table,
    uint entry_count,
    EntryArray& candidate_entries,
    ScaleTable& scale_table
    ){
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        printf("idx:%d\n",idx);
        if(idx>=entry_count)
          return;
        HashEntry &entry = candidate_entries[idx];
        printf("ee:%d\n");
        int3 ancest_pos = scale_table.GetAncestor(entry.pos);

        return;
    }
////////////////////
/// Host code
///////////////////

//should change scale
double SpreadMeasurementTerm(
    ScaleTable &scale_table,
    HashTable &hash_table,
    EntryArray &candidate_entries,
    PointCloud &pc_gpu,
    Mesh& mesh_m,
    BlockArray& blocks,
    GeometryHelper geometry_helper
    ){
        Timer timer;
        timer.Tick();
        const uint threads_per_block = 256;
        uint entry_count = pc_gpu.count();
        const dim3 grid_size((entry_count + threads_per_block - 1)
                             / threads_per_block, 1);
        const dim3 block_size(threads_per_block, 1);

        uint occupied_block_count = candidate_entries.count();
        if (occupied_block_count == 0)
        return -1;
        const uint threads_per_block2 = BLOCK_SIZE;
        const dim3 grid_size2(occupied_block_count, 1);
        const dim3 block_size2(threads_per_block2, 1);

        ClearExistedJoinValueKernel <<<grid_size2, block_size2>>>(
            candidate_entries,
            mesh_m,
            blocks,
            geometry_helper);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        SpreadMeasurementTermKernel <<<grid_size, block_size >>>(
            scale_table,
            hash_table,
            pc_gpu.GetGPUPtr(),
            pc_gpu.count(),
            mesh_m,
            blocks,
            geometry_helper
        );
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        return timer.Tock();
    }

double ComputeRoughnessTerm(
    ScaleTable &scale_table,
    HashTable &hash_table,
    EntryArray &candidate_entries,
    Mesh &mesh_m,
    BlockArray &blocks,
    GeometryHelper geometry_helper
    ){
        Timer timer;
        timer.Tick();
        uint occupied_block_count = candidate_entries.count();
        if (occupied_block_count == 0)
        return -1;

        const uint threads_per_block = BLOCK_SIZE;
        const dim3 grid_size(occupied_block_count, 1);
        const dim3 block_size(threads_per_block, 1);


        // const dim3 grid_size(entry_count, 1);
        // const dim3 block_size(BLOCK_SIZE, 1);

        ComputeRoughnessTermKernel <<<grid_size, block_size >>>(
            scale_table,
            hash_table,
            candidate_entries,
            mesh_m,
            blocks,
            geometry_helper);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        GetMaxTermInAllVoxelsKernel <<<grid_size, block_size >>>(
            scale_table,
            hash_table,
            candidate_entries,
            mesh_m,
            blocks,
            geometry_helper);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        return timer.Tock();
    }


__global__
void GetBlockInformationForVisKernel(
    ScaleTable &scale_table,
    HashTable &hash_table,
    BlockArray &blocks,
    float3* pos,
    int* length,
    int* join_state,
    int entry_count,
    GeometryHelper geometry_helper
    ){

        for(int i=0;i<entry_count;i++){
            int3 curr_pos = hash_table.entry(i).pos;
            pos[i] = geometry_helper.VoxelToWorld(curr_pos);
            int curr_scale = scale_table.GetScale(curr_pos).scale;
            length[i] = curr_scale * geometry_helper.voxel_size;
            join_state[i] = hash_table.entry(i).will_join; //will_join = 1:Red, will_join = 0:Black
        }

        return;
    }

double GetBlockInformationForVis(
    ScaleTable &scale_table,
    HashTable &hash_table,
    BlockArray &blocks,
    JoinProcessVis &join_vis,
    GeometryHelper geometry_helper
    ){
        Timer timer;
        timer.Tick();
        uint entry_count = hash_table.entry_count;
        const dim3 grid_size(1,1);
        const dim3 block_size(1,1);

        float3* pos;
        int* length;
        int* join_state;
        cudaMalloc(&pos, entry_count * sizeof(float3));
        cudaMalloc(&length, entry_count * sizeof(int));
        cudaMalloc(&join_state, entry_count * sizeof(int));
        GetBlockInformationForVisKernel <<< grid_size, block_size >>>(
            scale_table,
            hash_table,
            blocks,
            pos,
            length,
            join_state,
            entry_count,
            geometry_helper);

        join_vis.Resize(entry_count);

        cudaMemcpy(join_vis.pos_toShow,pos,entry_count * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(join_vis.length_toShow,length,entry_count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(join_vis.join_state_toShow,join_state,entry_count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(pos);
        cudaFree(length);
        cudaFree(join_state);
        return timer.Tock();
    }

void CountScaleNumberSerial(
    ScaleTable scale_table,
    HashTable hash_table
    ){
        int large_size = 20;
        int* output_cpu = new int[large_size];
        int* output;
        checkCudaErrors(cudaMalloc(&output, sizeof(int)*large_size));
        checkCudaErrors(cudaMemset(output, 0, sizeof(int)*large_size));
        const dim3 grid_size(1,1);
        const dim3 block_size(1,1);
        CountScaleNumberSerialKernel<<<grid_size, block_size>>>(
            scale_table,
            hash_table,
            hash_table.entry_count,
            output,
            large_size);
        checkCudaErrors(cudaMemcpy(output_cpu, output, sizeof(int)*large_size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(output));

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        std::ofstream o("scale_counter.txt",std::ios::app);
        uint sum = 0;
        uint eight = 1;
        for(int i=1;i<large_size;i++){
            o << output_cpu[i]<<" ";
            sum += output_cpu[i]*eight;
            eight = 8 * eight;
            // std::cout<<"layer "<<i<<":"<<output_cpu[i]<<std::endl;
        }
        count_num++;
        o<<"("<<sum<<")("<<count_num<<")"<<std::endl;
        o.close();
        return;
    }

double JoinBlocks(
    ScaleTable &scale_table,
    HashTable &hash_table,
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh_,
    PointCloud pc_gpu,
    GeometryHelper geometry_helper
    ){
        Timer timer;
        timer.Tick();

        bool IfLogCountScale = false;

        const uint threads_per_block1 = 256;
        // uint entry_count = hash_table.entry_count;
        uint entry_count = candidate_entries.count();
        if(entry_count <= 0)
        return timer.Tock();
        const dim3 grid_size((entry_count + threads_per_block1 - 1)
                             / threads_per_block1, 1);
        const dim3 block_size(threads_per_block1, 1);


        // int entry_count = candidate_entries.count();
        const uint threads_per_block2 = BLOCK_SIZE;

        if(entry_count <= 0)
        return timer.Tock();

        const dim3 grid_size2(entry_count, 1);
        const dim3 block_size2(threads_per_block2, 1);

        // const dim3 grid_size(entry_count, 1);
        // const dim3 block_size(BLOCK_SIZE, 1);

        printf("entry count:%d %d\n",entry_count, threads_per_block2);

        if(IfLogCountScale){
            CountScaleNumberSerial(
                scale_table,
                hash_table
                );
        }

        // OutputBlockCorner(candidate_entries,geometry_helper);

        int* curr_scale_;
        bool* join_signal_;
        bool* If_free;
       
        checkCudaErrors(cudaMalloc(&curr_scale_, sizeof(int)*entry_count));
        checkCudaErrors(cudaMalloc(&join_signal_, sizeof(bool)*entry_count));
        checkCudaErrors(cudaMalloc(&If_free, sizeof(bool)*8*entry_count));

        ClearWillJoinKernel <<<grid_size, block_size >>>(
            scale_table,
            hash_table,
            candidate_entries);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  ClearJoinSignalKernel<<<grid_size2, block_size2 >>>(
      scale_table,
      candidate_entries,
      curr_scale_,
      join_signal_,
      If_free,
      entry_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  //alloc more nodes.

  SetJoinFlagsKernel <<<grid_size, block_size >>>(
      scale_table,
      hash_table,
      If_free,
      candidate_entries);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

 //CheckScaleTable<<<dim3(1,1),dim3(1,1)>>>(scale_table);
 hash_table.ResetNeighborMutexes();
 scale_table.ResetMutexes();
 //CheckHashTable<<<dim3(1,1),dim3(1,1)>>>(hash_table);
 
  // SetJoinSignalKernel<<<grid_size2, block_size2 >>>(
  //         scale_table,
  //         hash_table,
  //         candidate_entries,
  //         blocks,
  //         curr_scale_,
  //         join_signal_,
  //         If_free,
  //         geometry_helper);
  
  for(uint group_id = 0; group_id < 7; group_id++){
      hash_table.ResetMutexes();
      scale_table.ResetMutexes();
      AllocAllBrothersKernel<<<grid_size2, block_size2>>>(
          scale_table,
          hash_table,
          candidate_entries,
          curr_scale_,
          join_signal_,
          If_free,
          group_id,
          geometry_helper
      );
      
      hash_table.ResetMutexes();
      scale_table.ResetMutexes();
      SetJoinSignalEightTimeKernel <<<grid_size2, block_size2>>>(
          scale_table,
          hash_table,
          candidate_entries,
          blocks,
          curr_scale_,
          join_signal_,
          If_free,
          group_id,
          geometry_helper
      );
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  float* tmp_sdf;
  float* tmp_inv_sigma2;
  const dim3 block_count(blocks.count(),1);
  checkCudaErrors(cudaMalloc(&tmp_sdf, blocks.count()*threads_per_block2*sizeof(float) ));
  checkCudaErrors(cudaMalloc(&tmp_inv_sigma2, blocks.count()*threads_per_block2*sizeof(float)));
  GatherValueKernel <<<block_count, block_size2>>>(
      blocks,
      tmp_sdf,
      tmp_inv_sigma2
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  printf("block count:%d\n", blocks.count(),entry_count,threads_per_block2);
  /*
  CheckJoinSeedValid<<<grid_size2, block_size2>>>(
      scale_table,
      hash_table,
      candidate_entries,
      blocks,
      curr_scale_,
      join_signal_,
      If_free,
      geometry_helper
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());*/

  scale_table.ResetMutexes();
  JoinBlocksKernel <<<grid_size2, block_size2 >>>(
      scale_table,
      hash_table,
      candidate_entries,
      blocks,
      curr_scale_,
      join_signal_,
      If_free,
      tmp_sdf,
      tmp_inv_sigma2,
      geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());


  // if(IfLogCountScale){
  //     CountScaleNumberSerial(
  //         scale_table,
  //         hash_table
  //         );
  // }
  scale_table.ResetMutexes();
  AddBrothersScaleKernel <<<grid_size2, block_size2 >>>(
      scale_table,
      hash_table,
      candidate_entries,
      blocks,
      curr_scale_,
      join_signal_,
      If_free,
      geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

/*
   CheckKernel <<<grid_size2, block_size2 >>>(
           scale_table,
           hash_table,
           candidate_entries,
           blocks,
           curr_scale_,
           join_signal_,
           If_free,
           geometry_helper);
   checkCudaErrors(cudaDeviceSynchronize());
   checkCudaErrors(cudaGetLastError());
*/
  // if(IfLogCountScale){
  //     CountScaleNumberSerial(
  //         scale_table,
  //         hash_table
  //         );
  // }
/*
  RemovePreNodeKernel <<<grid_size2, block_size2 >>>(
      scale_table,
      hash_table,
      candidate_entries,
      blocks,
      curr_scale_,
      join_signal_,
      If_free,
      geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
*/
  // if(IfLogCountScale){
  //     CountScaleNumberSerial(
  //         scale_table,
  //         hash_table
  //         );
  // }

  hash_table.ResetMutexes();

  RemoveBrothersKernel <<<grid_size2, block_size2 >>>(
      scale_table,
      hash_table,
      candidate_entries,
      blocks,
      curr_scale_,
      join_signal_,
      If_free,
      geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  
  FreeKernel <<<grid_size2, block_size2>>>(
      hash_table,
      candidate_entries,
      blocks,
      mesh_
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  
  cudaFree(curr_scale_);
  cudaFree(join_signal_);
  cudaFree(If_free);
  cudaFree(tmp_sdf);
  cudaFree(tmp_inv_sigma2);
  

  // if(IfLogCountScale){
  //     CountScaleNumberSerial(
  //         scale_table,
  //         hash_table
  //         );
  // }
      
  return timer.Tock();
}


/// Compress discrete hash table entries
double CollectAllBlocks(
    HashTable &hash_table,
    ScaleTable &scale_table,
    EntryArray &candidate_entries
    ) {
        Timer timer;
        timer.Tick();
        LOG(INFO) << "Block count in all(before): "
        << candidate_entries.count();
        const uint threads_per_block = 256;

        uint entry_count = hash_table.entry_count;
        const dim3 grid_size((entry_count + threads_per_block - 1)
                             / threads_per_block, 1);
        const dim3 block_size(threads_per_block, 1);

        candidate_entries.reset_count();

        CollectAllBlocksKernel <<<grid_size, block_size >>>(
            hash_table,
            scale_table,
            candidate_entries);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        //std::cout<<hash_table.entry_count<<std::endl;

        // This function only to count the node number, can shut down when running
        // CountScaleNumberSerial(
        //   scale_table,
        //   hash_table
        // );
        /*
        const dim3 grid_size2((candidate_entries.count() + threads_per_block - 1)/threads_per_block, 1);
        if(candidate_entries.count()>0){
        CheckBlockKernel<<<grid_size2, block_size>>>(
            hash_table,
            candidate_entries.count(),
            candidate_entries,
            scale_table
        );
        }
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
       
        candidate_entries.reset_count();
        CollectAllBlocksKernel <<<grid_size, block_size >>>(
            hash_table,
            candidate_entries);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
       */
       
        LOG(INFO) << "Block count in all: "
        << candidate_entries.count();

        std::ofstream o("block_counter.txt",std::ios::app);
        o<<candidate_entries.count()<<std::endl;
        o.close();

        return timer.Tock();
    }

double CollectBlocksInFrustum(
    HashTable &hash_table,
    Sensor   &sensor,
    GeometryHelper &geometry_helper,
    EntryArray &candidate_entries
    ) {

        Timer timer;
        timer.Tick();
        const uint threads_per_block = 256;

        uint entry_count = hash_table.entry_count;

        const dim3 grid_size((entry_count + threads_per_block - 1)
                             / threads_per_block, 1);
        const dim3 block_size(threads_per_block, 1);

        candidate_entries.reset_count();
        CollectBlocksInFrustumKernel <<<grid_size, block_size >>>(
            hash_table,
            sensor.sensor_params(),
            sensor.cTw(),
            geometry_helper,
            candidate_entries);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        LOG(INFO) << "Block count in view frustum: "
        << candidate_entries.count();
        return timer.Tock();
    }

