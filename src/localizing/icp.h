
#ifndef ICP_H
#define ICP_H

#include <helper_cuda.h>
#include "core/common.h"
#include "core/hash_table.h"
#include "core/mesh.h"
#include "core/entry_array.h"
#include "core/block_array.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"
#include "core/PointCloud.h"
#include <vector>
#include <Eigen/Dense>

// #define N 35947
// //#define N 761
// #define M_PI 3.1415926

struct Iter_para //Interation paraments
{
	int ControlN;//控制点个数
	int Maxiterate;//最大迭代次数
	double threshold;//阈值
	double acceptrate;//接收率

};


float4x4 do_icp(PointCloud&, PointCloud& ,float4x4&);

#endif //ICP_H
