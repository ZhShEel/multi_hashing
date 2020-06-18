#include "icp.h"
#include "geometry/spatial_query.h"

// Catch the cuda error

// /******************************************************
// 函数描述：计算两个点云之间最近点的距离误差,GPU核函数
// 输入参数：cloud_target目标点云矩阵，cloud_source原始点云矩阵
// 输出参数：error 最近点距离误差和误差函数的值,ConQ与P对应的控制点矩阵
// ********************************************************/

__global__ void kernelIterativeClosestPoint(float *P, float *Q, int nP, int nQ, int pointsPerThread, float *Q_select_device, int *min_index_device)
{

	//__shared__ int min_index_device[N];
	//__syncthreads();
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	printf("idx:%d %d %d %d\n",idx, pointsPerThread,nP,nQ);
	for (int i = 0; i < pointsPerThread; i++) {
		/* Handle exceptions */
		int pIdx = idx * pointsPerThread + i; // The location in P
		if (pIdx < nP) {
			/* For each point in Q */
			float minDist = 9999999; // Change this later
			int minIndex = -1;
			int pValIdx = pIdx * 3;
			for (int j = 0; j < nQ; j++) {
				int qValIdx = j * 3;
				printf("see:%d %d\n",qValIdx,j);
				float dx = P[pValIdx] - Q[qValIdx];
				float dy = P[pValIdx + 1] - Q[qValIdx + 1];
				float dz = P[pValIdx + 2] - Q[qValIdx + 2];
				float dist = sqrtf(dx*dx + dy*dy + dz*dz);
				// printf("compare:%f %f\n",dist,minDist );
				/* Update the nearest point */
				if (dist < minDist) {
					minDist = dist;
					minIndex = j;
				}
			}
			min_index_device[pIdx] = minIndex;
		}
	}
	//__syncthreads(); 
	/* Copy the data to Qselect */
	for (int i = 0; i < pointsPerThread; i++) {
		int pIdx = idx * pointsPerThread + i;
		if (pIdx < nP) {
			int qIdx = min_index_device[pIdx];
			int qValIdx = qIdx * 3;
			// printf("qq:%d %d\n",qIdx, qValIdx);
			Q_select_device[pIdx * 3] = Q[qValIdx];
			Q_select_device[pIdx * 3 + 1] = Q[qValIdx + 1];
			Q_select_device[pIdx * 3 + 2] = Q[qValIdx + 2];
		}
	}
}

__global__
void kernelIterativeClosestPointKernel(
      float* P,
      float* Q,
      int nP,
      int nQ,
      float *Q_select,
      int *min_index_device
) {

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= nP)
    return;
  //fill knn[x] - knn[x+k_num-1]
  // int* nearest_index = new int[k_num];
  // float* nearest_dis = new float[k_num];

  float minDist = 9999999;
  int minIndex = -1;
  // min_index_device[x] = -1;
  for(int j=0;j<3;j++)
    Q_select[x*3+j] = 999999;

  for(int i=0;i<nQ;i++){
    float dis = sqrt((Q[i*3]-P[x*3])*(Q[i*3]-P[x*3])
                    +(Q[i*3+1]-P[x*3+1])*(Q[i*3+1]-P[x*3+1])
                    +(Q[i*3+2]-P[x*3+2])*(Q[i*3+2]-P[x*3+2]));

    if(dis < minDist){
    	minDist = dis;
    	minIndex = i;
    }
    min_index_device[x] = minIndex;
  }

  Q_select[x*3] = Q[minIndex*3];
  Q_select[x*3+1] = Q[minIndex*3+1];
  Q_select[x*3+2] = Q[minIndex*3+2];

  return;
}


void cudaFindNearest(int numBlocks, int threadsPerBlock, float *P, float *Q, int nP, int nQ, float *Q_select, int *min_index_device) {
	/* Assign points to each thread */
	// int pointsPerThread = (nP + numBlocks * threadsPerBlock - 1) / (numBlocks * threadsPerBlock);

	//printf("%d\n", pointsPerThread);
	// kernelIterativeClosestPoint << <numBlocks, threadsPerBlock >> > (P, Q, nP, nQ, pointsPerThread, Q_select, min_index_device);
	const dim3 grid_size(nP, 1);
	const dim3 block_size(threadsPerBlock, 1);
	kernelIterativeClosestPointKernel <<< grid_size, block_size >>> (P,Q,nP,nQ,Q_select,min_index_device);
	checkCudaErrors(cudaThreadSynchronize());

}



// *****************************************************
// 函数描述：求两个点云之间的变换矩阵
// 输入参数：ConP目标点云控制点3*N，ConQ原始点云控制点3*N
// 输出参数：transformation_matrix点云之间变换参数4*4
// *******************************************************
float4x4 GetTransform(float *Pselect, float *Qselect, int nsize)
{

	Eigen::MatrixXf ConP = Eigen::Map<Eigen::MatrixXf>(Pselect, 3, nsize);
	Eigen::MatrixXf ConQ = Eigen::Map<Eigen::MatrixXf>(Qselect, 3, nsize);
	//求点云中心并移到中心点
	Eigen::VectorXf MeanP = ConP.rowwise().mean();
	Eigen::VectorXf MeanQ = ConQ.rowwise().mean();
	// std::cout << MeanP <<std::endl<< MeanQ << std::endl;
	Eigen::MatrixXf ReP = ConP.colwise() - MeanP;
	Eigen::MatrixXf ReQ = ConQ.colwise() - MeanQ;
	//求解旋转矩阵
	//Eigen::MatrixXd H = ReQ*(ReP.transpose());
	Eigen::MatrixXf H = ReP*(ReQ.transpose());
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3f U = svd.matrixU();
	Eigen::Matrix3f V = svd.matrixV();
	float det = (U * V.transpose()).determinant();
	Eigen::Vector3f diagVec(1.0, 1.0, det);
	Eigen::MatrixXf R = V * diagVec.asDiagonal() * U.transpose();
	//Eigen::MatrixXd R = H*((ReP*(ReP.transpose())).inverse());
	//求解平移向量
	Eigen::VectorXf T = MeanQ - R*MeanP;

	Eigen::MatrixXf Transmatrix = Eigen::Matrix4f::Identity();
	Transmatrix.block(0, 0, 3, 3) = R;
	Transmatrix.block(0, 3, 3, 1) = T;
	// std::cout << Transmatrix << std::endl;

	float4x4 ans;
	ans.m11 = Transmatrix(0,0);  ans.m12 = Transmatrix(0,1);  ans.m13 = Transmatrix(0,2);  ans.m14 = Transmatrix(0,3);
	ans.m21 = Transmatrix(1,0);  ans.m22 = Transmatrix(1,1);  ans.m23 = Transmatrix(1,2);  ans.m24 = Transmatrix(1,3);
	ans.m31 = Transmatrix(2,0);  ans.m32 = Transmatrix(2,1);  ans.m33 = Transmatrix(2,2);  ans.m34 = Transmatrix(2,3);
	ans.m41 = Transmatrix(3,0);  ans.m42 = Transmatrix(3,1);  ans.m43 = Transmatrix(3,2);  ans.m44 = Transmatrix(3,3);

	return ans;
}

float GetIcpError(float* P, float* Q, int number){
    float error = 0;
    for(int i=0;i<number;i++){
    	error += (P[i*3]-Q[i*3])*(P[i*3]-Q[i*3])+(P[i*3+1]-Q[i*3+1])*(P[i*3+1]-Q[i*3+1])+(P[i*3+2]-Q[i*3+2])*(P[i*3+2]-Q[i*3+2]);
    }
    return error / number;
}


// /******************************************************
// 函数描述：点云变换
// 输入参数：ConP点云控制点3*N，transformation_matrix点云之间变换参数4*4
// 输出参数：NewP新的点云控制点3*N
// ********************************************************/
void Transform(float *P, const float4x4 Transmatrix,int nsize, float *newP)
{

	////float *NewP= (float *)malloc(3*nsize * sizeof(float));
	for (int i = 0; i < nsize; i++)
	{
		int ValIdx = i * 3;
		newP[ValIdx] = Transmatrix.m11*P[ValIdx] + Transmatrix.m12*P[ValIdx + 1] + Transmatrix.m13*P[ValIdx + 2] + Transmatrix.m14;
		newP[ValIdx+1] = Transmatrix.m21*P[ValIdx] + Transmatrix.m22*P[ValIdx + 1] + Transmatrix.m23*P[ValIdx + 2] + Transmatrix.m24;
		newP[ValIdx+2] = Transmatrix.m31*P[ValIdx] + Transmatrix.m32*P[ValIdx + 1] + Transmatrix.m33*P[ValIdx + 2] + Transmatrix.m34;
	}
	//Eigen::MatrixXd ConP = Map<Eigen::MatrixXd>(P, 3, nsize);
	//Eigen::MatrixXd NewP = (R*ConP).colwise() + T;
	//newP = NewP.data();
}


__global__ void kernelTransform()
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
}

/******************************************************
函数功能：ICP算法，计算两个点云之间的转换关系
输入参数：cloud_target目标点云，cloud_source原始点云
Iter迭代参数
输出参数：transformation_matrix 转换参数
********************************************************/
void icp_process(PointCloud cloud_target_gpu,
	PointCloud cloud_source_gpu,
	const Iter_para Iter, float4x4 &transformation_matrix)
{
	int nP = cloud_target_gpu.count();
	int nQ = cloud_source_gpu.count();

    Point* cloud_target = new Point[nP];;
    Point* cloud_source = new Point[nQ];;
    cloud_target_gpu.TransferPtToHost(cloud_target);
    cloud_source_gpu.TransferPtToHost(cloud_source);
	//1.寻找P中点在Q中距离最近的点
	int p_size = sizeof(float) * nP * 3;//p size
	int q_size = sizeof(float) * nQ * 3;

	/*Data on host*/
	Eigen::MatrixXf P = Eigen::MatrixXf::Zero(nP,3);
	for(int i=0;i<nP;i++){
		P(i,0) = cloud_target[i].x;
		P(i,1) = cloud_target[i].y;
		P(i,2) = cloud_target[i].z;
	}
	Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(nQ,3);
	for(int i=0;i<nQ;i++){
		Q(i,0) = cloud_source[i].x;
		Q(i,1) = cloud_source[i].y;
		Q(i,2) = cloud_source[i].z;
	}
	float *P_origin = P.data();

	float *P_host = P.data();
	float *Q_host = Q.data();

	float * Q_select = (float *)malloc(p_size);
	/*Data on device*/
	float * P_device;
	float * Q_device;
	float * Q_selectdevice;
	int* min_index_device;

	/*Malloc space in gpu*/
	cudaMalloc(&P_device, p_size);
	cudaMalloc(&Q_device, q_size);
	cudaMalloc(&Q_selectdevice, p_size);
	cudaMalloc(&min_index_device, sizeof(int) * nP);

	/*copy data from memory to cuda*/
	cudaMemcpy(Q_device, Q_host, q_size, cudaMemcpyHostToDevice);
	/* set cuda block*/
	int numBlocks = 32;
	int threadsPerBlock =64;

	int i = 1;
	while (i < Iter.Maxiterate)
	{
		// printf("Iter: %d\n", i);
		//gpu
		// copy selectP data from memory to cuda
		cudaMemcpy(P_device, P_host, p_size, cudaMemcpyHostToDevice);
		/* Find cloest poiny in cloudsource*/
		cudaFindNearest(numBlocks, threadsPerBlock, P_device, Q_device, nP, nQ, Q_selectdevice, min_index_device);
		/* copy the Q_select*/
		cudaError_t status = cudaMemcpy(Q_select, Q_selectdevice, p_size, cudaMemcpyDeviceToHost);
		if (status == cudaSuccess) 
		{ 
			// printf("valid.\n"); 
        }
		//cpu
		//2.求解对应的刚体变换
		transformation_matrix = GetTransform(P_host, Q_select, nP);
		Transform(P_host, transformation_matrix, nP, P_host);
		float error = GetIcpError(P_host, Q_select, nP);

		////3.刚体变换的并行实现
		//float *transformation_matrix_host = transformation_matrix.data();
		//cudaMemcpy(P_device, P_host, p_size, cudaMemcpyHostToDevice);
		//cuTransform(numBlocks, threadsPerBlock, P_device, transformation_matrix, nP);

		//4.迭代上述过程直至收敛
		if (abs(error) < Iter.ControlN*Iter.acceptrate*Iter.threshold)//80%点误差小于0.01
		{
			break;
		}
		i++;
	}
	transformation_matrix = GetTransform(P_origin, P_host, nP);
	cudaFree(P_device);
	cudaFree(Q_device);
	cudaFree(Q_selectdevice); 
	cudaFree(min_index_device);
}

void print4x4Matrix(const float4x4 & matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix.m11, matrix.m12, matrix.m13);
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix.m21, matrix.m22, matrix.m23);
	printf("    | %6.3f %6.3f %6.3f | \n", matrix.m31, matrix.m32, matrix.m33);
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix.m14, matrix.m24, matrix.m34);
}

// Eigen::MatrixXd Transform(const Eigen::MatrixXd ConP, const Eigen::MatrixXd Transmatrix)
// {
// 	Eigen::MatrixXd R = Transmatrix.block(0, 0, 3, 3);
// 	Eigen::VectorXd T = Transmatrix.block(0, 3, 3, 1);

// 	Eigen::MatrixXd NewP = (R*ConP).colwise() + T;
// 	return NewP;
// }

float4x4 do_icp(PointCloud& pc_now, PointCloud& pc_last, float4x4& cTw_last){
	// Eigen::MatrixXd cloud_in = ReadFile("../data/bunny.txt");
	// Eigen::MatrixXd cloud_icp;

	// Defining a rotation matrix and translation vector
	// Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

	// A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
	// float theta = M_PI / 8;  // The angle of rotation in radians
	// transformation_matrix(0, 0) = cos(theta);
	// transformation_matrix(0, 1) = -sin(theta);
	// transformation_matrix(1, 0) = sin(theta);
	// transformation_matrix(1, 1) = cos(theta);

	// // A translation on Z axis (0.4 meters)
	// transformation_matrix(2, 3) = 0.2;
	// transformation_matrix(1, 3) = 0;
	// // Display in terminal the transformation matrix
	// std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
	// print4x4Matrix(transformation_matrix);

	// cloud_icp=Transform(cloud_in, transformation_matrix);
	
	//icp algorithm
	float4x4 matrix_icp;
	Iter_para iter;
	iter.ControlN = 35947;
	iter.Maxiterate = 20;
	iter.threshold = 0.001;
	iter.acceptrate = 0.8;
	// Getinfo();
	icp_process(pc_now, pc_last, iter, matrix_icp);
	//cout << matrix_icp << endl;
	return matrix_icp;

}





