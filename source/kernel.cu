// Intel Compiler    ICL /O3 /QxB droppletxp2.cpp

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <cuda_runtime.h>
//#include <nvrtc_helper.h>
//#include <cudaProfiler.h>
//#include <helper_functions.h>

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>

//using namespace System;
#define D (unsigned short) 4
#define Q (unsigned short) 19
#define b (unsigned short) 24
#define tsize (unsigned short) 20
#define c2 (float) 2.0   
#define block_dim_x (int) 4
#define block_dim_y (int) 4
#define block_dim_z (int) 4

#define size_x (int) 16
#define size_y (int) 72
#define size_z (int) 16


#define square (bool) false  //
#define strip (bool) false  //
#define begingravity (long) 0
#define beginboundary (long) 0
#define rho_eq (float) 2.6
#define edg (int) 6

#define CUDA (bool) true
//#define print_cuda (bool) true
//#define Debug  (bool) true
//#define print_cache (bool) true

#define viewstep (unsigned long)  500
#define printstep (unsigned long) 500
#define readwall (bool) true	
#define u_zero (bool) false
#define floattruncate (float) 1e-12
#define damp0 (float) 1.0
#define noslip (bool) true
#define removemomentum (int) 5
#define PI (float) 3.1415926535897932384626433832795

#define d0 (float) 0.33333333333333333333333  //float d0 = 1.0 / 3.0;
#define d1 (float) 0.5                        //float d1 = 1.0 / c2;
#define d2 (float) 0.02777777777777777777777777777//d2 = (1.0 - d0) / b;
#define d3 (float) 0.083333333333333333333333//float d3 = D / (c2*b);
#define d4 (float) 0.125                      //float d4 = D*(D + 2) / (2 * c2*c2*b);
#define d5 (float) 0.04166666666666666667     //float d5 = D / (2 * b*c2);


#define psi_0 (float) 0.6
#define psi_eq (float) 0.6//  = psi_0;
#define T_LA (float) 0.16
#define T_LS1 (float) 0.064// 0.4*T_LA;
#define T_LS2 (float) 0.064// 0.4*T_LA;

#define t1 (int) 8
#define t2 (int) 9
#define t3 (int) 8

unsigned short x_cen = 8;
unsigned short y_cen = 32;
unsigned short z_cen = 8;

float gr = 0.00000;  /////////////////////////
unsigned long time_limit = 200000;   ////////////////////////////////////////////////////////////////////////////////
int fileindex = 100003;
bool readtext = false;	/////////////////////////
float r  = 20;
float lx = 4;
float ly = 40;
float lz = 40; 
float tao = 1;

float gx = 0.0;
float gy = 5e-5;
float gz = 0.0;
__device__ const float3 g_cuda = { 0.0f, 1e-4f, 0.0f };

float width = (float)size_z - 2.0*edg;
float cv = 0.4;




float phi0 = 1;
float rho_sum = 0;
float gask = (float) (0.1 / rho_eq);

//float speedlimit = sqrt(2.0/3.0);
//bool pause = false;

unsigned short k_shift = 0;



void action(void);
void action_gpu(void);
void printparameters(void);
void initialize(void);
void initwall(void);
void clearlist(void);
void printcrosssection(void);
void printprofile(void);
void loadvalues(int index1);
void position(void);
void printvalues(void);
void zero_tmp(void);
void zero_v(void);
void draw_img(unsigned long time_index);
void check_data(void);
void init_wall_cache(void);

float expff(float val);

float tsample[tsize];
float n[size_x][size_y][size_z] = { 0 };
float ux[size_x][size_y][size_z] = { 0 };
float uy[size_x][size_y][size_z] = { 0 };
float uz[size_x][size_y][size_z] = { 0 };
float psi[size_x][size_y][size_z] = { 0 };
bool wall[size_x][size_y][size_z] = { false };
float3 u[size_x][size_y][size_z] = { 0 };
int count[size_x][size_y][size_z] = { 0 };

float  n_tmp[size_x][size_y][size_z] = { 0 };
float ux_tmp[size_x][size_y][size_z] = { 0 };
float uy_tmp[size_x][size_y][size_z] = { 0 };
float uz_tmp[size_x][size_y][size_z] = { 0 };
float psi_tmp[size_x][size_y][size_z] = { 0 };
float3 u_tmp[size_x][size_y][size_z] = { 0 };
float3 netv = { 0.0f, 0.0f, 0.0f };
float cache_tmp[size_x][size_y][size_z][19] = { 0 };
unsigned int wall_cache[size_x][size_y][size_z] = {0};

float exptable[100];
float netvy_ave;
float maxcache, mincache, finishtime;
float rcx_cen, rcx_edg, acx_cen, acx_edg;
float netvx, netvy, netvz;
float hrc, hac;
unsigned long runtime = 0;
unsigned short cur, nex;


float rho_sum_cache[viewstep] = { 0 };
float netvx_cache[viewstep] = { 0 };
float netvy_cache[viewstep] = { 0 };
float netvz_cache[viewstep] = { 0 };
float hac_cache[viewstep] = { 0 };
float hrc_cache[viewstep] = { 0 };
float netvy_ave_cache[viewstep] = { 0 };
float netvy_ave_all = 0;
float hrc_ave = 0;
float hac_ave = 0;
float acca_cache = 0;
float rcca_cache = 0;
clock_t start, c_now, c_last, finish;


void cuda_run(cudaError_t my_err);

__device__ float3 netv_cuda ;
__device__ float rho_sum_cuda ;

void check_data(void){
	int i, j, k;
	FILE *errfile = fopen("error.log", "wt");


	bool test[10] = { true };
	for (k = 0; k < size_z; k++){
		for (j = 0; j < size_y; j++){
			for (i = 0; i < size_x; i++){
				test[0] = fabs(n[i][j][k] - n_tmp[i][j][k]) > (0.01 * n[i][j][k]);
				test[1] = fabs(psi[i][j][k] - psi_tmp[i][j][k]) > (0.01 * psi[i][j][k]);
				test[2] = fabs(ux[i][j][k] - u_tmp[i][j][k].x) >(0.01 * ux[i][j][k]);
				test[3] = fabs(uy[i][j][k] - u_tmp[i][j][k].y) >(0.01 * uy[i][j][k]);
				test[4] = fabs(uz[i][j][k] - u_tmp[i][j][k].z) >(0.01 * uz[i][j][k]);
				fprintf(errfile, "%i\t%i\t%i\t%i\n", i, j, k, count[i][j][k]);
				if (test[0] || test[1]){// || test[2] || test[3] || test[4]){
					//fprintf (errfile, "%i\t%i\t%i\t\t%i\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\n", i, j, k, test[0], test[1], test[2], test[3], test[4], n[i][j][k], n_tmp[i][j][k], psi[i][j][k], psi_tmp[i][j][k]);
					

					
					printf("%i\t%i\t%i\t\t%i\t%i\t%i\t%i\t%i\t%f\t%f\t%f\t%f\n", i, j, k, test[0], test[1], test[2], test[3], test[4], n[i][j][k], n_tmp[i][j][k], psi[i][j][k], psi_tmp[i][j][k]);
					/**
						if ((i == 5) && (j == 56) && (k == 10)){
						printf("ux %f %f uy %f %f uz %f %f\n", ux[i][j][k], u_tmp[i][j][k].x, uy[i][j][k] , u_tmp[i][j][k].y, uz[i][j][k] , u_tmp[i][j][k].z);
					
						test[5] = true;
					}**/


				}
			}
		}
	}
	fclose(errfile);
}

void printparameters(void){
	FILE *fpara = fopen("parameter.out", "wt");

	fprintf(fpara, "gr=%.12lf\nruntime=%i\nD=%i\nQ=%i\nb=%i\nc2=%.12lf\nSize_x=%i\nSize_y=%i\nSize_z=%i\ntime_limit=%i\nrho_eq=%.12lf\n\n\n", gr, runtime, D, Q, b, c2, size_x, size_y, size_z, time_limit, rho_eq);
//	fprintf(fpara, "r=%.12lf\ntao=%.12lf\nd0=%.12lf\ngx=%.12lf\ngy=%.12lf\ngz=%.12lf\nT_LA=%.12lf\npsi_0=%.12lf\npsi_eq=%.12lf\nT_LS1=%.12lf\nT_LS2=%.12lf\ngask=%.12lf\nx_cen=%i\ny_cen=%i\nz_cen=%i\n\n", r, tao, d0, gx, gy, gz, T_LA, psi_0, psi_eq, T_LS1, T_LS2, gask, x_cen, y_cen, z_cen);

	fprintf(fpara, "d1=%.12lf\nd2=%.12lf\nd3=%.12lf\nd4=%.12lf\nd5=%.12lf\n\n", d1, d2, d3, d4, d5);
	fclose(fpara);
}

void initialize(void){
	
	int count;
	for (count = 0; count < 100; count++){
		exptable[count] = (float)exp(-float(count) / 20.0);
	}


	if (readtext){
		loadvalues(fileindex);
	}
	else {
		initwall();
	}

	init_wall_cache();

	netvy_ave_all = 0;
	hrc_ave = 0;
	hac_ave = 0;
	acca_cache = 0;
	rcca_cache = 0;
}

__global__ void testing(void){
	printf("My ID = { %d, %d, %d }\tMy Block = { %d, %d, %d }\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

__device__ __forceinline__ int index(int x, int y, int z) {
	return x * size_y*size_z + y * size_z + z;
}

__global__ void action_gpu_kernel(float *n, float3 *u, float *psi, bool *wall, float *cache_tmp){

	float n_rho, n_rho2;
	float vx, vy, vz, v2;
	float vxx, vyy, vzz;

	float d3vx, d3vy, d3vz;
	float d4vxx, d4vyy, d4vzz;
	float d4vxy2, d4vyz2, d4vxz2;
	float d2d5, d4xxyy, d4yyzz, d4zzxx;
	float psila_x;
	float psila_y;
	float psila_z;


	int i = (block_dim_x * blockIdx.x + threadIdx.x + size_x - 1) % size_x;
	int j = (block_dim_y * blockIdx.y + threadIdx.y + size_y - 1) % size_y;
	int k = (block_dim_z * blockIdx.z + threadIdx.z + size_z - 1) % size_z;

	int my_i = threadIdx.x;
	int my_j = threadIdx.y;
	int my_k = threadIdx.z;

	int my_i_p, my_i_n;
	int my_j_p, my_j_n;
	int my_k_p, my_k_n;

	int ijk = index(i, j, k);
	int ijkc = ijk*19;
	float n_ijk = n[ijk];
	float psi_ijk = psi[ijk];
	float3 u_ijk = u[ijk];

	float T_LA_psi;
	
	__shared__ float psi_cache[block_dim_x][block_dim_y][block_dim_z];


	psi_cache[my_i][my_j][my_k] = psi_ijk;

	//*********

	my_i_p = my_i + 1;
	my_i_n = my_i - 1;
	my_j_p = my_j + 1;
	my_j_n = my_j - 1;
	my_k_p = my_k + 1;
	my_k_n = my_k - 1;

	//********** Load values to shared memory


	__syncthreads();



	

	if (( my_i > 0 && my_i < ( block_dim_x - 1 ) ) && ( my_j > 0 && my_j < ( block_dim_y - 1 ) ) && ( my_k > 0 && my_k < ( block_dim_z - 1 ) )){
		if (!wall[ijk]){
			T_LA_psi = T_LA * psi_cache[my_i][my_j][my_k];

			psila_x = T_LA_psi * (2 * (psi_cache[my_i_p][my_j][my_k] - psi_cache[my_i_n][my_j][my_k]) + psi_cache[my_i_p][my_j_p][my_k] + psi_cache[my_i_p][my_j_n][my_k] + psi_cache[my_i_p][my_j][my_k_p] + psi_cache[my_i_p][my_j][my_k_n] - psi_cache[my_i_n][my_j_p][my_k] - psi_cache[my_i_n][my_j_n][my_k] - psi_cache[my_i_n][my_j][my_k_p] - psi_cache[my_i_n][my_j][my_k_n]);
			psila_y = T_LA_psi * (2 * (psi_cache[my_i][my_j_p][my_k] - psi_cache[my_i][my_j_n][my_k]) + psi_cache[my_i_p][my_j_p][my_k] + psi_cache[my_i_n][my_j_p][my_k] + psi_cache[my_i][my_j_p][my_k_p] + psi_cache[my_i][my_j_p][my_k_n] - psi_cache[my_i_p][my_j_n][my_k] - psi_cache[my_i_n][my_j_n][my_k] - psi_cache[my_i][my_j_n][my_k_p] - psi_cache[my_i][my_j_n][my_k_n]);
			psila_z = T_LA_psi * (2 * (psi_cache[my_i][my_j][my_k_p] - psi_cache[my_i][my_j][my_k_n]) + psi_cache[my_i_p][my_j][my_k_p] + psi_cache[my_i_n][my_j][my_k_p] + psi_cache[my_i][my_j_p][my_k_p] + psi_cache[my_i][my_j_n][my_k_p] - psi_cache[my_i_p][my_j][my_k_n] - psi_cache[my_i_n][my_j][my_k_n] - psi_cache[my_i][my_j_p][my_k_n] - psi_cache[my_i][my_j_n][my_k_n]);

			vx = g_cuda.x + (u_ijk.x + psila_x) / n_ijk;
			vy = g_cuda.y + (u_ijk.y + psila_y) / n_ijk;
			vz = g_cuda.z + (u_ijk.z + psila_z) / n_ijk;

			vxx = vx*vx;
			vyy = vy*vy;
			vzz = vz*vz;

			v2 = vxx + vyy + vzz;

			d3vx = d3*vx;
			d3vy = d3*vy;
			d3vz = d3*vz;

			d4vxx = d4*vxx;
			d4vyy = d4*vyy;
			d4vzz = d4*vzz;

			d4vxy2 = 2 * d4*vx*vy;
			d4vyz2 = 2 * d4*vy*vz;
			d4vxz2 = 2 * d4*vz*vx;

			d2d5 = d2 - d5*v2;

			d4xxyy = d4vxx + d4vyy;
			d4yyzz = d4vyy + d4vzz;
			d4zzxx = d4vzz + d4vxx;

			n_rho = n_ijk;
			n_rho2 = 2 * n_rho;



			cache_tmp[ijkc + 0] = n_rho2 * (d2d5 + d3vx + d4vxx);
			cache_tmp[ijkc + 1] = n_rho2 * (d2d5 - d3vx + d4vxx);

			cache_tmp[ijkc + 2] = n_rho2 * (d2d5 + d3vy + d4vyy);
			cache_tmp[ijkc + 3] = n_rho2 * (d2d5 - d3vy + d4vyy);

			cache_tmp[ijkc + 4] = n_rho2 * (d2d5 + d3vz + d4vzz);
			cache_tmp[ijkc + 5] = n_rho2 * (d2d5 - d3vz + d4vzz);

			cache_tmp[ijkc + 6] = n_rho * (d2d5 + d3vx + d3vy + d4xxyy + d4vxy2);
			cache_tmp[ijkc + 7] = n_rho * (d2d5 + d3vx - d3vy + d4xxyy - d4vxy2);
			cache_tmp[ijkc + 8] = n_rho * (d2d5 - d3vx + d3vy + d4xxyy - d4vxy2);
			cache_tmp[ijkc + 9] = n_rho * (d2d5 - d3vx - d3vy + d4xxyy + d4vxy2);

			cache_tmp[ijkc + 10] = n_rho * (d2d5 + d3vx + d3vz + d4zzxx + d4vxz2);
			cache_tmp[ijkc + 11] = n_rho * (d2d5 + d3vx - d3vz + d4zzxx - d4vxz2);
			cache_tmp[ijkc + 12] = n_rho * (d2d5 - d3vx + d3vz + d4zzxx - d4vxz2);
			cache_tmp[ijkc + 13] = n_rho * (d2d5 - d3vx - d3vz + d4zzxx + d4vxz2);

			cache_tmp[ijkc + 14] = n_rho * (d2d5 + d3vy + d3vz + d4yyzz + d4vyz2);
			cache_tmp[ijkc + 15] = n_rho * (d2d5 + d3vy - d3vz + d4yyzz - d4vyz2);
			cache_tmp[ijkc + 16] = n_rho * (d2d5 - d3vy + d3vz + d4yyzz - d4vyz2);
			cache_tmp[ijkc + 17] = n_rho * (d2d5 - d3vy - d3vz + d4yyzz + d4vyz2);
			
			cache_tmp[ijkc + 18] = n_rho * (d0 - d1*v2);
		}
	}
}

__global__ void action_s2_gpu_kernel(float *n, float3 *u, float *psi, bool *wall, float *cache_tmp){

	float n_rho, n_rho2;
	float vx, vy, vz, v2;
	float vxx, vyy, vzz;

	float d3vx, d3vy, d3vz;
	float d4vxx, d4vyy, d4vzz;
	float d4vxy2, d4vyz2, d4vxz2;
	float d2d5, d4xxyy, d4yyzz, d4zzxx;
	float psila_x;
	float psila_y;
	float psila_z;

	int my_i = threadIdx.x;
	int my_j = threadIdx.y;
	int my_k = threadIdx.z;

	int i = ((block_dim_x) * blockIdx.x + my_i + size_x - 1) % size_x;
	int j = ((block_dim_y) * blockIdx.y + my_j + size_y - 1) % size_y;
	int k = ((block_dim_z) * blockIdx.z + my_k + size_z - 1) % size_z;

	int my_i_p, my_i_n;
	int my_j_p, my_j_n;
	int my_k_p, my_k_n;

	int ijk = index(i, j, k);
	int ijkc = ijk*19;
	float n_ijk = n[ijk];
	float n_ijk_inv = 1.0/n_ijk;
	float psi_ijk = psi[ijk];
	float3 u_ijk = u[ijk];

	float T_LA_psi;
	
	__shared__ float psi_cache[block_dim_z+2][block_dim_y+2][block_dim_x+2];


	psi_cache[my_k][my_j][my_i] = psi_ijk;

	//*********

	my_i_p = my_i + 1;
	my_i_n = my_i - 1;
	my_j_p = my_j + 1;
	my_j_n = my_j - 1;
	my_k_p = my_k + 1;
	my_k_n = my_k - 1;

	//********** Load values to shared memory


	__syncthreads();



	

	if (( my_i > 0 && (my_i <=  block_dim_x ) ) && ( my_j > 0 && my_j <= block_dim_y ) && ( my_k > 0 && my_k <= block_dim_z )){
	
			T_LA_psi = T_LA * psi_ijk;

			/**/ float psi_i_p_j_k = psi_cache[my_k][my_j][my_i_p];
			/**/ float psi_i_n_j_k = psi_cache[my_k][my_j][my_i_n];
			/**/ float psi_i_j_p_k = psi_cache[my_k][my_j_p][my_i];
			/**/ float psi_i_j_n_k = psi_cache[my_k][my_j_n][my_i];
			/**/ float psi_i_j_k_p = psi_cache[my_k_p][my_j][my_i];
			/**/ float psi_i_j_k_n = psi_cache[my_k_n][my_j][my_i];
			/**/ float psi_i_p_j_p_k = psi_cache[my_k][my_j_p][my_i_p];
			/**/ float psi_i_p_j_n_k = psi_cache[my_k][my_j_n][my_i_p];
			/**/ float psi_i_p_j_k_p = psi_cache[my_k_p][my_j][my_i_p];
			/**/ float psi_i_p_j_k_n = psi_cache[my_k_n][my_j][my_i_p];
			/**/ float psi_i_n_j_p_k = psi_cache[my_k][my_j_p][my_i_n];
			/**/ float psi_i_n_j_n_k = psi_cache[my_k][my_j_n][my_i_n];
			/**/ float psi_i_n_j_k_p = psi_cache[my_k_p][my_j][my_i_n];
			/**/ float psi_i_n_j_k_n = psi_cache[my_k_n][my_j][my_i_n];
			/**/ float psi_i_j_p_k_p = psi_cache[my_k_p][my_j_p][my_i];
			/**/ float psi_i_j_p_k_n = psi_cache[my_k_n][my_j_p][my_i];
			/**/ float psi_i_j_n_k_p = psi_cache[my_k_p][my_j_n][my_i];
			/**/ float psi_i_j_n_k_n = psi_cache[my_k_n][my_j_n][my_i];


			psila_x = T_LA_psi * (2 * (psi_i_p_j_k - psi_i_n_j_k) + psi_i_p_j_p_k + psi_i_p_j_n_k + psi_i_p_j_k_p + psi_i_p_j_k_n - psi_i_n_j_p_k - psi_i_n_j_n_k - psi_i_n_j_k_p - psi_i_n_j_k_n);
			psila_y = T_LA_psi * (2 * (psi_i_j_p_k - psi_i_j_n_k) + psi_i_p_j_p_k + psi_i_n_j_p_k + psi_i_j_p_k_p + psi_i_j_p_k_n - psi_i_p_j_n_k - psi_i_n_j_n_k - psi_i_j_n_k_p - psi_i_j_n_k_n);
			psila_z = T_LA_psi * (2 * (psi_i_j_k_p - psi_i_j_k_n) + psi_i_p_j_k_p + psi_i_n_j_k_p + psi_i_j_p_k_p + psi_i_j_n_k_p - psi_i_p_j_k_n - psi_i_n_j_k_n - psi_i_j_p_k_n - psi_i_j_n_k_n);

			vx = g_cuda.x + (u_ijk.x + psila_x) * n_ijk_inv;
			vy = g_cuda.y + (u_ijk.y + psila_y) * n_ijk_inv;
			vz = g_cuda.z + (u_ijk.z + psila_z) * n_ijk_inv;

			vxx = vx*vx;
			vyy = vy*vy;
			vzz = vz*vz;

			v2 = vxx + vyy + vzz;

			d3vx = d3*vx;
			d3vy = d3*vy;
			d3vz = d3*vz;

			d4vxx = d4*vxx;
			d4vyy = d4*vyy;
			d4vzz = d4*vzz;

			d4vxy2 = 2 * d4*vx*vy;
			d4vyz2 = 2 * d4*vy*vz;
			d4vxz2 = 2 * d4*vz*vx;

			d2d5 = d2 - d5*v2;

			d4xxyy = d4vxx + d4vyy;
			d4yyzz = d4vyy + d4vzz;
			d4zzxx = d4vzz + d4vxx;

			n_rho = n_ijk * (int)(!wall[ijk]);
			n_rho2 = 2 * n_rho;



			cache_tmp[ijkc + 0] = n_rho2 * (d2d5 + d3vx + d4vxx);
			cache_tmp[ijkc + 1] = n_rho2 * (d2d5 - d3vx + d4vxx);

			cache_tmp[ijkc + 2] = n_rho2 * (d2d5 + d3vy + d4vyy);
			cache_tmp[ijkc + 3] = n_rho2 * (d2d5 - d3vy + d4vyy);

			cache_tmp[ijkc + 4] = n_rho2 * (d2d5 + d3vz + d4vzz);
			cache_tmp[ijkc + 5] = n_rho2 * (d2d5 - d3vz + d4vzz);

			cache_tmp[ijkc + 6] = n_rho * (d2d5 + d3vx + d3vy + d4xxyy + d4vxy2);
			cache_tmp[ijkc + 7] = n_rho * (d2d5 + d3vx - d3vy + d4xxyy - d4vxy2);
			cache_tmp[ijkc + 8] = n_rho * (d2d5 - d3vx + d3vy + d4xxyy - d4vxy2);
			cache_tmp[ijkc + 9] = n_rho * (d2d5 - d3vx - d3vy + d4xxyy + d4vxy2);

			cache_tmp[ijkc + 10] = n_rho * (d2d5 + d3vx + d3vz + d4zzxx + d4vxz2);
			cache_tmp[ijkc + 11] = n_rho * (d2d5 + d3vx - d3vz + d4zzxx - d4vxz2);
			cache_tmp[ijkc + 12] = n_rho * (d2d5 - d3vx + d3vz + d4zzxx - d4vxz2);
			cache_tmp[ijkc + 13] = n_rho * (d2d5 - d3vx - d3vz + d4zzxx + d4vxz2);

			cache_tmp[ijkc + 14] = n_rho * (d2d5 + d3vy + d3vz + d4yyzz + d4vyz2);
			cache_tmp[ijkc + 15] = n_rho * (d2d5 + d3vy - d3vz + d4yyzz - d4vyz2);
			cache_tmp[ijkc + 16] = n_rho * (d2d5 - d3vy + d3vz + d4yyzz - d4vyz2);
			cache_tmp[ijkc + 17] = n_rho * (d2d5 - d3vy - d3vz + d4yyzz + d4vyz2);
			
			cache_tmp[ijkc + 18] = n_rho * (d0 - d1*v2);
		
	}
}

__global__ void action_new_i_gpu_kernel(float *n, float3 *u, float *psi, bool *wall, float *cache_tmp){

	/**/ float n_rho, n_rho2;
	/**/ float vx, vy, vz, v2;
	/**/ float vxx, vyy, vzz;

	/**/ float d3vx, d3vy, d3vz;
	/**/ float d4vxx, d4vyy, d4vzz;
	/**/ float d4vxy2, d4vyz2, d4vxz2;
	/**/ float d2d5, d4xxyy, d4yyzz, d4zzxx;
	/**/ float psila_x;
	/**/ float psila_y;
	/**/ float psila_z;

	
	/**/ int my_i = threadIdx.x;
	/**/ int my_j = threadIdx.y;
	/**/ int my_k = threadIdx.z;

	/**/ int i = block_dim_x * blockIdx.x + my_i;
	/**/ int j = block_dim_y * blockIdx.y + my_j;
	/**/ int k = block_dim_z * blockIdx.z + my_k;

	int i_0 = i;
	int j_0 = j;
	int k_0 = k;

	/**/ int i_p, i_n;
	/**/ int j_p, j_n;
	/**/ int k_p, k_n;

	/**/ int ijk;
	/**/ int ijkc;
	/**/ float n_ijk;
	

	/**/ float T_LA_psi;
	unsigned long long time[10];
	
	//__shared__ float psi_cache[block_dim_x][block_dim_y][block_dim_z];


	//psi_cache[my_i][my_j][my_k] = psi_ijk;

	//*********
	time[0] = clock();

	i_p = ((i + 1) % size_x) * size_y*size_z;
	i_n = ((i + size_x - 1) % size_x) * size_y*size_z;
	i *=  size_y*size_z;
	
	j_p = ((j + 1) % size_y) * size_z;
	j_n = ((j + size_y - 1) % size_y) * size_z;

	k_p = (k + 1) % size_z;
	k_n = (k + size_z - 1) % size_z;
	j *= size_z;

	ijk = i + j + k;
	ijkc = ijk * 19;

	time[1] = clock();

	//u_ijk = u[ijk];
	n_ijk = n[ijk];

	time[2] = clock();

	volatile float3 u_ijk= u[ijk];
	float n_ijk_inv = 1.0f/n_ijk;

	time[3] = clock();
	//********** Load values to shared memory


	
	if (!wall[ijk]){
		T_LA_psi = T_LA * psi[ijk];

		time[4] = clock();

		volatile float psi_i_p_j_k = psi[i_p+j+k];
		volatile float psi_i_n_j_k = psi[i_n+j+k];
		volatile float psi_i_j_p_k = psi[i+j_p+k];
		volatile float psi_i_j_n_k = psi[i+j_n+k];
		volatile float psi_i_j_k_p = psi[i+j+k_p];
		volatile float psi_i_j_k_n = psi[i+j+k_n];
		volatile float psi_i_p_j_p_k = psi[i_p+j_p+k];
		volatile float psi_i_p_j_n_k = psi[i_p+j_n+k];
		volatile float psi_i_p_j_k_p = psi[i_p+j+k_p];
		volatile float psi_i_p_j_k_n = psi[i_p+j+k_n];
  		volatile float psi_i_n_j_p_k = psi[i_n+j_p+k];
		volatile float psi_i_n_j_n_k = psi[i_n+j_n+k];
		volatile float psi_i_n_j_k_p = psi[i_n+j+k_p];
		volatile float psi_i_n_j_k_n = psi[i_n+j+k_n];
		volatile float psi_i_j_p_k_p = psi[i+j_p+k_p];
		volatile float psi_i_j_p_k_n = psi[i+j_p+k_n];
		volatile float psi_i_j_n_k_p = psi[i+j_n+k_p];
		volatile float psi_i_j_n_k_n = psi[i+j_n+k_n];

		time[5] = clock();

		psila_x = T_LA_psi * (2 * (psi_i_p_j_k - psi_i_n_j_k) + psi_i_p_j_p_k + psi_i_p_j_n_k + psi_i_p_j_k_p + psi_i_p_j_k_n - psi_i_n_j_p_k - psi_i_n_j_n_k - psi_i_n_j_k_p - psi_i_n_j_k_n);
		psila_y = T_LA_psi * (2 * (psi_i_j_p_k - psi_i_j_n_k) + psi_i_p_j_p_k + psi_i_n_j_p_k + psi_i_j_p_k_p + psi_i_j_p_k_n - psi_i_p_j_n_k - psi_i_n_j_n_k - psi_i_j_n_k_p - psi_i_j_n_k_n);
		psila_z = T_LA_psi * (2 * (psi_i_j_k_p - psi_i_j_k_n) + psi_i_p_j_k_p + psi_i_n_j_k_p + psi_i_j_p_k_p + psi_i_j_n_k_p - psi_i_p_j_k_n - psi_i_n_j_k_n - psi_i_j_p_k_n - psi_i_j_n_k_n);

		time[6] = clock();
		//psila_x = T_LA_psi * (2 * (psi[i_p+j+k] - psi[i_n+j+k]) + psi[i_p+j_p+k] + psi[i_p+j_n+k] + psi[i_p+j+k_p] + psi[i_p+j+k_n] - psi[i_n+j_p+k] - psi[i_n+j_n+k] - psi[i_n+j+k_p] - psi[i_n+j+k_n]);
		//psila_y = T_LA_psi * (2 * (psi[i+j_p+k] - psi[i+j_n+k]) + psi[i_p+j_p+k] + psi[i_n+j_p+k] + psi[i+j_p+k_p] + psi[i+j_p+k_n] - psi[i_p+j_n+k] - psi[i_n+j_n+k] - psi[i+j_n+k_p] - psi[i+j_n+k_n]);
		//psila_z = T_LA_psi * (2 * (psi[i+j+k_p] - psi[i+j+k_n]) + psi[i_p+j+k_p] + psi[i_n+j+k_p] + psi[i+j_p+k_p] + psi[i+j_n+k_p] - psi[i_p+j+k_n] - psi[i_n+j+k_n] - psi[i+j_p+k_n] - psi[i+j_n+k_n]);

		vx = g_cuda.x + (u_ijk.x + psila_x) * n_ijk_inv;
		vy = g_cuda.y + (u_ijk.y + psila_y) * n_ijk_inv;
		vz = g_cuda.z + (u_ijk.z + psila_z) * n_ijk_inv;

		vxx = vx*vx;
		vyy = vy*vy;
		vzz = vz*vz;

		v2 = vxx + vyy + vzz;

		d3vx = d3*vx;
		d3vy = d3*vy;
		d3vz = d3*vz;

		d4vxx = d4*vxx;
		d4vyy = d4*vyy;
		d4vzz = d4*vzz;

		d4vxy2 = 2 * d4*vx*vy;
		d4vyz2 = 2 * d4*vy*vz;
		d4vxz2 = 2 * d4*vz*vx;

		d2d5 = d2 - d5*v2;

		d4xxyy = d4vxx + d4vyy;
		d4yyzz = d4vyy + d4vzz;
		d4zzxx = d4vzz + d4vxx;

		n_rho = n_ijk;
		n_rho2 = 2 * n_rho;

		time[7] = clock();

		cache_tmp[ijkc + 0] = n_rho2 * (d2d5 + d3vx + d4vxx);
		cache_tmp[ijkc + 1] = n_rho2 * (d2d5 - d3vx + d4vxx);

		cache_tmp[ijkc + 2] = n_rho2 * (d2d5 + d3vy + d4vyy);
		cache_tmp[ijkc + 3] = n_rho2 * (d2d5 - d3vy + d4vyy);

		cache_tmp[ijkc + 4] = n_rho2 * (d2d5 + d3vz + d4vzz);
		cache_tmp[ijkc + 5] = n_rho2 * (d2d5 - d3vz + d4vzz);

		cache_tmp[ijkc + 6] = n_rho * (d2d5 + d3vx + d3vy + d4xxyy + d4vxy2);
		cache_tmp[ijkc + 7] = n_rho * (d2d5 + d3vx - d3vy + d4xxyy - d4vxy2);
		cache_tmp[ijkc + 8] = n_rho * (d2d5 - d3vx + d3vy + d4xxyy - d4vxy2);
		cache_tmp[ijkc + 9] = n_rho * (d2d5 - d3vx - d3vy + d4xxyy + d4vxy2);

		cache_tmp[ijkc + 10] = n_rho * (d2d5 + d3vx + d3vz + d4zzxx + d4vxz2);
		cache_tmp[ijkc + 11] = n_rho * (d2d5 + d3vx - d3vz + d4zzxx - d4vxz2);
		cache_tmp[ijkc + 12] = n_rho * (d2d5 - d3vx + d3vz + d4zzxx - d4vxz2);
		cache_tmp[ijkc + 13] = n_rho * (d2d5 - d3vx - d3vz + d4zzxx + d4vxz2);

		cache_tmp[ijkc + 14] = n_rho * (d2d5 + d3vy + d3vz + d4yyzz + d4vyz2);
		cache_tmp[ijkc + 15] = n_rho * (d2d5 + d3vy - d3vz + d4yyzz - d4vyz2);
		cache_tmp[ijkc + 16] = n_rho * (d2d5 - d3vy + d3vz + d4yyzz - d4vyz2);
		cache_tmp[ijkc + 17] = n_rho * (d2d5 - d3vy - d3vz + d4yyzz + d4vyz2);
			
		cache_tmp[ijkc + 18] = n_rho * (d0 - d1*v2);
		
		time[8] = clock();

/**		if (i_0 == t1 && j_0 == t2 && k_0 == t3){
			for (int i1 = 1; i1 < 9; i1++){
//				printf("%i : %llu\n", i1, time[i1]-time[i1-1]);
			}
			
		}**/
	}
	
}

__global__ void action_new_s_gpu_kernel(float *n, float3 *u, float *psi, bool *wall, float *cache_tmp){

	/**/ float n_rho, n_rho2;
	/**/ float vx, vy, vz, v2;
	/**/ float vxx, vyy, vzz;

	/**/ float d3vx, d3vy, d3vz;
	/**/ float d4vxx, d4vyy, d4vzz;
	/**/ float d4vxy2, d4vyz2, d4vxz2;
	/**/ float d2d5, d4xxyy, d4yyzz, d4zzxx;
	/**/ float psila_x;
	/**/ float psila_y;
	/**/ float psila_z;

	/**/ int my_i = threadIdx.x;
	/**/ int my_j = threadIdx.y;
	/**/ int my_k = threadIdx.z;

	/**/ int i = block_dim_x * blockIdx.x + my_i;
	/**/ int j = block_dim_y * blockIdx.y + my_j;
	/**/ int k = block_dim_z * blockIdx.z + my_k;

	/**/ int i_0 = i;
	/**/ int j_0 = j;
	/**/ int k_0 = k;

	/**/ int i_p, i_n;
	/**/ int j_p, j_n;
	/**/ int k_p, k_n;

	/**/ int ijk;
	/**/ int ijkc;
	/**/ float n_ijk;
	

	/**/ float T_LA_psi;
	// unsigned long long time[10];

	//__shared__ float psi_cache[(block_dim_x + 2) * (block_dim_y + 2) * (block_dim_z + 2)];


	//psi_cache[my_i][my_j][my_k] = psi_ijk;

	//*********
	// time[0] = clock();

	i_p = ((i + 1) % size_x) * size_y * size_z;
	i_n = ((i + size_x - 1) % size_x) * size_y * size_z;
	i *=  size_y * size_z;
	
	j_p = ((j + 1) % size_y) * size_z;
	j_n = ((j + size_y - 1) % size_y) * size_z;
	j *= size_z;

	k_p = (k + 1) % size_z;
	k_n = (k + size_z - 1) % size_z;


	ijk = i + j + k;
	ijkc = ijk * 19;

	// time[1] = clock();

	//u_ijk = u[ijk];
	n_ijk = n[ijk];

	// time[2] = clock();

	volatile float3 u_ijk= u[ijk];
	float n_ijk_inv = 1.0f/n_ijk;

	// time[3] = clock();
	//********** Load values to shared memory
	//int wall_switch = !wall[ijk];

	
	//if (!wall[ijk]){
	T_LA_psi = T_LA * psi[ijk];

	// time[4] = clock();

	/**/ float psi_i_p_j_k = psi[i_p+j+k];
	/**/ float psi_i_n_j_k = psi[i_n+j+k];
	/**/ float psi_i_j_p_k = psi[i+j_p+k];
	/**/ float psi_i_j_n_k = psi[i+j_n+k];
	/**/ float psi_i_j_k_p = psi[i+j+k_p];
	/**/ float psi_i_j_k_n = psi[i+j+k_n];
	/**/ float psi_i_p_j_p_k = psi[i_p+j_p+k];
	/**/ float psi_i_p_j_n_k = psi[i_p+j_n+k];
	/**/ float psi_i_p_j_k_p = psi[i_p+j+k_p];
	/**/ float psi_i_p_j_k_n = psi[i_p+j+k_n];
  	/**/ float psi_i_n_j_p_k = psi[i_n+j_p+k];
	/**/ float psi_i_n_j_n_k = psi[i_n+j_n+k];
	/**/ float psi_i_n_j_k_p = psi[i_n+j+k_p];
	/**/ float psi_i_n_j_k_n = psi[i_n+j+k_n];
	/**/ float psi_i_j_p_k_p = psi[i+j_p+k_p];
	/**/ float psi_i_j_p_k_n = psi[i+j_p+k_n];
	/**/ float psi_i_j_n_k_p = psi[i+j_n+k_p];
	/**/ float psi_i_j_n_k_n = psi[i+j_n+k_n];
	   
	// time[5] = clock();

	psila_x = T_LA_psi * (2 * (psi_i_p_j_k - psi_i_n_j_k) + psi_i_p_j_p_k + psi_i_p_j_n_k + psi_i_p_j_k_p + psi_i_p_j_k_n - psi_i_n_j_p_k - psi_i_n_j_n_k - psi_i_n_j_k_p - psi_i_n_j_k_n);
	psila_y = T_LA_psi * (2 * (psi_i_j_p_k - psi_i_j_n_k) + psi_i_p_j_p_k + psi_i_n_j_p_k + psi_i_j_p_k_p + psi_i_j_p_k_n - psi_i_p_j_n_k - psi_i_n_j_n_k - psi_i_j_n_k_p - psi_i_j_n_k_n);
	psila_z = T_LA_psi * (2 * (psi_i_j_k_p - psi_i_j_k_n) + psi_i_p_j_k_p + psi_i_n_j_k_p + psi_i_j_p_k_p + psi_i_j_n_k_p - psi_i_p_j_k_n - psi_i_n_j_k_n - psi_i_j_p_k_n - psi_i_j_n_k_n);

	// time[6] = clock();
	//psila_x = T_LA_psi * (2 * (psi[i_p+j+k] - psi[i_n+j+k]) + psi[i_p+j_p+k] + psi[i_p+j_n+k] + psi[i_p+j+k_p] + psi[i_p+j+k_n] - psi[i_n+j_p+k] - psi[i_n+j_n+k] - psi[i_n+j+k_p] - psi[i_n+j+k_n]);
	//psila_y = T_LA_psi * (2 * (psi[i+j_p+k] - psi[i+j_n+k]) + psi[i_p+j_p+k] + psi[i_n+j_p+k] + psi[i+j_p+k_p] + psi[i+j_p+k_n] - psi[i_p+j_n+k] - psi[i_n+j_n+k] - psi[i+j_n+k_p] - psi[i+j_n+k_n]);
	//psila_z = T_LA_psi * (2 * (psi[i+j+k_p] - psi[i+j+k_n]) + psi[i_p+j+k_p] + psi[i_n+j+k_p] + psi[i+j_p+k_p] + psi[i+j_n+k_p] - psi[i_p+j+k_n] - psi[i_n+j+k_n] - psi[i+j_p+k_n] - psi[i+j_n+k_n]);

	vx = (g_cuda.x + (u_ijk.x + psila_x) * n_ijk_inv);
	vy = (g_cuda.y + (u_ijk.y + psila_y) * n_ijk_inv);
	vz = (g_cuda.z + (u_ijk.z + psila_z) * n_ijk_inv);

	vxx = vx*vx;
	vyy = vy*vy;
	vzz = vz*vz;

	v2 = vxx + vyy + vzz;

	d3vx = d3*vx;
	d3vy = d3*vy;
	d3vz = d3*vz;

	d4vxx = d4*vxx;
	d4vyy = d4*vyy;
	d4vzz = d4*vzz;

	d4vxy2 = 2 * d4*vx*vy;
	d4vyz2 = 2 * d4*vy*vz;
	d4vxz2 = 2 * d4*vz*vx;

	d2d5 = d2 - d5*v2;

	d4xxyy = d4vxx + d4vyy;
	d4yyzz = d4vyy + d4vzz;
	d4zzxx = d4vzz + d4vxx;

	n_rho = n_ijk * (!wall[ijk]);
	n_rho2 = 2 * n_rho;

	// time[7] = clock();

	cache_tmp[ijkc + 0] = n_rho2 * (d2d5 + d3vx + d4vxx);
	cache_tmp[ijkc + 1] = n_rho2 * (d2d5 - d3vx + d4vxx);

	cache_tmp[ijkc + 2] = n_rho2 * (d2d5 + d3vy + d4vyy);
	cache_tmp[ijkc + 3] = n_rho2 * (d2d5 - d3vy + d4vyy);

	cache_tmp[ijkc + 4] = n_rho2 * (d2d5 + d3vz + d4vzz);
	cache_tmp[ijkc + 5] = n_rho2 * (d2d5 - d3vz + d4vzz);

	cache_tmp[ijkc + 6] = n_rho * (d2d5 + d3vx + d3vy + d4xxyy + d4vxy2);
	cache_tmp[ijkc + 7] = n_rho * (d2d5 + d3vx - d3vy + d4xxyy - d4vxy2);
	cache_tmp[ijkc + 8] = n_rho * (d2d5 - d3vx + d3vy + d4xxyy - d4vxy2);
	cache_tmp[ijkc + 9] = n_rho * (d2d5 - d3vx - d3vy + d4xxyy + d4vxy2);

	cache_tmp[ijkc + 10] = n_rho * (d2d5 + d3vx + d3vz + d4zzxx + d4vxz2);
	cache_tmp[ijkc + 11] = n_rho * (d2d5 + d3vx - d3vz + d4zzxx - d4vxz2);
	cache_tmp[ijkc + 12] = n_rho * (d2d5 - d3vx + d3vz + d4zzxx - d4vxz2);
	cache_tmp[ijkc + 13] = n_rho * (d2d5 - d3vx - d3vz + d4zzxx + d4vxz2);

	cache_tmp[ijkc + 14] = n_rho * (d2d5 + d3vy + d3vz + d4yyzz + d4vyz2);
	cache_tmp[ijkc + 15] = n_rho * (d2d5 + d3vy - d3vz + d4yyzz - d4vyz2);
	cache_tmp[ijkc + 16] = n_rho * (d2d5 - d3vy + d3vz + d4yyzz - d4vyz2);
	cache_tmp[ijkc + 17] = n_rho * (d2d5 - d3vy - d3vz + d4yyzz + d4vyz2);
			
	cache_tmp[ijkc + 18] = n_rho * (d0 - d1*v2);
		
	// time[8] = clock();
	/**
	if (i_0 == t1 && j_0 == t2 && k_0 == t3){
		for (int i1 = 1; i1 < 9; i1++){
			printf("%i : %llu\n", i1, time[i1]- time[i1-1]);
		}		
	}	**/
	//}
}
 
__global__ void tranaction_new_gpu_kernel(float *n, float3 *u, float *psi, bool *wall, float *cache_tmp){

	 volatile int i = block_dim_x * blockIdx.x + threadIdx.x;
	 volatile int j = block_dim_y * blockIdx.y + threadIdx.y;
	 volatile int k = block_dim_z * blockIdx.z + threadIdx.z;

	 volatile int i_p, i_n;
	 volatile int j_p, j_n;
	 volatile int k_p, k_n;

	 volatile int my_i = threadIdx.x;
	 volatile int my_j = threadIdx.y;
	 volatile int my_k = threadIdx.z;

	 volatile int ijk;
	 volatile int ijkc;

	 volatile float tmp_n = 0.0f;
	 volatile float n_tmp_cache_0 = 0.0f;
	 volatile float3 u_tmp_cache_0 = {0.0f, 0.0f, 0.0f};


	

	//*********
	
	i_p = ((i + size_x + 1) % size_x) * size_y * size_z;
	i_n = ((i + size_x - 1) % size_x) * size_y * size_z;
	i *=  size_y * size_z;
	
	j_p = ((j + size_y + 1) % size_y) * size_z;
	j_n = ((j + size_y - 1) % size_y) * size_z;

	k_p = (k + size_z + 1) % size_z;
	k_n = (k + size_z - 1) % size_z;
	j *= size_z;

	ijk = i + j + k;
	ijkc = ijk * 19;

	//********** Load values to shared memory


	

	if (!wall[ijk]){
		
		//count[ijk]++;
			
 		if (!wall[(i_n + j + k)]){
			tmp_n =  cache_tmp[(i_n + j + k) * 19 + 0 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
		}else if (noslip) {
			tmp_n = cache_tmp[ijkc + 1];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_p + j + k)]){
			tmp_n =  cache_tmp[(i_p + j + k) * 19 + 1 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
		}
		else if (noslip){
			tmp_n =  cache_tmp[ijkc + 0];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i + j_n + k)]){
			tmp_n = cache_tmp[(i + j_n + k) * 19 + 2 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 3];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i + j_p + k)]){
			tmp_n = cache_tmp[(i + j_p + k) * 19 + 3 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 2];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i + j + k_n)]){
			tmp_n = cache_tmp[(i + j + k_n) * 19 + 4 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 5];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i + j + k_p)]){
			tmp_n = cache_tmp[(i + j + k_p) * 19 + 5 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 4];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_n + j_n + k)]){
			tmp_n = cache_tmp[(i_n + j_n + k) * 19 + 6 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 9];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}
		//  __syncthreads();

		if (!wall[(i_n + j_p + k)]){
			tmp_n = cache_tmp[(i_n + j_p + k) * 19 + 7 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 8];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_p + j_n + k)]){
			tmp_n = cache_tmp[(i_p + j_n + k) * 19 + 8 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 7];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_p + j_p + k)]){
			tmp_n = cache_tmp[(i_p + j_p + k) * 19 + 9 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 6];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_n + j + k_n)]){
			tmp_n = cache_tmp[(i_n + j + k_n) * 19 + 10 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 13];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_n + j + k_p)]){
			tmp_n = cache_tmp[(i_n + j + k_p) * 19 + 11 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 12];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_p + j + k_n)]){
			tmp_n = cache_tmp[(i_p + j + k_n) * 19 + 12 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 11];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i_p + j + k_p)]){
			tmp_n = cache_tmp[(i_p + j + k_p) * 19 + 13 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 10];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i + j_n + k_n)]){
			tmp_n = cache_tmp[(i + j_n + k_n) * 19 + 14 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 17];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i + j_n + k_p)]){
			tmp_n = cache_tmp[(i + j_n + k_p) * 19 + 15 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 16];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();
	
		if (!wall[(i + j_p + k_n)]){
			tmp_n = cache_tmp[(i + j_p + k_n) * 19 + 16 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 15];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

		if (!wall[(i + j_p + k_p)]){
			tmp_n = cache_tmp[(i + j_p + k_p) * 19 + 17 ];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}
		else if (noslip){
			tmp_n = cache_tmp[ijkc + 14];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}
															  
		//  __syncthreads();

		n[ijk]   = n_tmp_cache_0 + cache_tmp[ijkc + 18 ];
		u[ijk].x = u_tmp_cache_0.x;
		u[ijk].y = u_tmp_cache_0.y;
		u[ijk].z = u_tmp_cache_0.z;
		psi[ijk] = 1.0f - expf(-n[ijk]);
		
	}

}

__global__ void tranaction_new_i_gpu_kernel(float *n, float3 *u, float *psi, float *cache_tmp, unsigned int *wall_cache){
	 
	 /**/ int my_i = threadIdx.x;
	 /**/ int my_j = threadIdx.y;
	 /**/ int my_k = threadIdx.z;

	 /**/ int i = block_dim_x * blockIdx.x + my_i;
	 /**/ int j = block_dim_y * blockIdx.y + my_j;
	 /**/ int k = block_dim_z * blockIdx.z + my_k;

	 /**/ int i_p, i_n;
	 /**/ int j_p, j_n;
	 /**/ int k_p, k_n;


	 /**/ int ijk = index(i,j,k);
	 /**/ int ijkc = ijk * 19;

	 /**/ float tmp_n = 0.0f;
	 /**/ float n_tmp_cache_0 = 0.0f;
	 /**/ float3 u_tmp_cache_0 = {0.0f, 0.0f, 0.0f};

	 /**/ unsigned int my_wall = wall_cache[ijk];
	

	//*********
	
	i_p = (i + 1) % size_x;// * size_y * size_z;
	i_n = (i + size_x - 1) % size_x;// * size_y * size_z;
	//i *=  size_y * size_z;
	
	j_p = (j + 1) % size_y;// * size_z;
	j_n = (j + size_y - 1) % size_y;// * size_z;

	k_p = (k + 1) % size_z;
	k_n = (k + size_z - 1) % size_z;
	//j *= size_z;



	//********** Load values to shared memory


	

	if (!(my_wall & 1)){
		
		//count[ijk]++;
			

		tmp_n =  cache_tmp[index(i_n,j,k) * 19 + 0 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x += tmp_n;

		if (my_wall & 2) {
			tmp_n = cache_tmp[ijkc + 1];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
		}

		//  __syncthreads();

		
		tmp_n =  cache_tmp[index(i_p,j,k) * 19 + 1 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x -= tmp_n;
		
		if (my_wall & 4){
			tmp_n =  cache_tmp[ijkc + 0];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
		}

		//  __syncthreads();

		
		tmp_n = cache_tmp[index(i,j_n,k) * 19 + 2 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.y += tmp_n;
		
		if (my_wall & 8){
			tmp_n = cache_tmp[ijkc + 3];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}

		//  __syncthreads();

 
		tmp_n = cache_tmp[index(i,j_p,k) * 19 + 3 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.y -= tmp_n;
	 
		if (my_wall & 16){
			tmp_n = cache_tmp[ijkc + 2];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}

		//  __syncthreads();

 
		tmp_n = cache_tmp[index(i,j,k_n) * 19 + 4 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.z += tmp_n;
	 
		if (my_wall & 32){
			tmp_n = cache_tmp[ijkc + 5];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

 
		tmp_n = cache_tmp[index(i,j,k_p) * 19 + 5 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.z -= tmp_n;
		 
		if (my_wall & 64){
			tmp_n = cache_tmp[ijkc + 4];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();

	 
		tmp_n = cache_tmp[index(i_n,j_n,k) * 19 + 6 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x += tmp_n;
		u_tmp_cache_0.y += tmp_n;
	 
		if (my_wall & 128){
			tmp_n = cache_tmp[ijkc + 9];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}
		//  __syncthreads();

	 
		tmp_n = cache_tmp[index(i_n,j_p,k) * 19 + 7 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x += tmp_n;
		u_tmp_cache_0.y -= tmp_n;
	 
		if (my_wall & 256){
			tmp_n = cache_tmp[ijkc + 8];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}

		//  __syncthreads();

	 
		tmp_n = cache_tmp[index(i_p,j_n,k) * 19 + 8 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x -= tmp_n;
		u_tmp_cache_0.y += tmp_n;
	 
		if (my_wall & 512){
			tmp_n = cache_tmp[ijkc + 7];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.y += tmp_n;
		}

		//  __syncthreads();

 
		tmp_n = cache_tmp[index(i_p,j_p,k) * 19 + 9 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x -= tmp_n;
		u_tmp_cache_0.y -= tmp_n;
	 
		if (my_wall & 1024){
			tmp_n = cache_tmp[ijkc + 6];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.y -= tmp_n;
		}

		//  __syncthreads();

	 
		tmp_n = cache_tmp[index(i_n,j,k_n) * 19 + 10 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x += tmp_n;
		u_tmp_cache_0.z += tmp_n;
	 
		if (my_wall & 2048){
			tmp_n = cache_tmp[ijkc + 13];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

	 
		tmp_n = cache_tmp[index(i_n,j,k_p) * 19 + 11 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x += tmp_n;
		u_tmp_cache_0.z -= tmp_n;
		 
		if (my_wall & 4096){
			tmp_n = cache_tmp[ijkc + 12];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();

 
		tmp_n = cache_tmp[index(i_p,j,k_n) * 19 + 12 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x -= tmp_n;
		u_tmp_cache_0.z += tmp_n;
	 
		if (my_wall & 8192){
			tmp_n = cache_tmp[ijkc + 11];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

 
		tmp_n = cache_tmp[index(i_p,j,k_p) * 19 + 13 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.x -= tmp_n;
		u_tmp_cache_0.z -= tmp_n;
 
		if (my_wall & 16384){
			tmp_n = cache_tmp[ijkc + 10];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.x -= tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();

 
		tmp_n = cache_tmp[index(i,j_n,k_n) * 19 + 14 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.y += tmp_n;
		u_tmp_cache_0.z += tmp_n;
	 
		if (my_wall & 32768){
			tmp_n = cache_tmp[ijkc + 17];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

	 
		tmp_n = cache_tmp[index(i,j_n,k_p) * 19 + 15 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.y += tmp_n;
		u_tmp_cache_0.z -= tmp_n;
	 
		if (my_wall & 65536){
			tmp_n = cache_tmp[ijkc + 16];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y += tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}

		//  __syncthreads();
	
	 
		tmp_n = cache_tmp[index(i,j_p,k_n) * 19 + 16 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.y -= tmp_n;
		u_tmp_cache_0.z += tmp_n;
	 
		if (my_wall & 131072){
			tmp_n = cache_tmp[ijkc + 15];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
			u_tmp_cache_0.z += tmp_n;
		}

		//  __syncthreads();

	 
		tmp_n = cache_tmp[index(i,j_p,k_p) * 19 + 17 ];
		n_tmp_cache_0   += tmp_n;
		u_tmp_cache_0.y -= tmp_n;
		u_tmp_cache_0.z -= tmp_n;
	 
		if (my_wall & 262144){
			tmp_n = cache_tmp[ijkc + 14];
			n_tmp_cache_0   += tmp_n;
			u_tmp_cache_0.y -= tmp_n;
			u_tmp_cache_0.z -= tmp_n;
		}
															  
		//  __syncthreads();

		n[ijk]   = n_tmp_cache_0 + cache_tmp[ijkc + 18];
		u[ijk].x = u_tmp_cache_0.x;
		u[ijk].y = u_tmp_cache_0.y;
		u[ijk].z = u_tmp_cache_0.z;
		psi[ijk] = 1.0f - __expf(-n[ijk]);
		
	}

}

__global__ void init_tmp(float *n, float *n_tmp, float3 *u_tmp, float *psi, bool *wall){
	int i = block_dim_x * blockIdx.x + threadIdx.x;
	int j = block_dim_y * blockIdx.y + threadIdx.y;
	int k = block_dim_z * blockIdx.z + threadIdx.z;
	int ijk = index(i, j, k);

	//printf("My ID = { %d, %d, %d }\tMy Block = { %d, %d, %d }\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

	if (!wall[ijk]){


		n_tmp[ijk] = 0.0;
		u_tmp[ijk].x = 0.0;
		u_tmp[ijk].y = 0.0;
		u_tmp[ijk].z = 0.0;

		rho_sum_cuda = 0.0;
		netv_cuda.x = 0.0;
		netv_cuda.y = 0.0;
		netv_cuda.z = 0.0;

		//psi[ijk] = (float)1.0 - expf(-n[ijk]);
		

	}
																																										   
}

__global__ void commit_tmp(float *n, float3 *u, float *n_tmp, float3 *u_tmp, float *psi, bool *wall){
	int i = block_dim_x * blockIdx.x + threadIdx.x;
	int j = block_dim_y * blockIdx.y + threadIdx.y;
	int k = block_dim_z * blockIdx.z + threadIdx.z;
	int ijk = index(i, j, k);

	//printf("My ID = { %d, %d, %d }\tMy Block = { %d, %d, %d }\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

	if (!wall[ijk]){


		n[ijk] =   n_tmp[ijk];
		u[ijk].x = u_tmp[ijk].x;
		u[ijk].y = u_tmp[ijk].y;
		u[ijk].z = u_tmp[ijk].z;

		rho_sum_cuda += n[ijk];
		netv_cuda.x += u[ijk].x;
		netv_cuda.y += u[ijk].y;
		netv_cuda.z += u[ijk].z;

		//if (n[ijk] < 1e-5){ printf("Low Value"); }

		psi[ijk] = (float)1.0 - expf(-n[ijk]);
		


	}

	__syncthreads();
}

void action(void){
	long i, j, k;
	int i_p, i_n, j_p, j_n, k_p, k_n, k_nn, k_pp;
	float n_rho, n_rho2;
	float vx, vy, vz, v2;
	float vxx, vyy, vzz;

	float d3vx, d3vy, d3vz;
	float d4vxx, d4vyy, d4vzz;
	float d4vxy2, d4vyz2, d4vxz2;
	float d2d5, d4xxyy, d4yyzz, d4zzxx;
	float psila_x;
	float psila_y;
	float psila_z;
	float cache[19];
	
	rho_sum = 0;
	netvx = 0;
	netvy = 0;
	netvz = 0;


	for (k = 0; k<size_z; k++){

		//***************************************

		//if ((long)runtime > beginboundary){

		k_p = k + 1;
		k_n = k - 1;
		k_nn = k - 2;
		k_pp = k + 2;

		if (k_p == size_z) k_p = 0;
		if (k_n == -1) k_n = size_z - 1;


		//***************************************

		for (j = 0; j<size_y; j++){
			//***************************************
			j_p = j + 1;
			j_n = j - 1;
			if (j_p == size_y) j_p = 0;
			if (j_n == -1) j_n = size_y - 1;
			//***************************************

			for (i = 0; i<size_x; i++){
				float T_LA_psi;

				//***************************************
				i_p = i + 1;
				i_n = i - 1;

				if (i_p == size_x) i_p = 0;
				if (i_n == -1) i_n = size_x - 1;
				//***************************************

				if (k_pp < size_z - 1){//if ((k_pp < size_z) && (k_pp < size_z-1)){
					n_tmp[i][j][k_pp] = 0;
					ux_tmp[i][j][k_pp] = 0;
					uy_tmp[i][j][k_pp] = 0;
					uz_tmp[i][j][k_pp] = 0;
				}



				if (!wall[i][j][k]){
					//float cachesum = 0;
					//float rhocache = 0;
					//int cacheint;



					T_LA_psi = T_LA * psi[i][j][k];

					psila_x = T_LA_psi * (2 * (psi[i_p][j][k] - psi[i_n][j][k]) + psi[i_p][j_p][k] + psi[i_p][j_n][k] + psi[i_p][j][k_p] + psi[i_p][j][k_n] - psi[i_n][j_p][k] - psi[i_n][j_n][k] - psi[i_n][j][k_p] - psi[i_n][j][k_n]);
					psila_y = T_LA_psi * (2 * (psi[i][j_p][k] - psi[i][j_n][k]) + psi[i_p][j_p][k] + psi[i_n][j_p][k] + psi[i][j_p][k_p] + psi[i][j_p][k_n] - psi[i_p][j_n][k] - psi[i_n][j_n][k] - psi[i][j_n][k_p] - psi[i][j_n][k_n]);
					psila_z = T_LA_psi * (2 * (psi[i][j][k_p] - psi[i][j][k_n]) + psi[i_p][j][k_p] + psi[i_n][j][k_p] + psi[i][j_p][k_p] + psi[i][j_n][k_p] - psi[i_p][j][k_n] - psi[i_n][j][k_n] - psi[i][j_p][k_n] - psi[i][j_n][k_n]);


					vx = gx + (ux[i][j][k] + psila_x) / n[i][j][k];
					vy = gy + (uy[i][j][k] + psila_y) / n[i][j][k];
					vz = gz + (uz[i][j][k] + psila_z) / n[i][j][k];


					vxx = vx*vx;
					vyy = vy*vy;
					vzz = vz*vz;

					v2 = vxx + vyy + vzz;

					d3vx = d3*vx;
					d3vy = d3*vy;
					d3vz = d3*vz;

					d4vxx = d4*vxx;
					d4vyy = d4*vyy;
					d4vzz = d4*vzz;

					d4vxy2 = 2 * d4*vx*vy;
					d4vyz2 = 2 * d4*vy*vz;
					d4vxz2 = 2 * d4*vz*vx;

					d2d5 = d2 - d5*v2;

					d4xxyy = d4vxx + d4vyy;
					d4yyzz = d4vyy + d4vzz;
					d4zzxx = d4vzz + d4vxx;

					n_rho = n[i][j][k];
					n_rho2 = 2 * n_rho;



					cache[0] = n_rho2 * (d2d5 + d3vx + d4vxx);
					cache[1] = n_rho2 * (d2d5 - d3vx + d4vxx);

					cache[2] = n_rho2 * (d2d5 + d3vy + d4vyy);
					cache[3] = n_rho2 * (d2d5 - d3vy + d4vyy);

					cache[4] = n_rho2 * (d2d5 + d3vz + d4vzz);
					cache[5] = n_rho2 * (d2d5 - d3vz + d4vzz);

					cache[6] = n_rho * (d2d5 + d3vx + d3vy + d4xxyy + d4vxy2);
					cache[7] = n_rho * (d2d5 + d3vx - d3vy + d4xxyy - d4vxy2);
					cache[8] = n_rho * (d2d5 - d3vx + d3vy + d4xxyy - d4vxy2);
					cache[9] = n_rho * (d2d5 - d3vx - d3vy + d4xxyy + d4vxy2);

					cache[10] = n_rho * (d2d5 + d3vx + d3vz + d4zzxx + d4vxz2);
					cache[11] = n_rho * (d2d5 + d3vx - d3vz + d4zzxx - d4vxz2);
					cache[12] = n_rho * (d2d5 - d3vx + d3vz + d4zzxx - d4vxz2);
					cache[13] = n_rho * (d2d5 - d3vx - d3vz + d4zzxx + d4vxz2);

					cache[14] = n_rho * (d2d5 + d3vy + d3vz + d4yyzz + d4vyz2);
					cache[15] = n_rho * (d2d5 + d3vy - d3vz + d4yyzz - d4vyz2);
					cache[16] = n_rho * (d2d5 - d3vy + d3vz + d4yyzz - d4vyz2);
					cache[17] = n_rho * (d2d5 - d3vy - d3vz + d4yyzz + d4vyz2);

					cache[18] = n_rho * (d0 - d1*v2);

					n_tmp[i][j][k] += cache[18];

					if (!wall[i_p][j][k]){
						n_tmp[i_p][j][k] += cache[0];
						ux_tmp[i_p][j][k] += cache[0];

						//rhocache += cache[0];
					}
					else if (noslip) {
						n_tmp[i][j][k] += cache[0];
						ux_tmp[i][j][k] -= cache[0];

						//rhocache += cache[0];

						//printf("0 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);

					}

					if (!wall[i_n][j][k]){
						n_tmp[i_n][j][k] += cache[1];
						ux_tmp[i_n][j][k] -= cache[1];

						//rhocache += cache[1];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[1];
						ux_tmp[i][j][k] += cache[1];

						//rhocache += cache[1];

						//printf("1 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i][j_p][k]){
						n_tmp[i][j_p][k] += cache[2];
						uy_tmp[i][j_p][k] += cache[2];

						//rhocache += cache[2];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[2];
						uy_tmp[i][j][k] -= cache[2];

						//rhocache += cache[2];

						//printf("2 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i][j_n][k]){
						n_tmp[i][j_n][k] += cache[3];
						uy_tmp[i][j_n][k] -= cache[3];

						//rhocache += cache[3];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[3];
						uy_tmp[i][j][k] += cache[3];

						//rhocache += cache[3];

						//printf("3 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i][j][k_p]){
						n_tmp[i][j][k_p] += cache[4];
						uz_tmp[i][j][k_p] += cache[4];

						//rhocache += cache[4];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[4];
						uz_tmp[i][j][k] -= cache[4];

						//rhocache += cache[4];

						//printf("4 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i][j][k_n]){
						n_tmp[i][j][k_n] += cache[5];
						uz_tmp[i][j][k_n] -= cache[5];

						//rhocache += cache[5];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[5];
						uz_tmp[i][j][k] += cache[5];

						//rhocache += cache[5];

						//printf("5 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i_p][j_p][k]){
						n_tmp[i_p][j_p][k] += cache[6];
						ux_tmp[i_p][j_p][k] += cache[6];
						uy_tmp[i_p][j_p][k] += cache[6];

						//rhocache += cache[6];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[6];
						ux_tmp[i][j][k] -= cache[6];
						uy_tmp[i][j][k] -= cache[6];

						//rhocache += cache[6];

						//printf("6 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}


					if (!wall[i_p][j_n][k]){
						n_tmp[i_p][j_n][k] += cache[7];
						ux_tmp[i_p][j_n][k] += cache[7];
						uy_tmp[i_p][j_n][k] -= cache[7];

						//rhocache += cache[7];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[7];
						ux_tmp[i][j][k] -= cache[7];
						uy_tmp[i][j][k] += cache[7];

						//rhocache += cache[7];

						//printf("7 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i_n][j_p][k]){
						n_tmp[i_n][j_p][k] += cache[8];
						ux_tmp[i_n][j_p][k] -= cache[8];
						uy_tmp[i_n][j_p][k] += cache[8];

						//rhocache += cache[8];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[8];
						ux_tmp[i][j][k] += cache[8];
						uy_tmp[i][j][k] -= cache[8];

						//rhocache += cache[8];

						//printf("8 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i_n][j_n][k]){
						n_tmp[i_n][j_n][k] += cache[9];
						ux_tmp[i_n][j_n][k] -= cache[9];
						uy_tmp[i_n][j_n][k] -= cache[9];

						//rhocache += cache[9];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[9];
						ux_tmp[i][j][k] += cache[9];
						uy_tmp[i][j][k] += cache[9];

						//rhocache += cache[9];

						//printf("9 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i_p][j][k_p]){
						n_tmp[i_p][j][k_p] += cache[10];
						ux_tmp[i_p][j][k_p] += cache[10];
						uz_tmp[i_p][j][k_p] += cache[10];

						//rhocache += cache[10];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[10];
						ux_tmp[i][j][k] -= cache[10];
						uz_tmp[i][j][k] -= cache[10];

						//rhocache += cache[10];

						//printf("10 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i_p][j][k_n]){
						n_tmp[i_p][j][k_n] += cache[11];
						ux_tmp[i_p][j][k_n] += cache[11];
						uz_tmp[i_p][j][k_n] -= cache[11];

						//rhocache += cache[11];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[11];
						ux_tmp[i][j][k] -= cache[11];
						uz_tmp[i][j][k] += cache[11];

						//rhocache += cache[11];

						//printf("11 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i_n][j][k_p]){
						n_tmp[i_n][j][k_p] += cache[12];
						ux_tmp[i_n][j][k_p] -= cache[12];
						uz_tmp[i_n][j][k_p] += cache[12];

						//rhocache += cache[12];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[12];
						ux_tmp[i][j][k] += cache[12];
						uz_tmp[i][j][k] -= cache[12];

						//rhocache += cache[12];

						//printf("12 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i_n][j][k_n]){
						n_tmp[i_n][j][k_n] += cache[13];
						ux_tmp[i_n][j][k_n] -= cache[13];
						uz_tmp[i_n][j][k_n] -= cache[13];

						//rhocache += cache[13];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[13];
						ux_tmp[i][j][k] += cache[13];
						uz_tmp[i][j][k] += cache[13];

						//rhocache += cache[13];

						//printf("13 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i][j_p][k_p]){
						n_tmp[i][j_p][k_p] += cache[14];
						uy_tmp[i][j_p][k_p] += cache[14];
						uz_tmp[i][j_p][k_p] += cache[14];

						//rhocache += cache[14];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[14];
						uy_tmp[i][j][k] -= cache[14];
						uz_tmp[i][j][k] -= cache[14];

						//rhocache += cache[14];

						//printf("14 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}


					if (!wall[i][j_p][k_n]){
						n_tmp[i][j_p][k_n] += cache[15];
						uy_tmp[i][j_p][k_n] += cache[15];
						uz_tmp[i][j_p][k_n] -= cache[15];

						//rhocache += cache[15];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[15];
						uy_tmp[i][j][k] -= cache[15];
						uz_tmp[i][j][k] += cache[15];

						//rhocache += cache[15];

						//printf("15 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i][j_n][k_p]){
						n_tmp[i][j_n][k_p] += cache[16];
						uy_tmp[i][j_n][k_p] -= cache[16];
						uz_tmp[i][j_n][k_p] += cache[16];

						//rhocache += cache[16];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[16];
						uy_tmp[i][j][k] += cache[16];
						uz_tmp[i][j][k] -= cache[16];

						//rhocache += cache[16];

						//printf("16 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					if (!wall[i][j_n][k_n]){
						n_tmp[i][j_n][k_n] += cache[17];
						uy_tmp[i][j_n][k_n] -= cache[17];
						uz_tmp[i][j_n][k_n] -= cache[17];

						//rhocache += cache[17];
					}
					else if (noslip){
						n_tmp[i][j][k] += cache[17];
						uy_tmp[i][j][k] += cache[17];
						uz_tmp[i][j][k] += cache[17];

						//rhocache += cache[17];

						//printf("17 %i %i %i %.12lf\n",i][my_j][my_k, n_tmp[i][j][k]);
					}

					//Test
					for (int i1 = 0; i1 < 19; i1++){
						cache_tmp[i][j][k][i1] = cache[i1];
					}

					// *****************************************************
#if defined Debug
#if defined print_cache					
					int i2 = 0;
					if (i_p == t1 && j == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_n == t1 && j == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j_p == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j_n == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j == t2 && k_p==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j == t2 && k_n==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_p == t1 && j_p == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_p == t1 && j_n == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_n == t1 && j_p == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_n == t1 && j_n == t2 && k==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_p == t1 && j == t2 && k_p==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_p == t1 && j == t2 && k_n==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_n == t1 && j == t2 && k_p==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i_n == t1 && j == t2 && k_n==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j_p == t2 && k_p==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j_p == t2 && k_n==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j_n == t2 && k_p==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					i2++;
					if (i == t1 && j_n == t2 && k_n==t3){
						printf("CPU,%i,%i,%i,%i,%i,%i,%i,%f,%f,%f,%f,%f\n",i,j,k,i_p,j,k,i2,cache[i2],n_tmp[i][j][k],ux_tmp[i][j][k],uy_tmp[i][j][k],uz_tmp[i][j][k]);
					}
					
					//			printf ("%.12lf %.12lf %.12lf %.12lf %.12lf %.12lf\n", n_rho, cachesum + n_rho * (d0 - d1*v2), rhocache + n_rho * (d0 - d1*v2), n_rho * (d0 - d1*v2), cachesum, rhocache);  //<==

#endif
#endif
				}




				if ((k_nn >= 1) && (!wall[i][j][k_nn])){
					

					n[i][j][k_nn] = (n_tmp[i][j][k_nn]);
					ux[i][j][k_nn] = ux_tmp[i][j][k_nn];
					uy[i][j][k_nn] = uy_tmp[i][j][k_nn];
					uz[i][j][k_nn] = uz_tmp[i][j][k_nn];
					/*
					ux[i][j][k_nn] = (float)(damp0*(ux_tmp[i][j][k_nn]-fmod(ux_tmp[i][j][k_nn],floattruncate)));
					uy[i][j][k_nn] = (float)(damp0*(uy_tmp[i][j][k_nn]-fmod(uy_tmp[i][j][k_nn],floattruncate)));
					uz[i][j][k_nn] = (float)(damp0*(uz_tmp[i][j][k_nn]-fmod(uz_tmp[i][j][k_nn],floattruncate)));*/
					rho_sum += n[i][j][k_nn];
					netvx += ux[i][j][k_nn];
					netvy += uy[i][j][k_nn];
					netvz += uz[i][j][k_nn];

					//psi[i][j][k_nn] = phi0*expff(-psi_eq/n[i][j][k_nn]);
					psi[i][j][k_nn] = (float)1.0 - expff(-n[i][j][k_nn]);
					/*if (n[i][j][k_nn] < 0.2){
					psi[i][j][k_nn] = 0.1;
					}else {
					psi[i][j][k_nn] = 1;
					}*/


				}
			}
		}
	}

	k_nn = size_z - 2;
	k_n = size_z - 1;
	for (j = 0; j<size_y; j++){
		for (i = 0; i< size_x; i++){
	

			if (!wall[i][j][k_n]){

				n[i][j][k_n] = (n_tmp[i][j][k_n]);
				ux[i][j][k_n] = ux_tmp[i][j][k_n];
				uy[i][j][k_n] = uy_tmp[i][j][k_n];
				uz[i][j][k_n] = uz_tmp[i][j][k_n];
				/*
				ux[i][j][k_n] = (float)(damp0*(ux_tmp[i][j][k_n]-fmod(ux_tmp[i][j][k_n],floattruncate)));
				uy[i][j][k_n] = (float)(damp0*(uy_tmp[i][j][k_n]-fmod(uy_tmp[i][j][k_n],floattruncate)));
				uz[i][j][k_n] = (float)(damp0*(uz_tmp[i][j][k_n]-fmod(uz_tmp[i][j][k_n],floattruncate)));*/

				rho_sum += n[i][j][k_n];
				netvx += ux[i][j][k_n];
				netvy += uy[i][j][k_n];
				netvz += uz[i][j][k_n];

				//psi[i][j][k_n] = phi0*expff(-psi_eq/n[i][j][k_n]);
				psi[i][j][k_n] = (float)1.0 - expff(-n[i][j][k_n]);

				/*if (n[i][j][k_n] < 0.2){
				psi[i][j][k_n] = 0.1;
				}else {
				psi[i][j][k_n] = 1;
				}*/

			}





			if (!wall[i][j][k_nn]){

				n[i][j][k_nn] = (n_tmp[i][j][k_nn]);
				ux[i][j][k_nn] = ux_tmp[i][j][k_nn];
				uy[i][j][k_nn] = uy_tmp[i][j][k_nn];
				uz[i][j][k_nn] = uz_tmp[i][j][k_nn];
				/*
				ux[i][j][k_nn] = (float)(damp0*(ux_tmp[i][j][k_nn]-fmod(ux_tmp[i][j][k_nn],floattruncate)));
				uy[i][j][k_nn] = (float)(damp0*(uy_tmp[i][j][k_nn]-fmod(uy_tmp[i][j][k_nn],floattruncate)));
				uz[i][j][k_nn] = (float)(damp0*(uz_tmp[i][j][k_nn]-fmod(uz_tmp[i][j][k_nn],floattruncate)));*/

				rho_sum += n[i][j][k_nn];
				netvx += ux[i][j][k_nn];
				netvy += uy[i][j][k_nn];
				netvz += uz[i][j][k_nn];
				//psi[i][j][k_nn] = phi0*expff(-psi_eq/n[i][j][k_nn]);
				psi[i][j][k_nn] = (float)1.0 - expff(-n[i][j][k_nn]);
				/*if (n[i][j][k_nn] < 0.5){
				psi[i][j][k_nn] = 0.1;
				}else {
				psi[i][j][k_nn] = 1;
				}/**/

			}

			if (!wall[i][j][0]){

				n[i][j][0] = (n_tmp[i][j][0]);
				ux[i][j][0] = ux_tmp[i][j][0];
				uy[i][j][0] = uy_tmp[i][j][0];
				uz[i][j][0] = uz_tmp[i][j][0];
				/*ux[i][j][0] = (float)(damp0*(ux_tmp[i][j][0]-fmod(ux_tmp[i][j][0],floattruncate)));
				uy[i][j][0] = (float)(damp0*(uy_tmp[i][j][0]-fmod(uy_tmp[i][j][0],floattruncate)));
				uz[i][j][0] = (float)(damp0*(uz_tmp[i][j][0]-fmod(uz_tmp[i][j][0],floattruncate)));*/
				rho_sum += n[i][j][0];
				netvx += ux[i][j][0];
				netvy += uy[i][j][0];
				netvz += uz[i][j][0];

				//psi[i][j][0] = phi0*expff(-psi_eq/n[i][j][0]);
				psi[i][j][0] = (float)1.0 - expff(-n[i][j][0]);
				//printf("%.12lf %.12lf %.12lf\n", ux[i][j][15],uy[i][j][15],uz[i][j][15]);
			}
		}
	}
	netvy_ave_all += netvy;
	netvx /= rho_sum;
	netvy /= rho_sum;
	netvz /= rho_sum;
}

void position(void){
	int i, j;
	int max_cen = 0;
	int max_edg = 0;
	float cpoint_cen, cpoint_edg;
	float rca_tmp, aca_tmp;
	i = x_cen;

	for (j = 0; j < size_y; j++){
		if (n[i][j][z_cen] > max_cen) max_cen = (int)n[i][j][z_cen];
		if (n[i][j][edg] > max_edg) max_edg = (int)n[i][j][edg];
	}

	cpoint_cen = max_cen*cv;
	cpoint_edg = max_edg*cv;


	for (j = 0; j < size_y; j++){
		float n_cen1 = n[i][j][z_cen];
		float n_cen2;
		float n_edg1 = n[i][j][edg];
		float n_edg2;

		if (j < size_y - 1){
			n_cen2 = n[i][j + 1][z_cen];
			n_edg2 = n[i][j + 1][edg];
		}
		else {
			n_cen2 = n[i][0][z_cen];
			n_edg2 = n[i][0][edg];
		}


		if ((n_cen2 >= cpoint_cen) && (n_cen1 < cpoint_cen)){
			rcx_cen = cpoint_cen / (n_cen2 - n_cen1) + j;
		}


		if ((n_cen2 <= cpoint_cen) && (n_cen1 > cpoint_cen)){
			acx_cen = cpoint_cen / (n_cen2 - n_cen1) + j;
		}

		if ((n_edg2 >= cpoint_edg) && (n_edg1 < cpoint_edg)){
			rcx_edg = cpoint_edg / (n_edg2 - n_edg1) + j;
		}


		if ((n_edg2 <= cpoint_edg) && (n_edg1 > cpoint_edg)){
			acx_edg = cpoint_edg / (n_edg2 - n_edg1) + j;
		}
	}


	if (acx_cen  < acx_edg - 40){
		acx_cen += size_y;
	}

	if (rcx_edg < rcx_cen - 40){
		rcx_edg += size_y;
	}

	hac = (acx_cen - acx_edg) / width;
	hrc = (rcx_edg - rcx_cen) / width;

	hac_ave += hac;
	hrc_ave += hrc;

	aca_tmp = PI - (float)atan((1.0 - 4.0*hac*hac) / fabs(hac) / 4.0);
	rca_tmp = PI - (float)atan((1.0 - 4.0*hrc*hrc) / fabs(hrc) / 4.0);

	if (hac < 0) aca_tmp = 180 - aca_tmp;
	if (hrc < 0) rca_tmp = 180 - rca_tmp;

	acca_cache += cos(aca_tmp);
	rcca_cache += cos(rca_tmp);

}

void printprofile(void){
	short i, j, k;
	float scale;
	FILE *fout = fopen("profile.out", "wt");
	//printf("---------------------------------------------------\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n%f\n---------------------------------------------------\n|", rho_sum);
	do {

		scanf("%f", &scale);


		for (i = x_cen; i <= x_cen; i++){
			for (k = 0; k < size_z; k++){
				for (j = 0; j < size_y; j++){


					if (n[i][j][k] >= scale*rho_eq){
						fprintf(fout, "O");
						printf("O");
					}
					else if (n[i][j][k] >0) {
						fprintf(fout, "X");
						printf("X");
					}
					else if (n[i][j][k] == 0){
						fprintf(fout, " ");
						printf(" ");
					}
					else {
						fprintf(fout, "-");
						printf("-");

					}
				}
				fprintf(fout, "\n");
				printf("\n");
			}
		}
		printf("\n%f\n", scale);
		fclose(fout);
		if (scale == 999)printvalues();
	} while (scale != 0);

}

void zero_tmp(void){
	int i, j, k;
	for (k = 0; k < size_z; k++){
		for (j = 0; j < size_y; j++){
			for (i = 0; i < size_x; i++){


				if (!wall[i][j][k]){
					n_tmp[i][j][k] = 0;
					ux_tmp[i][j][k] = 0;
					uy_tmp[i][j][k] = 0;
					uz_tmp[i][j][k] = 0;
					psi_tmp[i][j][k] = 0;

				}
			}
		}
	}

}

void zero_v(void){
	int i, j, k;
	for (k = 0; k < size_z; k++){
		for (j = 0; j < size_y; j++){
			for (i = 0; i < size_x; i++){


				ux[i][j][k] = 0;
				uy[i][j][k] = 0;
				uz[i][j][k] = 0;
				u[i][j][k] = { 0, 0, 0 };

			}
		}
	}

}

void printvalues(void){
	int i, j, k;
	char filename[25];
	int temp = sprintf(filename, "profile%06i.out", runtime);
	FILE *fvalue = fopen(filename, "wt");

	fprintf(fvalue, "runtime=%i\nD=%i\nQ=%i\nb=%i\nc2=%.12lf\nSize_x=%i\nSize_y=%i\nSize_z=%i\ntime_limit=%i\nrho_eq=%.12lf\n\n\n", runtime, D, Q, b, c2, size_x, size_y, size_z, time_limit, rho_eq);
	fprintf(fvalue, "r=%.12lf\ntao=%.12lf\nd0=%.12lf\ngx=%.12lf\ngy=%.12lf\ngz=%.12lf\nT_LA=%.12lf\n psi_0=%.12lf\npsi_eq=%.12lf\nT_LS1=%.12lf\nT_LS2=%.12lf\ngask=%.12lf\nx_cen=%i\ny_cen=%i\nz_cen=%i\nfinishtime=%.5lf\n\n", r, tao, d0, gx, gy, gz, T_LA, psi_0, psi_eq, T_LS1, T_LS2, gask, x_cen, y_cen, z_cen, finishtime);

	fprintf(fvalue, "d1=%.12lf\nd2=%.12lf\nd3=%.12lf\nd4=%.12lf\nd5=%.12lf\n\n", d1, d2, d3, d4, d5);

	for (i = 0; i < size_x; i++){
		for (j = 0; j < size_y; j++){
			for (k = 0; k< size_z; k++){
				fprintf(fvalue, "%i %i %i %.12lf %.12lf %.12lf %.12lf %.12lf\n", i, j, k, n[i][j][k], ux[i][j][k], uy[i][j][k], uz[i][j][k], psi[i][j][k]);
			}
		}
	}

	for (i = 0; i < size_x; i++){
		for (j = 0; j < size_y; j++){
			for (k = 0; k< size_z; k++){
				if (wall[i][j][k]){
					fprintf(fvalue, "%i %i %i 1\n", i, j, k);
				}
				else {
					fprintf(fvalue, "%i %i %i 0\n", i, j, k);
				}
			}
		}
	}

	fclose(fvalue);
}

void printcrosssection(void){
	int i, j, k;
	char filename[25];
	int temp = sprintf(filename, "CSec%06i.out", runtime);
	FILE *fvalue = fopen(filename, "wt");

	fprintf(fvalue, "runtime=%i\nD=%i\nQ=%i\nb=%i\nc2=%.12lf\nSize_x=%i\nSize_y=%i\nSize_z=%i\ntime_limit=%i\nrho_eq=%.12lf\n\n\n", runtime, D, Q, b, c2, size_x, size_y, size_z, time_limit, rho_eq);
	fprintf(fvalue, "r=%.12lf\ntao=%.12lf\nd0=%.12lf\ngx=%.12lf\ngy=%.12lf\ngz=%.12lf\nT_LA=%.12lf\n psi_0=%.12lf\npsi_eq=%.12lf\nT_LS1=%.12lf\nT_LS2=%.12lf\ngask=%.12lf\nx_cen=%i\ny_cen=%i\nz_cen=%i\n\n", r, tao, d0, gx, gy, gz, T_LA, psi_0, psi_eq, T_LS1, T_LS2, gask, x_cen, y_cen, z_cen);

	fprintf(fvalue, "d1=%.12lf\nd2=%.12lf\nd3=%.12lf\nd4=%.12lf\nd5=%.12lf\n\n", d1, d2, d3, d4, d5);

	i = x_cen;

	for (j = 0; j < size_y; j++){
		for (k = 0; k< size_z; k++){
			fprintf(fvalue, "%i %i %i %.12lf %.12lf %.12lf %.12lf %.12lf\n", i, j, k, n[i][j][k], ux[i][j][k], uy[i][j][k], uz[i][j][k], psi[i][j][k]);
		}
	}


	fclose(fvalue);



}

void draw_img(unsigned long time_index){
	int j, k;
	char filename[25];
	float max = 0;
	float min = (float)1e37;
	int pix;
	int temp = sprintf(filename, "drop%06i.ppm", time_index);
	FILE *fvalue = fopen(filename, "wt");

	fprintf(fvalue, "P3 %i %i 255\n", size_y, size_z);



	for (j = 0; j < size_y; j++){
		for (k = 0; k< size_z; k++){
#if defined print_cuda
	  		if (n_tmp[x_cen][j][k] > max) max = n_tmp[x_cen][j][k];
			if (n_tmp[x_cen][j][k] < min) min = n_tmp[x_cen][j][k];
#else
			if (n[x_cen][j][k] > max) max = n[x_cen][j][k];
			if (n[x_cen][j][k] < min) min = n[x_cen][j][k];
#endif
		}
	}
	maxcache = max;
	mincache = min;


	for (k = 0; k< size_z; k++){
		for (j = 0; j < size_y; j++){
			if (!wall[x_cen][j][k]){
#if defined print_cuda
				pix = (int)(n_tmp[x_cen][j][k] * 255.0 / max);
#else
				pix = (int)(n[x_cen][j][k] * 255.0 / max);
#endif

				fprintf(fvalue, " %i %i %i\n", pix, pix, pix);
			}
			else {
				fprintf(fvalue, " %i %i %i\n", 200, 0, 200);
			}
		}
	}
	fclose(fvalue);



}

void loadvalues(int index1){
	int i, j, k, ii, jj, kk, tf;
	char filename[25];
	float dtemp, uxt, uyt, uzt, nt, psit;
	int itemp;
	int temp = sprintf(filename, "profile%06i.out", index1);
	FILE *fvalue = fopen(filename, "rt");

	fscanf(fvalue, "runtime=%i\nD=%i\nQ=%i\nb=%i\nc2=%f\nSize_x=%i\nSize_y=%i\nSize_z=%i\ntime_limit=%i\nrho_eq=%f\n\n\n", &itemp, &itemp, &itemp, &itemp, &dtemp, &itemp, &itemp, &itemp, &itemp, &dtemp);
	fscanf(fvalue, "r=%f\ntao=%f\nd0=%f\ngx=%f\ngy=%f\ngz=%f\nT_LA=%f\n psi_0=%f\npsi_eq=%f\nT_LS1=%f\nT_LS2=%f\ngask=%f\nx_cen=%i\ny_cen=%i\nz_cen=%i\nfinishtime=%f\n\n", &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &dtemp, &itemp, &itemp, &itemp, &dtemp);

	fscanf(fvalue, "d1=%f\nd2=%f\nd3=%f\nd4=%f\nd5=%f\n\n", &dtemp, &dtemp, &dtemp, &dtemp, &dtemp);

	for (i = 0; i < size_x; i++){
		for (j = 0; j < size_y; j++){
			for (k = 0; k< size_z; k++){
				fscanf(fvalue, "%i %i %i %f %f %f %f %f\n", &ii, &jj, &kk, &nt, &uxt, &uyt, &uzt, &psit);
				kk = (kk + k_shift) % size_z;
				n[ii][jj][kk] = nt;
				if (u_zero){
					ux[ii][jj][kk] = 0;
					uy[ii][jj][kk] = 0;
					uz[ii][jj][kk] = 0;
				}
				else{
					ux[ii][jj][kk] = uxt;
					uy[ii][jj][kk] = uyt;
					uz[ii][jj][kk] = uzt;
				}
				psi[ii][jj][kk] = psit;

			}
		}
	}

	if (readwall){
		for (i = 0; i < size_x; i++){
			for (j = 0; j < size_y; j++){
				for (k = 0; k< size_z; k++){
					fscanf(fvalue, "%i %i %i %i\n", &ii, &jj, &kk, &tf);

					if (tf == 1){
						wall[ii][jj][kk] = true;
					}
					else{
						wall[ii][jj][kk] = false;
					}
				}
			}
		}
	}
	else {
		initwall();
	}



	fclose(fvalue);

}

float expff(float val){
	float value = -val;

	if ((value <5) && (value >0)){
		int index1 = (int)(value * 20);
		return((exptable[index1 + 1] - exptable[index1]) * 20 * (value - (float)index1 / 20.0f) + exptable[index1]);
	}
	else return (exp(val));

}

void initwall(void){
	int i, j, k;
	for (i = 0; i < size_x; i++){
		for (j = 0; j < size_y; j++){
			for (k = 0; k < size_z; k++){
				//long r_tmp = ((i - x_cen)*(i - x_cen) + (j - y_cen)*(j - y_cen) + (k - z_cen)*(k - z_cen));
				//bool fluid = (r_tmp <= r*r);
				bool fluid = ((abs(i - x_cen) <= lx / 1.99) && ((float)abs(j - y_cen) <= ly / 1.99) && ((float)abs(k - z_cen) <= lz / 1.99));
				//float n_phi_eq;
				u_tmp[i][j][k] = { 0.0, 0.0, 0.0 };
				//if ( 
				if ((k == 0) || (k == (size_z - 1))){//
					wall[i][j][k] = true;

					if (j % 4 < 2){
						n[i][j][k] = 1.01010101f;
						psi[i][j][k] = T_LS1 / T_LA;
					}
					else {
						n[i][j][k] = 1.01010101f;
						psi[i][j][k] = T_LS1 / T_LA;


					}
				}
				else if (((k <= 4) || (k >= size_z - 5)) && (((j + i) % 2)<1) && (square)){//|| (k <=5)//(((k>=(size_z-4)))&& ((j%4)<2)){//
					wall[i][j][k] = true;
					n[i][j][k] = 1.01010101f;
					psi[i][j][k] = T_LS1 / T_LA;
				}
				else if (((k <= 4) || (k >= size_z - 5)) && (((j) % 2)<1) && (strip)){//|| (k <=5)//(((k>=(size_z-4)))&& ((j%4)<2)){//
					wall[i][j][k] = true;
					n[i][j][k] = 1.01010101f;
					psi[i][j][k] = T_LS1 / T_LA;
				}
				else {
					wall[i][j][k] = false;
					if (fluid){
						n[i][j][k] = rho_eq;
					}
					else {
						n[i][j][k] = gask*rho_eq;
					}
					//psi[i][j][k] = phi0*expff(-psi_eq/n[i][j][k]);
					psi[i][j][k] = 1.0f - expff(-n[i][j][k]);

				}
			}
		}
	}
}

void init_wall_cache(void){
	int i, i_p, i_n;
	int j, j_p, j_n;
	int k, k_p, k_n;

	for (i = 0; i < size_x; i++){
		for (j = 0; j < size_y; j++){
			for (k = 0; k < size_z; k++){

				i_p = (i + size_x + 1) % size_x;
				i_n = (i + size_x - 1) % size_x;
	
				j_p = (j + size_y + 1) % size_y;
				j_n = (j + size_y - 1) % size_y;

				k_p = (k + size_z + 1) % size_z;
				k_n = (k + size_z - 1) % size_z;

				if (wall[i][j][k]){
					wall_cache[i][j][k] |= 1;
				}

				if (wall[i_n][j][k]){
					wall_cache[i][j][k] |= 2;
				}

				
				if (wall[i_p][j][k]){
					wall_cache[i][j][k] |= 4;
				}

				if (wall[i][j_n][k]){
					wall_cache[i][j][k] |= 8;
				}
				
				if (wall[i][j_p][k]){
					wall_cache[i][j][k] |= 16;
				}

				if (wall[i][j][k_n]){
					wall_cache[i][j][k] |= 32;
				}

				if (wall[i][j][k_p]){
					wall_cache[i][j][k] |= 64;
				}

				if (wall[i_n][j_n][k]){
					wall_cache[i][j][k] |= 128;
				}

				if (wall[i_n][j_p][k]){
					wall_cache[i][j][k] |= 256;
				}

				if (wall[i_p][j_n][k]){
					wall_cache[i][j][k] |= 512;
				}

				if (wall[i_p][j_p][k]){
					wall_cache[i][j][k] |= 1024;
				}

				if (wall[i_n][j][k_n]){
					wall_cache[i][j][k] |= 2048;
				}

				if (wall[i_n][j][k_p]){
					wall_cache[i][j][k] |= 4096;
				}

				if (wall[i_p][j][k_n]){
					wall_cache[i][j][k] |= 8192;
				}

				if (wall[i_p][j][k_p]){
					wall_cache[i][j][k] |= 16384;
				}

				if (wall[i][j_n][k_n]){
					wall_cache[i][j][k] |= 32768;
				}

				if (wall[i][j_n][k_p]){
					wall_cache[i][j][k] |= 65536;
				}

				if (wall[i][j_p][k_n]){
					wall_cache[i][j][k] |= 131072;
				}

				if (wall[i][j_p][k_p]){
					wall_cache[i][j][k] |= 262144;
				}

			}
		}
	}


}

void cuda_run(cudaError_t my_err){
	if (my_err != cudaSuccess){
		printf("Cuda Error: %s\n", cudaGetErrorString( my_err) );
	}
}


int main(){
//	int loop2, loop3;
	int num_elements = size_x * size_y * size_z;
	cudaDeviceProp c_properties;
	size_t n_size_float   = num_elements * sizeof(float);
	size_t n_size_bool    = num_elements * sizeof(bool);
	size_t n_size_float3d = num_elements * sizeof(float3);
	size_t n_size_uint     = num_elements * sizeof(unsigned int);
	dim3 DimBlock(block_dim_x, block_dim_y, block_dim_z);
	dim3 DimBlock_2(block_dim_x + 2, block_dim_y + 2, block_dim_z + 2);
	dim3 DimGrid(size_x / (DimBlock.x), size_y / (DimBlock.y), size_z / (DimBlock.z));



	cuda_run(cudaGetDeviceProperties(&c_properties, 0));
	//testing <<< 2, 2 >> > ();

	//********************* Memory Allocation *******************
	
	float *n_cuda = NULL;
	float *psi_cuda = NULL;
	bool *wall_cuda = NULL;
	float3 *u_cuda = NULL;
	unsigned int *wall_cache_cuda = NULL;
	float *cache_tmp_cuda = NULL; 

	//cuda_run(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));     //set Shared Memory Bank Size

	cuda_run(cudaMalloc((void **) &n_cuda,    n_size_float)  );
	cuda_run(cudaMalloc((void **) &psi_cuda,  n_size_float)  );
	cuda_run(cudaMalloc((void **) &u_cuda,    n_size_float3d));
	cuda_run(cudaMalloc((void **) &wall_cuda, n_size_bool)   );
	cuda_run(cudaMalloc((void **) &cache_tmp_cuda, n_size_float*19));
	cuda_run(cudaMalloc((void **) &wall_cache_cuda, n_size_uint));						   
	
	//cuda_run(cudaMalloc((void **) &n_tmp_cuda,n_size_float)  );
	//cuda_run(cudaMalloc((void **) &u_tmp_cuda,n_size_float3d));
	//cuda_run(cudaMalloc((void **) &count_cuda, n_size_int));

	

	//******************* Initialize **************************

	//FILE *fbatch = fopen("batch.out", "wt");
	FILE *ftime = fopen("Time.out", "wt");
	int viewvalue = 0;
	int ijk;
	float rca, aca;


	//printparameters();

	initialize();

	//****************** Cuda Mem Copy to Dev ********************

	cuda_run(cudaMemcpy(n_cuda,         n,         n_size_float,      cudaMemcpyHostToDevice));
	cuda_run(cudaMemcpy(u_cuda,         u,         n_size_float3d,    cudaMemcpyHostToDevice));
	cuda_run(cudaMemcpy(psi_cuda,       psi,       n_size_float,      cudaMemcpyHostToDevice));
	cuda_run(cudaMemcpy(wall_cuda,      wall,      n_size_bool,       cudaMemcpyHostToDevice));
	cuda_run(cudaMemcpy(cache_tmp_cuda, cache_tmp, n_size_float * 19, cudaMemcpyHostToDevice));
	cuda_run(cudaMemcpy(wall_cache_cuda,wall_cache,n_size_uint,       cudaMemcpyHostToDevice));

	//***************** Main Loop *********************************

	start = clock();
	c_last = start;

	for (runtime = 0; runtime <= time_limit; runtime++){
		//gy = ceil((float)runtime/10000.0)*1e-7;

		viewvalue = (runtime) % viewstep;
		//netvx_cache[viewvalue]=netvx;
		netvy_cache[viewvalue] = netvy;
		hac_cache[viewvalue] = hac;
		hrc_cache[viewvalue] = hrc;
		//netvz_cache[viewvalue]=netvz;

		if (viewvalue == 0){
			int i, j;


			cuda_run(cudaMemcpy(n, n_cuda, n_size_float, cudaMemcpyDeviceToHost));

			//(cudaMemcpyFromSymbol(&netv, netv_cuda, sizeof(float3), 0, cudaMemcpyDeviceToHost));


			tsample[((viewvalue) % tsize)] = netvy;
			netvy_ave = 0;
			for (i = 0; i < tsize; i++){
				netvy_ave += tsample[i];
			}
			netvy_ave /= float(tsize);
			c_now = clock();

			printf("dt=%.0f %i %.8lf vy=%.8lf hac=%.5lf hrc=%.5lf\n", (float)(c_now-c_last), runtime, rho_sum, netvy, hac, hrc);
			for (j = 0; j<viewstep; j++){
				//fprintf(ftime, "dt=%f %i vx=%.12lf vy=%.12lf vz=%.12lf hac=%.12lf hrc=%.12lf\n", (float)(c_now - c_last), runtime - viewstep + j, netvx_cache[j], netvy_cache[j], netvz_cache[j], hac_cache[j], hrc_cache[j]);
			}
			c_last = c_now;
			draw_img(runtime);

		}

		//if ((runtime % (printstep)) == 0) printvalues();

#if defined CUDA
#if defined Debug
		zero_tmp();

		action();

		cuda_run(cudaMemcpy(cache_tmp_cuda, cache_tmp, n_size_float * 19, cudaMemcpyHostToDevice)); //test

		position();

#endif
   	
		cudaDeviceSynchronize();
	
		action_s2_gpu_kernel<<<DimGrid, DimBlock_2>>>(n_cuda, u_cuda, psi_cuda, wall_cuda, cache_tmp_cuda);
	
		cudaDeviceSynchronize();

		tranaction_new_i_gpu_kernel<<<DimGrid, DimBlock>>>(n_cuda, u_cuda, psi_cuda, cache_tmp_cuda, wall_cache_cuda);
	
#if defined Debug
		cudaDeviceSynchronize();
		cuda_run(cudaMemcpy(count, count_cuda, n_size_int, cudaMemcpyDeviceToHost));
		cuda_run(cudaMemcpy(n_tmp, n_cuda, n_size_float, cudaMemcpyDeviceToHost));
		cuda_run(cudaMemcpy(psi_tmp, psi_cuda, n_size_float, cudaMemcpyDeviceToHost));
		cuda_run(cudaMemcpy(u_tmp, u_cuda, n_size_float3d, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		check_data();
#endif
#endif

	}
	cudaDeviceSynchronize();
	for (ijk = 0; ijk<viewstep; ijk++){
		//fprintf(ftime, "%i vx=%.12lf vy=%.12lf vz=%.12lf hac=%.12lf hrc=%.12lf\n", runtime - viewstep + ijk, netvx_cache[ijk], netvy_cache[ijk], netvz_cache[ijk], hac_cache[ijk], hrc_cache[ijk]);
	}

	netvy_ave_all /= rho_sum*time_limit;
	hac_ave /= (time_limit);
	hrc_ave /= (time_limit);

	aca = (float) (180 - atan((1.0 - 4.0*hac_ave*hac_ave) / fabs(hac_ave) / 4.0) * 180 / PI);
	rca = (float) (180 - atan((1.0 - 4.0*hrc_ave*hrc_ave) / fabs(hrc_ave) / 4.0) * 180 / PI);

	if (hac_ave < 0) aca = 180 - aca;
	if (hrc_ave < 0) rca = 180 - rca;

	acca_cache /= time_limit;
	rcca_cache /= time_limit;

	printf("netvy_ave=%.12lf hac_ave=%.12lf, hrc_ave=%.12lf\n", netvy_ave_all, hac_ave, hrc_ave);
	printf("aca=%.12lf rca=%.12lf aca_ave=%.12lf rca_ave=%.12lf\n", aca, rca, acos(acca_cache) * 180 / PI, acos(rcca_cache) * 180 / PI);


	finish = clock();
	finishtime = (float)(finish - start) / CLOCKS_PER_SEC;
	printf("Time  =  %.8lf\n", finishtime);


	fclose(ftime);
	draw_img(runtime);
	printvalues();
	printcrosssection();
	printf("gr = %.12lf\n", gr);

	//fprintf(ftime, "Time  =  %.8lf\n", finishtime);
	//fprintf(ftime, "netvy_ave=%.12lf hac_ave=%.12lf, hrc_ave=%.12lf\n", netvy_ave_all, hac_ave, hrc_ave);
	//fprintf(ftime, "aca=%.12lf rca=%.12lf aca_ave=%.12lf rca_ave=%.12lf\n", aca, rca, acos(acca_cache) * 180 / PI, acos(rcca_cache) * 180 / PI);
	//fprintf(fbatch, "%.12lf %.12lf %.12lf %.12lf %.12lf %.12lf\n", gr, netvy_ave_all, acca_cache, rcca_cache, acos(acca_cache) * 180 / PI, acos(rcca_cache) * 180 / PI);

	cudaFree(n_cuda);
	cudaFree(psi_cuda);
	cudaFree(wall_cuda);
	cudaFree(u_cuda);
	cudaFree(cache_tmp_cuda);


	cudaDeviceReset();
	return 0;
}    

/**
if (!wall[index(i_p][my_j][my_k)]){
atomicAdd(&n_tmp[index(i_p, j, k)], cache[0]);
atomicAdd(&u_tmp[index(i_p, j, k)].x , cache[0]);
}else if (noslip) {
n_tmp_cache   += cache[0];
u_tmp_cache.x -= cache[0];
}

__syncthreads();  //2

if (!wall[index(i_n][my_j][my_k)]){
atomicAdd(&n_tmp[index(i_n, j, k)],  cache[1]);
atomicAdd(&u_tmp[index(i_n, j, k)].x , -cache[1]);
}
else if (noslip){
n_tmp_cache   += cache[1];
u_tmp_cache.x += cache[1];
}

__syncthreads();  //3

if (!wall[index(i][my_j_p][my_k)]){
atomicAdd(&n_tmp[index(i, j_p, k)] , cache[2]);
atomicAdd(&u_tmp[index(i, j_p, k)].y , cache[2]);
}
else if (noslip){
n_tmp_cache += cache[2];
u_tmp_cache.y -= cache[2];
}

__syncthreads();  //4

if (!wall[index(i][my_j_n][my_k)]){
atomicAdd(&n_tmp[index(i, j_n, k)] , cache[3]);
atomicAdd(&u_tmp[index(i, j_n, k)].y , -cache[3]);
}
else if (noslip){
n_tmp_cache   += cache[3];
u_tmp_cache.y += cache[3];
}

__syncthreads();  //5

if (!wall[index(i][my_j][my_k_p)]){
atomicAdd(&n_tmp[index(i, j, k_p)] , cache[4]);
atomicAdd(&u_tmp[index(i, j, k_p)].z , cache[4]);
}
else if (noslip){
n_tmp_cache   += cache[4];
u_tmp_cache.z -= cache[4];
}

__syncthreads();  //6

if (!wall[index(i][my_j][my_k_n)]){
atomicAdd(&n_tmp[index(i, j, k_n)] , cache[5]);
atomicAdd(&u_tmp[index(i, j, k_n)].z , -cache[5]);
}
else if (noslip){
n_tmp_cache   += cache[5];
u_tmp_cache.z += cache[5];
}

__syncthreads();  //7

if (!wall[index(i_p][my_j_p][my_k)]){
atomicAdd(&n_tmp[index(i_p, j_p, k)] , cache[6]);
atomicAdd(&u_tmp[index(i_p, j_p, k)].x , cache[6]);
atomicAdd(&u_tmp[index(i_p, j_p, k)].y , cache[6]);
}
else if (noslip){
n_tmp_cache += cache[6];
u_tmp_cache.x -= cache[6];
u_tmp_cache.y -= cache[6];
}

__syncthreads();  //8

if (!wall[index(i_p][my_j_n][my_k)]){
atomicAdd(&n_tmp[index(i_p, j_n, k)] , cache[7]);
atomicAdd(&u_tmp[index(i_p, j_n, k)].x , cache[7]);
atomicAdd(&u_tmp[index(i_p, j_n, k)].y , -cache[7]);
}
else if (noslip){
n_tmp_cache   , cache[7];
u_tmp_cache.x -= cache[7];
u_tmp_cache.y += cache[7];
}

__syncthreads();  //9

if (!wall[index(i_n][my_j_p][my_k)]){
atomicAdd(&n_tmp[index(i_n, j_p, k)] , cache[8]);
atomicAdd(&u_tmp[index(i_n, j_p, k)].x , -cache[8]);
atomicAdd(&u_tmp[index(i_n, j_p, k)].y , cache[8]);
}
else if (noslip){
n_tmp_cache   += cache[8];
u_tmp_cache.x += cache[8];
u_tmp_cache.y -= cache[8];
}

__syncthreads();  //10


if (!wall[index(i_n][my_j_n][my_k)]){
atomicAdd(&n_tmp[index(i_n, j_n, k)] , cache[9]);
atomicAdd(&u_tmp[index(i_n][my_j_n][my_k)].x , -cache[9]);
atomicAdd(&u_tmp[index(i_n][my_j_n][my_k)].y , -cache[9]);
}
else if (noslip){
n_tmp_cache   += cache[9];
u_tmp_cache.x += cache[9];
u_tmp_cache.y += cache[9];
}

__syncthreads();  //11

if (!wall[index(i_p][my_j][my_k_p)]){
atomicAdd(&n_tmp[index(i_p, j, k_p)] , cache[10]);
atomicAdd(&u_tmp[index(i_p, j, k_p)].x , cache[10]);
atomicAdd(&u_tmp[index(i_p, j, k_p)].z , cache[10]);
}
else if (noslip){
n_tmp_cache   += cache[10];
u_tmp_cache.x -= cache[10];
u_tmp_cache.z -= cache[10];
}

__syncthreads();   //12


if (!wall[index(i_p][my_j][my_k_n)]){
atomicAdd(&n_tmp[index(i_p, j, k_n)] , cache[11]);
atomicAdd(&u_tmp[index(i_p, j, k_n)].x , cache[11]);
atomicAdd(&u_tmp[index(i_p, j, k_n)].z , -cache[11]);
}
else if (noslip){
n_tmp_cache   += cache[11];
u_tmp_cache.x -= cache[11];
u_tmp_cache.z += cache[11];
}

__syncthreads();   //13

if (!wall[index(i_n][my_j][my_k_p)]){
atomicAdd(&n_tmp[index(i_n, j, k_p)] , cache[12]);
atomicAdd(&u_tmp[index(i_n, j, k_p)].x , -cache[12]);
atomicAdd(&u_tmp[index(i_n, j, k_p)].z , cache[12]);
}
else if (noslip){
n_tmp_cache   += cache[12];
u_tmp_cache.x += cache[12];
u_tmp_cache.z -= cache[12];
}

__syncthreads();   //14

if (!wall[index(i_n][my_j][my_k_n)]){
atomicAdd(&n_tmp[index(i_n, j, k_n)] , cache[13]);
atomicAdd(&u_tmp[index(i_n, j, k_n)].x , -cache[13]);
atomicAdd(&u_tmp[index(i_n, j, k_n)].z , -cache[13]);
}
else if (noslip){
n_tmp_cache   += cache[13];
u_tmp_cache.x += cache[13];
u_tmp_cache.z += cache[13];
}

__syncthreads();   //15

if (!wall[index(i][my_j_p][my_k_p)]){
atomicAdd(&n_tmp[index(i, j_p, k_p)] , cache[14]);
atomicAdd(&u_tmp[index(i, j_p, k_p)].y , cache[14]);
atomicAdd(&u_tmp[index(i, j_p, k_p)].z , cache[14]);
}
else if (noslip){
n_tmp_cache   += cache[14];
u_tmp_cache.y -= cache[14];
u_tmp_cache.z -= cache[14];
}

__syncthreads();  //16

if (!wall[index(i][my_j_p][my_k_n)]){
atomicAdd(&n_tmp[index(i, j_p, k_n)] , cache[15]);
atomicAdd(&u_tmp[index(i, j_p, k_n)].y , cache[15]);
atomicAdd(&u_tmp[index(i, j_p, k_n)].z , -cache[15]);
}
else if (noslip){
n_tmp_cache   += cache[15];
u_tmp_cache.y -= cache[15];
u_tmp_cache.z += cache[15];
}

__syncthreads();  //17

if (!wall[index(i][my_j_n][my_k_p)]){
atomicAdd(&n_tmp[index(i, j_n, k_p)] , cache[16]);
atomicAdd(&u_tmp[index(i, j_n, k_p)].y , -cache[16]);
atomicAdd(&u_tmp[index(i, j_n, k_p)].z , cache[16]);
}
else if (noslip){
n_tmp_cache   += cache[16];
u_tmp_cache.y += cache[16];
u_tmp_cache.z -= cache[16];
}

__syncthreads();  //18

if (!wall[index(i][my_j_n][my_k_n)]){
atomicAdd(&n_tmp[index(i, j_n, k_n)] , cache[17]);
atomicAdd(&u_tmp[index(i, j_n, k_n)].y , -cache[17]);
atomicAdd(&u_tmp[index(i, j_n, k_n)].z , -cache[17]);
}
else if (noslip){
n_tmp_cache   += cache[17];
u_tmp_cache.y += cache[17];
u_tmp_cache.z += cache[17];
}

__syncthreads();  //19

atomicAdd(&n_tmp[ijk], n_rho * (d0 - d1*v2) + n_tmp_cache);
atomicAdd(&u_tmp[ijk].x , u_tmp_cache.x);
atomicAdd(&u_tmp[ijk].y , u_tmp_cache.y);
atomicAdd(&u_tmp[ijk].z , u_tmp_cache.z);














if (my_i == 1){
int my_id = index(i_n, j, k);
n_tmp[my_id] += n_tmp_cache[my_i_n][my_j][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i_n][my_j][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i_n][my_j][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i_n][my_j][my_k].z;
}
else if (my_i == block_dim_x){
int my_id = index(i_p, j, k);
n_tmp[my_id] += n_tmp_cache[my_i_p][my_j][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i_p][my_j][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i_p][my_j][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i_p][my_j][my_k].z;
}


//****************************************

if (my_j == 1){
int my_id = index(i, j_n, k);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_n][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_n][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_n][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_n][my_k].z;

//********
if (my_i == 1){

int my_id = index(i_n, j_n, k);
n_tmp[my_id] += n_tmp_cache[my_i_n][my_j_n][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i_n][my_j_n][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i_n][my_j_n][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i_n][my_j_n][my_k].z;

}
else if (my_i == block_dim_x){
int my_id = index(i_p, j_n, k);
n_tmp[my_id] += n_tmp_cache[my_i_p][my_j_n][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i_p][my_j_n][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i_p][my_j_n][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i_p][my_j_n][my_k].z;
}
//*******
}
else if (my_j == block_dim_y){
int my_id = index(i, j_p, k);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_p][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_p][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_p][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_p][my_k].z;

//********
if (my_i == 1){

int my_id = index(i_n, j_p, k);
n_tmp[my_id] += n_tmp_cache[my_i_n][my_j_p][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i_n][my_j_p][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i_n][my_j_p][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i_n][my_j_p][my_k].z;

}
else if (my_i == block_dim_x){
int my_id = index(i_p, j_p, k);
n_tmp[my_id] += n_tmp_cache[my_i_p][my_j_p][my_k];
u_tmp[my_id].x += u_tmp_cache[my_i_p][my_j_p][my_k].x;
u_tmp[my_id].y += u_tmp_cache[my_i_p][my_j_p][my_k].y;
u_tmp[my_id].z += u_tmp_cache[my_i_p][my_j_p][my_k].z;
}
//*******
}

//********************

if (my_k == 1){
int my_id = index(i, j, k_n);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_n][my_k_n];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_n][my_k_n].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_n][my_k_n].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_n][my_k_n].z;

//*******************

if (my_j == 1){
int my_id = index(i, j_n, k_n);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_n][my_k_n];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_n][my_k_n].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_n][my_k_n].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_n][my_k_n].z;

//********
if (my_i == 1){

int my_id = index(i_n, j_n, k_n);
n_tmp[my_id] += n_tmp_cache[my_i_n][my_j_n][my_k_n];
u_tmp[my_id].x += u_tmp_cache[my_i_n][my_j_n][my_k_n].x;
u_tmp[my_id].y += u_tmp_cache[my_i_n][my_j_n][my_k_n].y;
u_tmp[my_id].z += u_tmp_cache[my_i_n][my_j_n][my_k_n].z;

}
else if (my_i == block_dim_x){
int my_id = index(i_p, j_n, k_n);
n_tmp[my_id] += n_tmp_cache[my_i_p][my_j_n][my_k_n];
u_tmp[my_id].x += u_tmp_cache[my_i_p][my_j_n][my_k_n].x;
u_tmp[my_id].y += u_tmp_cache[my_i_p][my_j_n][my_k_n].y;
u_tmp[my_id].z += u_tmp_cache[my_i_p][my_j_n][my_k_n].z;
}
//*******
}
else if (my_j == block_dim_y){
int my_id = index(i, j_p, k_n);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_p][my_k_n];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_p][my_k_n].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_p][my_k_n].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_p][my_k_n].z;

//********
if (my_i == 1){

int my_id = index(i_n, j_p, k_n);
n_tmp[my_id] += n_tmp_cache[my_i_n][my_j_p][my_k_n];
u_tmp[my_id].x += u_tmp_cache[my_i_n][my_j_p][my_k_n].x;
u_tmp[my_id].y += u_tmp_cache[my_i_n][my_j_p][my_k_n].y;
u_tmp[my_id].z += u_tmp_cache[my_i_n][my_j_p][my_k_n].z;

}
else if (my_i == block_dim_x){
int my_id = index(i_p, j_p, k_n);
n_tmp[my_id] += n_tmp_cache[my_i_p][my_j_p][my_k_n];
u_tmp[my_id].x += u_tmp_cache[my_i_p][my_j_p][my_k_n].x;
u_tmp[my_id].y += u_tmp_cache[my_i_p][my_j_p][my_k_n].y;
u_tmp[my_id].z += u_tmp_cache[my_i_p][my_j_p][my_k_n].z;
}
//*******
}
}
else if (my_k == block_dim_z){
int my_id = index(i, j, k_p);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_n][my_k_p];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_n][my_k_p].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_n][my_k_p].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_n][my_k_p].z;

//*******************

if (my_j == 1){
int my_id = index(i, j_n, k_p);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_n][my_k_p];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_n][my_k_p].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_n][my_k_p].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_n][my_k_p].z;

//********
if (my_i == 1){

int my_id = index(i_n, j_n, k_p);
n_tmp[my_id] += n_tmp_cache[my_i_n][my_j_n][my_k_p];
u_tmp[my_id].x += u_tmp_cache[my_i_n][my_j_n][my_k_p].x;
u_tmp[my_id].y += u_tmp_cache[my_i_n][my_j_n][my_k_p].y;
u_tmp[my_id].z += u_tmp_cache[my_i_n][my_j_n][my_k_p].z;

}
else if (my_i == block_dim_x){
int my_id = index(i_p, j_n, k_p);
n_tmp[my_id] += n_tmp_cache[my_i_p][my_j_n][my_k_p];
u_tmp[my_id].x += u_tmp_cache[my_i_p][my_j_n][my_k_p].x;
u_tmp[my_id].y += u_tmp_cache[my_i_p][my_j_n][my_k_p].y;
u_tmp[my_id].z += u_tmp_cache[my_i_p][my_j_n][my_k_p].z;
}
//*******
}
else if (my_j == block_dim_y){
int my_id = index(i, j_p, k_p);
n_tmp[my_id] += n_tmp_cache[my_i][my_j_p][my_k_p];
u_tmp[my_id].x += u_tmp_cache[my_i][my_j_p][my_k_p].x;
u_tmp[my_id].y += u_tmp_cache[my_i][my_j_p][my_k_p].y;
u_tmp[my_id].z += u_tmp_cache[my_i][my_j_p][my_k_p].z;

//********
if (my_i == 1){

int my_id = index(i_n, j_p, k_p);
n_tmp[my_id] += n_tmp_cache[my_i_n][my_j_p][my_k_p];
u_tmp[my_id].x += u_tmp_cache[my_i_n][my_j_p][my_k_p].x;
u_tmp[my_id].y += u_tmp_cache[my_i_n][my_j_p][my_k_p].y;
u_tmp[my_id].z += u_tmp_cache[my_i_n][my_j_p][my_k_p].z;

}
else if (my_i == block_dim_x){
int my_id = index(i_p, j_p, k_p);
n_tmp[my_id] += n_tmp_cache[my_i_p][my_j_p][my_k_p];
u_tmp[my_id].x += u_tmp_cache[my_i_p][my_j_p][my_k_p].x;
u_tmp[my_id].y += u_tmp_cache[my_i_p][my_j_p][my_k_p].y;
u_tmp[my_id].z += u_tmp_cache[my_i_p][my_j_p][my_k_p].z;
}
//*******
}
}
**/
