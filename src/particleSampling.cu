#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <util/checkCUDAError.h>
#include <vector>
#include <glm/gtx/transform.hpp>
#include "particleSampling.h"

struct Triangle
{
	glm::vec3 v[3];
};


static glm::ivec3 resolution = glm::ivec3(10);


static int num_vertices;
static int num_triangles;

static float * dev_positions;
static unsigned int* dev_indices;

static float * dev_first_depths;
static float * dev_second_depths;

static Triangle * dev_triangles;







void samplingInit(int num_v)
{
	//num_v is given by indices.size()

	//x,y width, temp
	int x_width = resolution.x;
	int y_width = resolution.y;

	num_vertices = num_v;
	num_triangles = num_v / 3;

	cudaFree(dev_positions);
	cudaMalloc(&dev_positions, num_vertices * 3 * sizeof(float));
	
	cudaFree(dev_indices);
	cudaMalloc(&dev_indices, num_vertices * sizeof(unsigned int));
	
	cudaFree(dev_triangles);
	cudaMalloc(&dev_indices, num_triangles * sizeof(Triangle));

	int depth_tex_size = x_width * y_width * sizeof(float);

	cudaFree(dev_first_depths);
	cudaMalloc(&dev_first_depths, depth_tex_size);
	cudaMemset(dev_first_depths, 0, depth_tex_size);

	cudaFree(dev_second_depths);
	cudaMalloc(&dev_second_depths, depth_tex_size);
	cudaMemset(dev_second_depths, 0, depth_tex_size);


	checkCUDAError("sampling Init");


}


__global__
void kernTriangleAssembly(int N,Triangle * triangles, float * position, unsigned int * indices)
{
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	

	if (threadId < N)
	{
		int v_id = indices[threadId];
		int tri_id = threadId / 3;
		int i = threadId - 3 * tri_id;
		triangles[tri_id].v[i] = glm::vec3(position[3 * v_id], position[3 * v_id + 1], position[3 * v_id + 2]);
	}
}





void samplingSetBuffers(float * hst_positions, unsigned int * hst_indices)
{
	cudaMemcpy(dev_positions, hst_positions, num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_indices, hst_indices, num_vertices * sizeof(unsigned int), cudaMemcpyHostToDevice);


	
	//triangle assembly
	const int blockSize = 192;
	dim3 blockCount((num_vertices + blockSize - 1) / blockSize);

	kernTriangleAssembly << <blockCount, blockSize >> >(num_vertices,dev_triangles, dev_positions, dev_indices);
}



void samplingFree()
{
	cudaFree(dev_positions);
	dev_positions = NULL;

	cudaFree(dev_indices);
	dev_indices = NULL;

	cudaFree(dev_triangles);
	dev_triangles = NULL;

	cudaFree(dev_first_depths);
	dev_first_depths = NULL;

	cudaFree(dev_second_depths);
	dev_second_depths = NULL;


}