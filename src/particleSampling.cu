#include <cmath>
#include <cstdio>
#include <cuda.h>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include <util/checkCUDAError.h>
#include <vector>
#include <glm/gtx/transform.hpp>
#include "particleSampling.h"

struct is_empty{
	__host__ __device__
		bool operator()(const ParticleWrapper &p)
	{
		return p.isEmpty;
	}
};

struct Triangle
{
	glm::vec3 v[3];
};

struct RayPeel
{
	glm::vec2 peel;
};


static glm::ivec3 resolution = glm::ivec3(10);
static int num_particles;
static int num_rays;
static float voxel_diam;

// Init ray grid; ray grid is XY, ortho along Z;
static glm::vec3 ray = glm::vec3(0.0f, 0.0f, 1.0f);

static int num_vertices;
static int num_triangles;

static float * dev_positions;
static unsigned int* dev_indices;
static Triangle * dev_triangles;
static ParticleWrapper * dev_particles;
static Particle * dev_particle_cache;
static RayPeel * dev_peels;

void samplingInit(int num_v, glm::vec3 m_resolution, float m_grid_length)
{
	//num_v is given by indices.size()

	resolution = m_resolution;
	num_rays = resolution.x * resolution.y;
	num_particles = num_rays * resolution.z;
	voxel_diam = m_grid_length;

	num_vertices = num_v;
	num_triangles = num_v / 3;

	cudaFree(dev_positions);
	cudaMalloc(&dev_positions, num_vertices * 3 * sizeof(float));
	
	cudaFree(dev_indices);
	cudaMalloc(&dev_indices, num_vertices * sizeof(unsigned int));
	
	cudaFree(dev_triangles);
	cudaMalloc(&dev_indices, num_triangles * sizeof(Triangle));

	cudaFree(dev_peels);
	cudaMalloc(&dev_peels, num_rays * sizeof(RayPeel));

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

	cudaFree(dev_peels);
	dev_peels = NULL;
}

/*************************************************************************************
* Sampling routine
**************************************************************************************/

__device__
void coordRemap(int &x, int &y, const glm::vec3 resolution){
	x = x - resolution.x / 2;
	y = resolution.y / 2 - y;
}

__global__
void intersect(RayPeel * rp, Triangle * tri, const glm::vec3 resolution, const float diameter, const glm::vec3 ray){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = x + y*resolution.x;
	if (x < resolution.x && y < resolution.y){
		coordRemap(x, y, resolution);
		rp[idx].peel = glm::vec2(-resolution.z/2, resolution.z/2)*diameter;
	}
}

__global__
void fillPeel(ParticleWrapper * p_out, RayPeel * rp, const glm::vec3 resolution, const float diameter){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.x * blockIdx.x + threadIdx.x;
	int idx = x + y*resolution.x;
	if (x < resolution.x && y < resolution.y){
		glm::vec2 p = rp[idx].peel;
		int depth = abs(p.y - p.x);
		coordRemap(x, y, resolution);
		for (int z = 0; z < depth; z++){
			p_out[idx + z].x = glm::vec3(x, y, z)*diameter;
			p_out[idx + z].isEmpty = false;
		}
	}
}

__global__
void translateParticle(Particle *p_out, ParticleWrapper *p_in, int size){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < size){
		p_out[threadId].x = p_in[threadId].x;
	}
}

int sampleParticles(Particle * p_out){
	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength, blockSideLength);
	dim3 blocksPerGrid(
		(resolution.x + blockSize.x - 1) / blockSize.x,
		(resolution.y + blockSize.y - 1) / blockSize.y);

	thrust::device_vector<ParticleWrapper> dev_grid(num_particles);
	dev_particles = thrust::raw_pointer_cast(&dev_grid[0]);

	cudaMalloc(&dev_particle_cache, num_particles * sizeof(Particle));
	
	// Depth peeling
		// Intersection test
	intersect << <blocksPerGrid, blockSize >> >(dev_peels, dev_triangles, resolution, voxel_diam, ray);
	checkCUDAError("Intersection");
		// Fill ray segment
	fillPeel << <blocksPerGrid, blockSize >> >(dev_particles, dev_peels, resolution, voxel_diam);
	checkCUDAError("Peel filling");
	// Stream compaction
	thrust::detail::normal_iterator<thrust::device_ptr<ParticleWrapper>> newGridEnd = thrust::remove_if(dev_grid.begin(), dev_grid.end(), is_empty());
	checkCUDAError("Compaction");
	dev_grid.erase(newGridEnd, dev_grid.end());
	int newSize = dev_grid.size();
	// Write to array of Particle
	translateParticle << <newSize, 1>> >(dev_particle_cache, dev_particles, newSize);
	checkCUDAError("Wrapper translation");
	// Copy to host
	cudaMemcpy(p_out, dev_particle_cache, newSize * sizeof(Particle), cudaMemcpyDeviceToHost);
	checkCUDAError("Memcpy");
	cudaFree(dev_particle_cache);

	// Return final array length
	return newSize;
}