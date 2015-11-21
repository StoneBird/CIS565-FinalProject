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
#include <glm/gtx/transform.hpp>
#include "simulate.h"
#include <vector>


static int num_particles;
static Particle * dev_particles;
static glm::vec3 * dev_predictPosition;

static float * dev_positions;


static float grid_length;
static glm::ivec3 grid_resolution;
static glm::vec3 grid_min_x;
static glm::vec3 grid_max_x;

static Voxel * dev_grid;
static int * dev_particle_voxel_id;



//void initSimulate(int num_rigidBody, RigidBody * rigidbodys, glm::vec3 bmin, glm::vec3 bmax, float particle_diameter)
//{
//	assembleParticleArray(num_rigidBody, rigidbodys);
//	initUniformGrid(bmin, bmax, particle_diameter);
//}

void assembleParticleArray(int num_rigidBody, RigidBody * rigidbodys)
{
	num_particles = 0;
	for (int i = 0; i < num_rigidBody; i++)
	{
		num_particles += rigidbodys[i].m_particles.size();
	}

	cudaMalloc(&dev_particles, num_particles * sizeof(Particle));

	cudaMalloc(&dev_predictPosition, num_particles * sizeof(glm::vec3));
	cudaMemset(dev_predictPosition, 0, num_particles * sizeof(glm::vec3));

	cudaMalloc(&dev_positions, 3 * num_particles * sizeof(float));
	cudaMemset(dev_positions, 0, 3 * num_particles * sizeof(float));
	checkCUDAError("ERROR: cudaMalloc");

	int cur = 0;
	for (int i = 0; i < num_rigidBody; i++)
	{
		// Particle objects
		int size = rigidbodys[i].m_particles.size();
		cudaMemcpy(dev_particles + cur, rigidbodys[i].m_particles.data(), size * sizeof(Particle), cudaMemcpyHostToDevice);
		cur += size;

		// TODO copy position values too so that particle sleeping works
	}
	checkCUDAError("ERROR: assemble particle array");
}

void initUniformGrid(glm::vec3 bmin, glm::vec3 bmax, float diameter)
{
	//init size
	grid_min_x = bmin;
	grid_max_x = bmax;

	grid_length = diameter;

	grid_resolution = ceil((grid_max_x - grid_min_x) / grid_length);


	int grid_size = grid_resolution.x * grid_resolution.y * grid_resolution.z;
	cudaMalloc(&dev_grid, grid_size * sizeof(Voxel));
	cudaMemset(dev_grid, 0, grid_size * sizeof(Voxel));

	cudaMalloc(&dev_particle_voxel_id, num_particles);
}

void endSimulation()
{
	cudaFree(dev_particles);

	cudaFree(dev_predictPosition);

	cudaFree(dev_positions);


	cudaFree(dev_grid);
	cudaFree(dev_particle_voxel_id);
}


__device__
glm::ivec3 gridMap(glm::vec3 x, glm::vec3 min_x, float grid_length)
{
	return (glm::ivec3)(floor((x - min_x) / grid_length));
}

__global__
void updateVoxelIndex(int N , glm::ivec3 grid_resolution, glm::vec3 min_x, float grid_length, glm::vec3 * particlePositions, Voxel * grid, int * ids )
{
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < N)
	{
		glm::ivec3 coordinate = gridMap(particlePositions[threadId], min_x, grid_length);

		//outof simulate area
		if (coordinate.x >= grid_resolution.x || coordinate.x < 0
			|| coordinate.y >= grid_resolution.y || coordinate.y < 0
			|| coordinate.z >= grid_resolution.z || coordinate.z < 0)
		{
			//don't assign to vertex
			printf("out of simulation region\n");	//test
			return;
		}



		int voxel_id = coordinate.x * grid_resolution.y * grid_resolution.z
			+ coordinate.y * grid_resolution.z
			+ coordinate.z;

		//not taken into account when n  > NUM_PRIACTICLE_VOXEL ? 
		grid[voxel_id].particle_id[grid[voxel_id].num] = threadId;
		grid[voxel_id].num += 1;

		ids[threadId] = voxel_id;
	}
}



__global__
void kernApplyForces(int N, Particle * particles, glm::vec3 * predictPosition, const glm::vec3 forces, const float delta_t)
{
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < N)
	{
		//TODO:delete this, for test only
		particles[threadId].invmass = 1.0f;

		//apply forces
		particles[threadId].v += particles[threadId].invmass * forces * delta_t;

		//predict positions
		predictPosition[threadId] = particles[threadId].x + particles[threadId].v * delta_t;
	}
}

__global__
void solveRigidBody(){
	// Collision detection & reaction (get particle force)
	// Compute momenta; linear and angular
	// Delta X for collision
	// Shape matching constraint
	// Delta X for shape matching
	// Average and update
}

__global__
void updatePositionFloatArray(int N, glm::vec3 * predictions, Particle * particles, float * positions)
{
	//N = num of particles
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < N)
	{
		// Particle sleeping
		// Truncate super small values so avoid if-statement
		particles[threadId].x = particles[threadId].x + glm::trunc((predictions[threadId] - particles[threadId].x)*100000.0f) / 100000.0f;

		positions[3 * threadId] = particles[threadId].x.x;
		positions[3 * threadId + 1] = particles[threadId].x.y;
		positions[3 * threadId + 2] = particles[threadId].x.z;
	}
}

void simulate(const glm::vec3 forces, const float delta_t, float * opengl_buffer)
{
	const int blockSizer = 192;
	dim3 blockCountr((num_particles + blockSizer - 1) / blockSizer);
	checkCUDAError("ERROR: LOL");

	//apply forces
	kernApplyForces << <blockCountr, blockSizer >> >(num_particles, dev_particles, dev_predictPosition, forces, delta_t);
	checkCUDAError("ERROR: apply forces update");

	//update voxel index
	//clean
	cudaMemset(dev_grid, 0, grid_resolution.x * grid_resolution.y * grid_resolution.z * sizeof(Voxel));
	cudaMemset(dev_grid, 0, grid_resolution.x * grid_resolution.y * grid_resolution.z * sizeof(Voxel));
	//update
	updateVoxelIndex << <blockCountr, blockSizer >> >(num_particles, grid_resolution, grid_min_x, grid_length, dev_predictPosition, dev_grid, dev_particle_voxel_id);
	checkCUDAError("ERROR: updateVoxelIndex");

	// Rigid body (partilce centric; one single kernel)
	solveRigidBody << <blockCountr, blockSizer>> >();

	//update to position float array
	updatePositionFloatArray << <blockCountr, blockSizer >> >(num_particles, dev_predictPosition, dev_particles, dev_positions);
	checkCUDAError("ERROR: copy to dev_position");

	cudaMemcpy(opengl_buffer, dev_positions, 3 * num_particles * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR: copy to opengl_buffer");
}