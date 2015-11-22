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

#define K_SPRING_COEFF (0.1f)
#define N_DAMPING_COEFF (0.0f)
#define K_SHEAR_COEFF (0.0f)


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

__global__
void transformParticlePositionPerRigidBody(int base,int size,Particle * particles,glm::mat4 mat){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < size){
		
		threadId += base;

		glm::vec4 tmp = mat * glm::vec4(particles[threadId].x, 1.0f);
		tmp /= tmp.w;
		particles[threadId].x = glm::vec3(tmp.x, tmp.y, tmp.z);

	}
}


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


	const int blockSizer = 192;
	int cur = 0;
	for (int i = 0; i < num_rigidBody; i++)
	{
		// Particle objects
		int size = rigidbodys[i].m_particles.size();
		cudaMemcpy(dev_particles + cur, rigidbodys[i].m_particles.data(), size * sizeof(Particle), cudaMemcpyHostToDevice);
		
		// translations and rotations of the rigid body should be done here
		dim3 blockCountr((size + blockSizer - 1) / blockSizer);
		transformParticlePositionPerRigidBody << <blockCountr, blockSizer >> >(cur, size, dev_particles, rigidbodys[i].getTransformMatrix());


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
			//printf("out of simulation region\n");	//test
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
		//apply forces
		particles[threadId].v += particles[threadId].invmass * forces * delta_t;

		//predict positions
		predictPosition[threadId] = particles[threadId].x + particles[threadId].v * delta_t;
	}
}



__device__
void hitTestVoxel(int num_voxel, float diameter, int particle_id, int voxel_id ,glm::vec3 * predict_positions, Particle * particles, Voxel * grid)
{
	if (voxel_id < 0 || voxel_id >= num_voxel)
	{
		//voxel_id is invalid
		return;
	}
	// Delta X for collision
	glm::vec3 delta_pos(0.0);
	// Average count
	int n = 0;
	for (int i = 0; i < grid[voxel_id].num; i++)
	{
		if (particles[grid[voxel_id].particle_id[i]].phase != particles[particle_id].phase)
		{
			// Distance vector from particle i to particle j (on particle centers)
			glm::vec3 d = predict_positions[grid[voxel_id].particle_id[i]] - predict_positions[particle_id];
			// If particles overlap
			if (glm::length(d) <= diameter)
			{
				n++;
				// Momentum weighing based on particle mass
				// Not true momentum, but approximation
				float momentWeight = -particles[particle_id].invmass / (particles[particle_id].invmass + particles[grid[voxel_id].particle_id[i]].invmass);
				// Move particle i along the vector so that i and j are in the post-collision states
				delta_pos += momentWeight * glm::normalize(d) * (diameter - glm::length(d));
			}
		}
	}
	// Apply average delta X position (treat as results of constraint solver)
	if (n > 0){
		predict_positions[particle_id] += delta_pos / (float)n;
	}
}


__global__
void handleCollision(int N, int num_voxel, float diameter, glm::ivec3 resolution
	, glm::vec3 * predictPositions, Particle * particles,Voxel * grid, int * ids, float delta_t)
{

	int particle_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (particle_id < N)
	{
		int voxel_id = ids[particle_id];

		// Collision detection & reaction; simplified SDF constraint
		// hitTest particles in neighbour voxel
		for (int x = -1; x <= 1; x++)
		{
			for (int y = -1; y <= 1; y++)
			{
				for (int z = -1; z <= 1; z++)
				{
					hitTestVoxel(num_voxel, diameter, particle_id,
						voxel_id + z * 1 + y * resolution.z + x * resolution.y * resolution.z,
						predictPositions, particles, grid);
				}
			}
		}
	}
}

__global__
void updatePositionFloatArray(int N, glm::vec3 * predictions, Particle * particles, float * positions, const float delta_t)
{
	//N = num of particles
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < N)
	{
		// Update velocity
		particles[threadId].v = (predictions[threadId] - particles[threadId].x) / delta_t;

		// Particle sleeping
		// Truncate super small values so avoid if-statement
		particles[threadId].x = particles[threadId].x + glm::trunc((predictions[threadId] - particles[threadId].x)*100000.0f) / 100000.0f;

		// Update positions
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
	int num_voxel = grid_resolution.x * grid_resolution.y * grid_resolution.z;
	cudaMemset(dev_grid, 0, num_voxel * sizeof(Voxel));
	// FIXME below memset breaks updateVoxelIndex; not sure why
	//cudaMemset(dev_particle_voxel_id, 0, num_voxel * sizeof(Voxel));

	//update
	updateVoxelIndex << <blockCountr, blockSizer >> >(num_particles, grid_resolution, grid_min_x, grid_length, dev_predictPosition, dev_grid, dev_particle_voxel_id);
	checkCUDAError("ERROR: updateVoxelIndex");

	//detect collisions and generate collision constraints
	handleCollision << <blockCountr, blockSizer >> >(num_particles, num_voxel, grid_length,
		grid_resolution, dev_predictPosition, dev_particles, dev_grid, dev_particle_voxel_id, delta_t);
	checkCUDAError("ERROR: handle collision");

	// Shape matching constraint
	// Delta X for shape matching
	// Average and update

	//update to position float array
	updatePositionFloatArray << <blockCountr, blockSizer >> >(num_particles, dev_predictPosition, dev_particles, dev_positions, delta_t);
	checkCUDAError("ERROR: copy to dev_position");

	cudaMemcpy(opengl_buffer, dev_positions, 3 * num_particles * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR: copy to opengl_buffer");
}