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










void assembleParticleArray(int num_rigidBody, RigidBody * rigidbodys)
{
	num_particles = 0;
	for (int i = 0; i < num_rigidBody; i++)
	{
		num_particles += rigidbodys[i].m_particle_pos.size();
	}

	cudaMalloc(&dev_particles, num_particles * sizeof(Particle));
	cudaMalloc(&dev_predictPosition, num_particles * sizeof(float));
	cudaMemset(dev_predictPosition, 0, num_particles * sizeof(float));

	cudaMalloc(&dev_positions, 3 * num_particles * sizeof(float));
	cudaMemset(dev_positions, 0, 3 * num_particles * sizeof(float));
	checkCUDAError("ERROR: cudaMalloc");

	int cur = 0;
	for (int i = 0; i < num_rigidBody; i++)
	{
		int size = rigidbodys[i].m_particle_pos.size();
		cudaMemcpy(dev_particles + cur, rigidbodys[i].m_particle_pos.data(), size * sizeof(Particle), cudaMemcpyHostToDevice);
		cur += size;
	}
	checkCUDAError("ERROR: assemble particle array");
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
		//predictPosition[threadId] = particles[threadId].x + particles[threadId].v * delta_t;

		//test TODO delete this
		particles[threadId].x = particles[threadId].x + particles[threadId].v * delta_t;
	}
}



__global__
void updatePositionFloatArray(int N, Particle * particles, float * positions)
{
	//N = num of particles
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < N)
	{
		positions[3 * threadId] = particles[threadId].x.x;
		positions[3 * threadId + 1] = particles[threadId].x.y;
		positions[3 * threadId + 2] = particles[threadId].x.z;
	}
}







void simulate(const glm::vec3 forces, const float delta_t, float * opengl_buffer)
{
	const int blockSizer = 192;
	dim3 blockCountr((num_particles + blockSizer - 1) / blockSizer);

	//apply forces
	kernApplyForces << <blockCountr, blockSizer >> >(num_particles, dev_particles, dev_predictPosition, forces, delta_t);
	checkCUDAError("ERROR: apply forces update");

	//TODO: constraints...


	//update to position float array
	updatePositionFloatArray << <blockCountr, blockSizer >> >(num_particles, dev_particles, dev_positions);
	checkCUDAError("ERROR: copy to dev_position");

	cudaMemcpy(opengl_buffer, dev_positions, 3 * num_particles * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR: copy to opengl_buffer");
}










void endSimulation()
{
	cudaFree(dev_particles);

	cudaFree(dev_predictPosition);

	cudaFree(dev_positions);
}


