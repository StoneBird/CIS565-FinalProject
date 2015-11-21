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
#include "particleSampling.h"
#include <vector>



static Particle * dev_particles;
static glm::vec3 * dev_predictPosition;

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


void applyForces(int num_particles,const glm::vec3 forces,const float delta_t)
{
	const int blockSizer = 192;
	dim3 blockCountr((num_particles + blockSizer - 1) / blockSizer);
	kernApplyForces << <blockCountr, blockSizer >> >(num_particles, dev_particles, dev_predictPosition, forces, delta_t);
	checkCUDAError("ERROR: apply forces update");
}


