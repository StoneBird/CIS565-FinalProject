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

#include "fluid.h"

#define H_KERNAL_WIDTH (0.1f)

#define NEIGHBOUR_R (2)
#define HITTEST_R (1)
#define LAMBDA_EPSILON (0.0f)

#define SPLEEFING_COFF (1000.0f)

#define RHO0 (1.0f)

static int num_rigidBodies;
static int num_particles;
static Particle * dev_particles;
static glm::vec3 * dev_predictPosition;
static glm::vec3 * dev_deltaPosition;

static int * dev_n;

static float * dev_positions;

static int loopSize_fluid = (NEIGHBOUR_R * 2 + 1)*(NEIGHBOUR_R * 2 + 1)*(NEIGHBOUR_R * 2 + 1);
static int loopSize_hit = (HITTEST_R * 2 + 1)*(HITTEST_R * 2 + 1)*(HITTEST_R * 2 + 1);
static int * dev_loopIdx_fluid;
static int * dev_loopIdx_hit;

static float grid_length;
static glm::ivec3 grid_resolution;
static glm::vec3 grid_min_x;
static glm::vec3 grid_max_x;

static Voxel * dev_grid;
static int * dev_particle_voxel_id;

static float * dev_lambda;
//lock per particle
//static int * dev_mutex;

//--------data for shape rematching--------------
//struct RigidBodyWrapper
//{
//	int base;	// first particle id
//	int size;	// size of particle
//	glm::vec3 cm_0;		//center of mass of original
//};
//__constant__ static RigidBodyWrapper* dev_rigidBodyWrappers;
glm::vec3 * hst_cm0 = NULL;
//__constant__ static glm::vec3* dev_rigid_body_cm_0;	//center mass origin
__constant__ static glm::vec3* dev_particle_x0;
//static glm::vec3* dev_particle_x0;
//-----------------------------------------------


//void initSimulate(int num_rigidBody, RigidBody * rigidbodys, glm::vec3 bmin, glm::vec3 bmax, float particle_diameter)
//{
//	assembleParticleArray(num_rigidBody, rigidbodys);
//	initUniformGrid(bmin, bmax, particle_diameter);
//}


__global__
void transformParticlePositionPerRigidBody(int base,int size, Particle * particles, glm::vec3* x0,glm::mat4 mat){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < size){
		
		threadId += base;

		x0[threadId] = particles[threadId].x;

		glm::vec4 tmp = mat * glm::vec4(particles[threadId].x, 1.0f);
		tmp /= tmp.w;
		particles[threadId].x = glm::vec3(tmp.x, tmp.y, tmp.z);

	}
}



void assembleParticleArray(int num_rigidBody, RigidBody * rigidbodys)
{
	hst_cm0 = new glm::vec3[num_rigidBody];

	num_rigidBodies = num_rigidBody;
	num_particles = 0;
	for (int i = 0; i < num_rigidBody; i++)
	{
		num_particles += rigidbodys[i].m_particles.size();
	}

	cudaMalloc(&dev_particles, num_particles * sizeof(Particle));

	cudaMalloc(&dev_predictPosition, num_particles * sizeof(glm::vec3));
	cudaMemset(dev_predictPosition, 0, num_particles * sizeof(glm::vec3));

	cudaMalloc(&dev_deltaPosition, num_particles * sizeof(glm::vec3));

	cudaMalloc(&dev_n, num_particles * sizeof(int));

	cudaMalloc(&dev_positions, 3 * num_particles * sizeof(float));
	cudaMemset(dev_positions, 0, 3 * num_particles * sizeof(float));
	checkCUDAError("ERROR: cudaMalloc");

	
	cudaMalloc(&dev_particle_x0, num_particles * sizeof(glm::vec3));
	

	//lambda
	cudaMalloc(&dev_lambda, num_particles * sizeof(float));
	cudaMemset(dev_lambda, 0, num_particles * sizeof(float));

	//lock
	//cudaMalloc(&dev_mutex, num_particles * sizeof(int));
	//cudaMemset(dev_mutex, 0, num_particles * sizeof(int));


	int cur = 0;
	//glm::vec3 * hst_cm0 = new glm::vec3[num_rigidBody];

	for (int i = 0; i < num_rigidBody; i++)
	{
		hst_cm0[i] = rigidbodys[i].getCenterOfMass();

		// Particle objects
		int size = rigidbodys[i].m_particles.size();
		cudaMemcpy(dev_particles + cur, rigidbodys[i].m_particles.data(), size * sizeof(Particle), cudaMemcpyHostToDevice);
		const int blockSizer = 192;
		dim3 blockCountr((size + blockSizer - 1) / blockSizer);
		transformParticlePositionPerRigidBody << <blockCountr, blockSizer >> >(cur, size, dev_particles, dev_particle_x0, rigidbodys[i].getTransformMatrix());



		// Initialize rest config positions
		//cudaMemcpy(dev_particle_x0 + cur, rigidbodys[i].m_x0.data(), size * sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cur += size;
	}

	//cudaMalloc(&dev_rigid_body_cm_0, num_rigidBody * sizeof(glm::vec3));
	//cudaMemcpy(dev_rigid_body_cm_0, hst_cm0, num_rigidBody * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	//delete []hst_cm0;

	checkCUDAError("ERROR: assemble particle array");
}

__global__
void initTestLoopIdx(int * fluidLoop, int * hitLoop, const glm::ivec3 resolution){
	int x, y, z, idSum2, idSum3, yTimesZ = resolution.y * resolution.z, j = 0;
	for (x = -NEIGHBOUR_R; x <= NEIGHBOUR_R; x++)
	{
		idSum2 = x * yTimesZ;
		for (y = -NEIGHBOUR_R; y <= NEIGHBOUR_R; y++)
		{
			idSum3 = idSum2 + y * resolution.z;
			for (z = -NEIGHBOUR_R; z <= NEIGHBOUR_R; z++)
			{
				fluidLoop[j] = idSum3 + z;
				j++;
			}
		}
	}

	j = 0;
	for (x = -HITTEST_R; x <= HITTEST_R; x++)
	{
		idSum2 = x * yTimesZ;
		for (y = -HITTEST_R; y <= HITTEST_R; y++)
		{
			idSum3 = idSum2 + y * resolution.z;
			for (z = -HITTEST_R; z <= HITTEST_R; z++)
			{
				hitLoop[j] = idSum3 + z;
				j++;
			}
		}
	}
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

	cudaMalloc(&dev_particle_voxel_id, num_particles *sizeof(int));

	cudaMalloc(&dev_loopIdx_fluid, loopSize_fluid *sizeof(int));
	cudaMalloc(&dev_loopIdx_hit, loopSize_hit *sizeof(int));

	initTestLoopIdx << <1, 1 >> >(dev_loopIdx_fluid, dev_loopIdx_hit, grid_resolution);
}

void endSimulation()
{
	cudaFree(dev_particles);

	cudaFree(dev_predictPosition);

	cudaFree(dev_deltaPosition);

	cudaFree(dev_n);

	cudaFree(dev_positions);


	cudaFree(dev_grid);
	cudaFree(dev_particle_voxel_id);

	cudaFree(dev_particle_x0);

	cudaFree(dev_lambda);

	cudaFree(dev_loopIdx_fluid);
	cudaFree(dev_loopIdx_hit);
	//lock
	//cudaFree(dev_mutex);


	if (hst_cm0 != NULL)
	{
		delete []hst_cm0;
		hst_cm0 = NULL;
	}
	
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
void hitTestVoxelSolid(const int num_voxel, const float diameter, const int particle_id, const int particlePhase, const float particleInvmass, const glm::vec3 particlePos,
const int voxel_id, const glm::vec3 * predict_positions, glm::vec3 * delta_positions,
	const Particle * particles, const Voxel * grid, int * dev_n)
{
	if (voxel_id < 0 || voxel_id >= num_voxel)
	{
		//voxel_id is invalid
		return;
	}
	// Delta X for collision
	glm::vec3 delta_pos(0.0), d;
	// Average count
	int n = 0;
	float delta;
	for (int i = 0; i < grid[voxel_id].num; i++)
	{
		if (particles[grid[voxel_id].particle_id[i]].phase == particlePhase)
		{
			continue;
		}
		// Distance vector from particle i to particle j (on particle centers)
		d = predict_positions[grid[voxel_id].particle_id[i]] - particlePos;
		delta = diameter - glm::length(d);
		// If particles overlap
		if (0 > delta)
		{
			continue;
		}
		// Momentum weighing based on particle mass
		// Not true momentum, but approximation
		delta = delta * (-particleInvmass / (particleInvmass + particles[grid[voxel_id].particle_id[i]].invmass));
		n++;
		// Move particle i along the vector so that i and j are in the post-collision states
		delta_pos += glm::normalize(d) * delta;
	}
	// Apply average delta X position (treat as results of constraint solver)
	delta_positions[particle_id] += delta_pos;
	dev_n[particle_id] += n;
}



// ------------------fluid---------------------
__device__
inline float getH(float diameter)
{
	//return ((float)NEIGHBOUR_R + 0.5f) * diameter;
	return 2.5f * diameter;
	//return (float)NEIGHBOUR_R * diameter;
}

__device__
inline float getRHO0(float diameter)
{
	return 0.7f * 1.0f / powf(diameter / 0.99f, 3.0f);
}

__device__
inline float SmoothKernel(float r, float h)
{
	//poly 6 kernel
	return r > h ? 0.0f : 315.0f / 64.0f / (float)PI / powf(h, 9.0f) * powf(h*h - r*r, 3.0f);


	//for test
	//float res = r > h ? 0.0f : 315.0f / 64.0f / (float)PI / powf(h, 9.0f) * powf(h*h - r*r, 3.0f);
	//printf("%f,%f\tres:%f\n", r, h, res);
	
	//return res;

	//nearest neighbour
	//return 1.0f / glm::dot(r,r);
}

__device__
inline glm::vec3 gradientSmoothKernel(glm::vec3 vec_r, float h)
{
	//r = || pi - pj||



	float r = glm::length(vec_r);
	//spiky kernel gradient

	return r>h ? glm::vec3(0.0f) : (-45.0f) / (float)PI / powf(h, 6.0f) * powf(h-r,2.0f) * glm::normalize(vec_r);
	//return glm::normalize(vec_r);


	//tmp
	//return 1.0f;
}

__device__
float fluidInfoSum(glm::vec3 & gradient, float & gradient2,
int num_voxel, float diameter, float H, int particle_id, glm::vec3 particlePos, int voxel_id, glm::vec3 * predict_positions, glm::vec3 * delta_positions,
Particle * particles, Voxel * grid, int * dev_n)
{
	if (voxel_id < 0 || voxel_id >= num_voxel)
	{
		//voxel_id is invalid
		return;
	}

	glm::vec3 g;
	float density = 0.0f;	//for particles in this voxel
	float distance;
	for (int i = 0; i < grid[voxel_id].num; i++)
	{
		if (grid[voxel_id].particle_id[i] == particle_id)
		{
			continue;
		}

		//pi - pj
		g = particlePos - predict_positions[grid[voxel_id].particle_id[i]];
		distance = glm::length(g);
		
		if (distance > H)
		{
			continue;
		}

		density += SmoothKernel(distance, H);

		g = gradientSmoothKernel(g, H);

		gradient += g;
		gradient2 += glm::dot(g,g);

	}
	//correct version should update based on gradient matrix of Kernal 
	//update all positions together

	return density;
}



__device__
void fluidNeighbourEnforce(float* lambda, //int * mutex,
int num_voxel, float diameter, float H, float oneOverRho, int particle_id, glm::vec3 particlePos, float particleLambda, int voxel_id, glm::vec3 * predict_positions, glm::vec3 * delta_positions,
Particle * particles, Voxel * grid, int * dev_n)
{
	if (voxel_id < 0 || voxel_id >= num_voxel)
	{
		//voxel_id is invalid
		return;
	}
	int gridSize = grid[voxel_id].num, gridPartId;
	glm::vec3 d;
	float delta_w;

	for (int i = 0; i < gridSize; i++)
	{
		gridPartId = grid[voxel_id].particle_id[i];
		
		if (gridPartId == particle_id)
		{
			continue;
		}
		
		// Distance vector from particle i to particle j (on particle centers)
		d = particlePos - predict_positions[gridPartId];
		
		if (glm::length(d) > H)
		{
			continue;
		}

		delta_w = oneOverRho * (particleLambda + lambda[gridPartId]);

		dev_n[particle_id] += 1;

		//gradent of W(pi-pj)
		delta_positions[particle_id] += delta_w * gradientSmoothKernel(d, H);
	}

	
}



//---------------------------------------------









//Handle Constraints
__global__
void handleCollision(int N, int num_voxel, float diameter, int * fluidLoop, int fluidLoopSize, int * hitLoop, int hitLoopSize, float* lambda,//int * mutex,
	 glm::vec3 * predictPositions, glm::vec3 * deltaPositions, Particle * particles,Voxel * grid, int * ids, float delta_t, int * dev_n)
{

	int particle_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (particle_id < N)
	{
		// Collision detection & reaction; simplified SDF constraint
		// hitTest particles in neighbour voxel

		// float particleInvmass
		float density = particles[particle_id].invmass;
		int particlePhase = particles[particle_id].phase;
		int i, idSum = ids[particle_id];
		glm::vec3 particlePos = predictPositions[particle_id];

		for (i = 0; i < hitLoopSize; i++)
		{
			hitTestVoxelSolid(num_voxel, diameter, particle_id, particlePhase, density, particlePos,
				idSum + hitLoop[i],
				predictPositions, deltaPositions, particles, grid, dev_n);
		}
		
		if (particles[particle_id].type == FLUID)
		{
			density = 0.0f;
			float H = getH(diameter);
			glm::vec3 sum_gradient(0.0f);
			float sum_gradient2 = 0.0f;

			//first loop used to get the sum of density rho_i, sum of gradient
			//fluid density constraint
			
			for (i = 0; i < fluidLoopSize; i++)
			{
				density += fluidInfoSum(sum_gradient, sum_gradient2,
					num_voxel, diameter, H, particle_id, particlePos,
					idSum + fluidLoop[i],
					predictPositions, deltaPositions, particles, grid, dev_n);
			}

			// when density / rho_0 -1.0f > 0 , move
			// i.e. when lambda < 0, move

			// float ci
			H = density / getRHO0(diameter) - 1.0f;
			// float denominator
			density = sum_gradient2 + glm::dot(sum_gradient, sum_gradient) + LAMBDA_EPSILON;
			// float lambda_i
			H = -20.0f * H / density;

			lambda[particle_id] = min(0.0f, H);

		}
	}
}

//fluid apply delta
__global__
void FluidApplyLambdaDelta(int N, int num_voxel, float diameter, int * fluidLoop, int loopSize, float* lambda,//int * mutex,
glm::vec3 * predictPositions, glm::vec3 * deltaPositions, Particle * particles, Voxel * grid, int * ids, float delta_t, int * dev_n)
{
	int particle_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (particle_id < N)
	{
		// Collision detection & reaction; simplified SDF constraint
		// hitTest particles in neighbour voxel

		if (particles[particle_id].type == FLUID)
		{
			int idSum = ids[particle_id];
			glm::vec3 particlePos = predictPositions[particle_id];
			float particleLambda = lambda[particle_id];

			float H = getH(diameter);
			float oneOverRho = 1.0f / getRHO0(diameter);

			for (int i = 0; i < loopSize; i++)
			{
				fluidNeighbourEnforce(lambda,//mutex,
					num_voxel, diameter, H, oneOverRho, particle_id, particlePos, particleLambda,
					idSum + fluidLoop[i],
					predictPositions, deltaPositions, particles, grid, dev_n);
			}
		}
	}
}

__global__
void setAValue(int base, int N, glm::mat3 * Apq, glm::vec3 * x0 , glm::vec3 * predict_x, glm::vec3 cm, glm::vec3 cm0)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < N)
	{
		//treat every particle in one rigid body has the same mass
		Apq[tid] = glm::outerProduct(predict_x[tid + base] - cm
			, x0[tid + base] - cm0);
	}
	
}


__global__
void shapeMatching(int base, int size, glm::vec3 * delta_positions, glm::vec3 * predictions, glm::vec3 *x0, glm::vec3 cm0, glm::vec3 cm, glm::mat3 Apq, int * dev_n){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < size){
		threadId += base;

		glm::mat3 R(0.0f), ROOT(0.0f);

		glm::mat3 A = glm::transpose(Apq)*Apq;

		// Denman¨CBeavers iteration
		// https://en.wikipedia.org/wiki/Square_root_of_a_matrix

		glm::mat3 Y = A, Z(1.0f);

		//older 8
		for (int i = 0; i < 16; i++){
			Y = 0.5f*(Y + glm::inverse(Z));
			Z = 0.5f*(Z + glm::inverse(Y));
		}

		ROOT = Y;

		// https://en.wikipedia.org/wiki/Polar_decomposition
		R = Apq * glm::inverse(ROOT);

		// Delta X for shape matching
		delta_positions[threadId] += R * (x0[threadId] - cm0) + cm - predictions[threadId];
		dev_n[threadId]++;
	}
}

__global__
void applyDelta(glm::vec3 * predictions, const glm::vec3 * delta, const int * n, const int num_particles){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < num_particles){
		// Average and update
		float nd = n[threadId] == 0 ? 1.0 : (float)n[threadId];
		predictions[threadId] += delta[threadId] / nd;
	}
}

__global__
void applyDeltaForCM(glm::vec3 * predictions, const glm::vec3 * delta, const int * n, const int num_particles, int base){
	//for one rigid body
	//predictions here are temporary, calculating for cm only
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < num_particles){
		// Average and update
		int offset = threadId + base;
		float nd = n[offset] == 0 ? 1.0 : (float)n[offset];
		predictions[threadId] += delta[offset] / nd;
	}
}

__global__
void updatePositionFloatArray(int N, glm::vec3 * predictions, Particle * particles, float * positions, const float delta_t)
{
	//N = num of particles
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadId < N)
	{
		glm::vec3 delta_d = predictions[threadId] - particles[threadId].x;
		// Update velocity
		particles[threadId].v = delta_d / delta_t;

		// Particle sleeping
		// Truncate super small values so avoid if-statement
		particles[threadId].x = particles[threadId].x + glm::trunc(delta_d*SPLEEFING_COFF) / SPLEEFING_COFF;
		//if (particles[threadId].type == SOLID)
		//{
		//	particles[threadId].x = particles[threadId].x + glm::trunc(delta_d*SPLEEFING_COFF) / SPLEEFING_COFF;
		//}
		//else
		//{
		//	particles[threadId].x = particles[threadId].x + delta_d;
		//}

		// Update positions
		positions[3 * threadId] = particles[threadId].x.x;
		positions[3 * threadId + 1] = particles[threadId].x.y;
		positions[3 * threadId + 2] = particles[threadId].x.z;
	}
}

void simulate(const glm::vec3 forces, const float delta_t, float * opengl_buffer, RigidBody * rigidBody)
{
	cudaMemset(dev_deltaPosition, 0, num_particles * sizeof(glm::vec3));
	cudaMemset(dev_n, 0, num_particles * sizeof(int));

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
	cudaMemset(dev_particle_voxel_id, 0, num_particles * sizeof(int));

	//update
	updateVoxelIndex << <blockCountr, blockSizer >> >(num_particles, grid_resolution, grid_min_x, grid_length, dev_predictPosition, dev_grid, dev_particle_voxel_id);
	//updateVoxelIndexBefore << <blockCountr, blockSizer >> >(num_particles, grid_resolution, grid_min_x, grid_length, dev_positions, dev_grid, dev_particle_voxel_id);
	checkCUDAError("ERROR: updateVoxelIndex");

	const int blockSizer2 = 128;

	//detect collisions and generate collision constraints
	//for fluid, get density and constraints
	handleCollision << <blockCountr, blockSizer >> >(num_particles, num_voxel, grid_length * 0.99f, dev_loopIdx_fluid, loopSize_fluid, dev_loopIdx_hit, loopSize_hit, dev_lambda,//dev_mutex,
		dev_predictPosition, dev_deltaPosition, dev_particles, dev_grid, dev_particle_voxel_id, delta_t, dev_n);
	checkCUDAError("ERROR: handle collision");

	//cudaDeviceSynchronize();
	
	FluidApplyLambdaDelta << <blockCountr, blockSizer >> >(num_particles, num_voxel, grid_length * 0.99f, dev_loopIdx_fluid, loopSize_fluid, dev_lambda,//dev_mutex,
		dev_predictPosition, dev_deltaPosition, dev_particles, dev_grid, dev_particle_voxel_id, delta_t, dev_n);
	checkCUDAError("ERROR: handle collision");
	
	//cudaDeviceSynchronize();

	//---- Shape matching constraint --------
	int base = 0;
	for (int i = 0; i < num_rigidBodies; i++)
	{
		ParticleType body_type = rigidBody[i].getType();
		int size = rigidBody[i].m_particles.size();
		dim3 blockCountrPerRigidBody((size + blockSizer - 1) / blockSizer);
		dim3 blockCountrPerRigidBody2((size + blockSizer2 - 1) / blockSizer2);

		//generate constraints
		if (body_type == SOLID)
		{
			//Rigid solid body part
			//TODO: turn into standard constraint method
			if (rigidBody[i].getInvMassScale() < FLT_EPSILON)
			{
				//static object, no need for shape matching
				base += size;
				continue;
			}

			thrust::device_vector<glm::mat3> dev_Apq(num_particles);
			glm::mat3 * dev_Apq_ptr = thrust::raw_pointer_cast(&dev_Apq[0]);

			//calculate current cm
			thrust::device_vector<glm::vec3> dev_px(size);	//predict position
			//glm::vec3 * dev_px_ptr = thrust::raw_pointer_cast(&dev_px[0]);
			//cudaMemcpy(dev_px_ptr, dev_predictPosition + base, size * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
			thrust::device_ptr<glm::vec3> dev_predict_base(dev_predictPosition);
			thrust::copy_n(dev_predict_base + base, size, dev_px.begin());

			//update collision delta for calculating cm
			glm::vec3 * dev_px_ptr = thrust::raw_pointer_cast(&dev_px[0]);
			applyDeltaForCM << <blockCountrPerRigidBody, blockSizer >> >(dev_px_ptr, dev_deltaPosition, dev_n, size, base);


			glm::vec3 cm = thrust::reduce(dev_px.begin(), dev_px.end(), glm::vec3(0.0), thrust::plus<glm::vec3>());
			cm = cm / ((float)size);

			//calculate A matrix
			// Pre-process; calculate individual outer products
			setAValue << <blockCountrPerRigidBody, blockSizer >> >(base, size, dev_Apq_ptr, dev_particle_x0, dev_predictPosition,
				cm, hst_cm0[i]);

			glm::mat3 Apq = thrust::reduce(dev_Apq.begin(), dev_Apq.end(), glm::mat3(0.0), thrust::plus<glm::mat3>());

			//modify predict positions
			// Also find A and R within the kernel
			shapeMatching << <blockCountrPerRigidBody2, blockSizer2 >> >(base, size, dev_deltaPosition, dev_predictPosition, dev_particle_x0, hst_cm0[i], cm, Apq, dev_n);

		}
		

		

		//next body
		base += size;
	}

	applyDelta << <blockCountr, blockSizer >> >(dev_predictPosition, dev_deltaPosition, dev_n, num_particles);

	//update to position float array
	updatePositionFloatArray << <blockCountr, blockSizer >> >(num_particles, dev_predictPosition, dev_particles, dev_positions, delta_t);
	checkCUDAError("ERROR: copy to dev_position");

	cudaMemcpy(opengl_buffer, dev_positions, 3 * num_particles * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR: copy to opengl_buffer");
}