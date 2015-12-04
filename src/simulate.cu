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
//#include <util/dsyevj3.c>

#define H_KERNAL_WIDTH (0.1f)

#define NEIGHBOUR_R (1)
#define LAMBDA_EPSILON (1.0f)

#define SPLEEFING_COFF (1000.0f)

#define RHO0 (1.0f)

static int num_rigidBodies;
static int num_particles;
static Particle * dev_particles;
static glm::vec3 * dev_predictPosition;
static glm::vec3 * dev_deltaPosition;

static int * dev_n;

static float * dev_positions;


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

	cudaFree(dev_deltaPosition);

	cudaFree(dev_n);

	cudaFree(dev_positions);


	cudaFree(dev_grid);
	cudaFree(dev_particle_voxel_id);

	cudaFree(dev_particle_x0);

	cudaFree(dev_lambda);
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
void hitTestVoxelSolid(int num_voxel, float diameter, int particle_id, int voxel_id, glm::vec3 * predict_positions, glm::vec3 * delta_positions,
	Particle * particles, Voxel * grid, int * dev_n)
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
			//&& particles[grid[voxel_id].particle_id[i]].type == SOLID)
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
	delta_positions[particle_id] += delta_pos;
	dev_n[particle_id] += n;
}




// ------------------fluid---------------------
__device__
inline float getH(float diameter)
{
	return ((float)NEIGHBOUR_R + 0.5f) * diameter;
	//return (float)NEIGHBOUR_R * diameter;
}

__device__
inline float getRHO0(float diameter)
{
	return 0.9f * 1.0f / powf(diameter / 0.99f, 3.0f);
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

	return r>h ? glm::vec3(0.0f) : (-45.0f) / (float)PI / powf(h, 6.0f) * 
		powf(h-r,2.0f) / r * vec_r;
	//return glm::normalize(vec_r);


	//tmp
	//return 1.0f;
}


__device__
void hitTestVoxelFluid_SolidCollision(int num_voxel, float diameter, int particle_id, int voxel_id, glm::vec3 * predict_positions, glm::vec3 * delta_positions,
Particle * particles, Voxel * grid, int * dev_n)
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
		if (grid[voxel_id].particle_id[i] == particle_id)
		{
			continue;
		}
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
	// Apply average delta X position (treat as results of constraint solver)
	delta_positions[particle_id] += delta_pos;
	dev_n[particle_id] += n;
}

__device__
float fluidInfoSum(glm::vec3 & gradient, float & gradient2,int & n, 
int num_voxel, float diameter, int particle_id, int voxel_id, glm::vec3 * predict_positions, glm::vec3 * delta_positions,
Particle * particles, Voxel * grid, int * dev_n)
{
	if (voxel_id < 0 || voxel_id >= num_voxel)
	{
		//voxel_id is invalid
		return;
	}

	
	float density = 0.0f;	//for particles in this voxel
	for (int i = 0; i < grid[voxel_id].num; i++)
	{
		if (grid[voxel_id].particle_id[i] == particle_id)
		{
			continue;
		}

		//if (particles[grid[voxel_id].particle_id[i]].type != FLUID)
		//{
		//	continue;
		//}

		//pi - pj
		glm::vec3 d = predict_positions[particle_id] - predict_positions[grid[voxel_id].particle_id[i]];
		
		
		//// If particles overlap
		//if (glm::length(d) < diameter)
		//{
		//	float momentWeight = -particles[particle_id].invmass / (particles[particle_id].invmass + particles[grid[voxel_id].particle_id[i]].invmass);
		//	// Move particle i along the vector so that i and j are in the post-collision states
		//	delta_pos += momentWeight * glm::normalize(d) * (diameter - glm::length(d));
		//	n++;
		//}
		

		n++;
		//density += (particles[grid[voxel_id].particle_id[i]].type == FLUID ? 1.0f : (particles[particle_id].invmass < FLT_EPSILON ? 10.0f : 1.0f / particles[particle_id].invmass))
		//	* SmoothKernel(predict_positions[particle_id] - predict_positions[grid[voxel_id].particle_id[i]], H_KERNAL_WIDTH);

		density += SmoothKernel(glm::length(d), getH(diameter));

		glm::vec3 g = gradientSmoothKernel(d, getH(diameter));
		
		//FACT: g and d has opposite direction

		gradient += g;
		gradient2 += glm::dot(g,g);

		//tmp
		//delta_pos += -0.5f * glm::normalize(d) * (diameter - glm::length(d));
		//delta_pos += - 0.0001f * glm::normalize(d) / glm::dot(d,d);

		

	}
	//correct version should update based on gradient matrix of Kernal 
	//update all positions together

	return density;
}



__device__
void fluidNeighbourEnforce(float* lambda, //int * mutex,
int num_voxel, float diameter, int particle_id, int voxel_id, glm::vec3 * predict_positions, glm::vec3 * delta_positions,
Particle * particles, Voxel * grid, int * dev_n)
{
	if (voxel_id < 0 || voxel_id >= num_voxel)
	{
		//voxel_id is invalid
		return;
	}


	//glm::vec3 delta_pos(0.0);

	for (int i = 0; i < grid[voxel_id].num; i++)
	{
		if (grid[voxel_id].particle_id[i] == particle_id)
		{
			continue;
		}

		//if (particles[grid[voxel_id].particle_id[i]].type != FLUID)
		//{
		//	continue;
		//}

		// Distance vector from particle i to particle j (on particle centers)
		glm::vec3 d = predict_positions[particle_id] - predict_positions[grid[voxel_id].particle_id[i]];
		
		//gradent of W(pi-pj)
		glm::vec3 g = gradientSmoothKernel(d, getH(diameter));

		
		delta_positions[particle_id] += 1.0f / getRHO0(diameter) 
			* (lambda[particle_id] + lambda[grid[voxel_id].particle_id[i]]) * g;

		dev_n[particle_id] += 1;
		
		//WARNING: race conditions may exist
		
		//lock
		
		//int old = mutex[grid[voxel_id].particle_id[i]];
		//int old;
		//do
		//{
		//	old = atomicCAS(&mutex[grid[voxel_id].particle_id[i]], 0, 1);
		//} while (0 != old);


		//while (atomicCAS(&mutex[grid[voxel_id].particle_id[i]], 0, 1) != 0){}

		//g *= min(0.0f, lambda_i);

		//if (/*particles[particle_id].y < -1.0f &&*/ particle_id == 1000)
		//{
		//	printf("lambda=%f,g.x=%f,g.y=%f,g.z=%f\n", lambda_i, g.x, g.y,g.z);
		//}

		//delta_positions[grid[voxel_id].particle_id[i]] += g;
		//delta_positions[grid[voxel_id].particle_id[i]] += - min(0.0f,lambda_i) * g;
		//dev_n[grid[voxel_id].particle_id[i]] += 1;


		//unlock
		//atomicExch(&mutex[grid[voxel_id].particle_id[i]], 0);
		//mutex[grid[voxel_id].particle_id[i]] = 0;
		//atomicCAS(&mutex[grid[voxel_id].particle_id[i]], 1, 0);



		//if (/*particles[particle_id].y < -1.0f &&*/ particle_id == 1000)
		//{
		//	printf("%f,%f,%f\n", lambda_i, d.x, g.x);
		//}
		

		//int assume = dev_n[grid[voxel_id].particle_id[i]];

		//do
		//{
		//	assume = atomicAdd(&dev_n[grid[voxel_id].particle_id[i]], 1);
		//} while (assume == dev_n[grid[voxel_id].particle_id[i]]);

		//delta_positions[grid[voxel_id].particle_id[i]] +=  min(0.0f,lambda_i) * g;
		

		

		//glm::vec3 delta = -lambda_i * g / rho_0;
		//atomicAdd(&delta_positions[grid[voxel_id].particle_id[i]].x, );
		//atomicAdd(float* address, float val);
	}

	
}



//---------------------------------------------









//Handle Constraints
__global__
void handleCollision(int N, int num_voxel, float diameter, glm::ivec3 resolution, float* lambda,//int * mutex,
	 glm::vec3 * predictPositions, glm::vec3 * deltaPositions, Particle * particles,Voxel * grid, int * ids, float delta_t, int * dev_n)
{

	int particle_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (particle_id < N)
	{
		int voxel_id = ids[particle_id];

		// Collision detection & reaction; simplified SDF constraint
		// hitTest particles in neighbour voxel


		if (particles[particle_id].type == SOLID)
		{
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int z = -1; z <= 1; z++)
					{
						hitTestVoxelSolid(num_voxel, diameter, particle_id,
							voxel_id + z * 1 + y * resolution.z + x * resolution.y * resolution.z,
							predictPositions, deltaPositions, particles, grid, dev_n);
					}
				}
			}
		}


	}


	__syncthreads();


	if (particle_id < N)
	{
		int voxel_id = ids[particle_id];


		if (particles[particle_id].type == FLUID)
		{
			//C_i(x) = density/density0 - 1.0
			//gradient_C_i(x)

			//delta_X = - C_i / sum( gradient(C_i)^2 )

			float density = 0.0f;
			glm::vec3 sum_gradient(0.0f);
			float sum_gradient2 = 0.0f;
			glm::vec3 delta_pos(0.0f);
			int n = 0;


			//solid collision (including self)
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int z = -1; z <= 1; z++)
					{
						//contains self collision
						/*hitTestVoxelFluid_SolidCollision(num_voxel, diameter, particle_id,
							voxel_id + z * 1 + y * resolution.z + x * resolution.y * resolution.z,
							predictPositions, deltaPositions, particles, grid, dev_n);*/

						//no self collision
						hitTestVoxelSolid(num_voxel, diameter, particle_id,
							voxel_id + z * 1 + y * resolution.z + x * resolution.y * resolution.z,
							predictPositions, deltaPositions, particles, grid, dev_n);
					}
				}
			}


			//first loop used to get the sum of density rho_i, sum of gradient
			//fluid density constraint
			//for (int x = -NEIGHBOUR_R; x <= NEIGHBOUR_R; x++)
			//{
			//	for (int y = -NEIGHBOUR_R; y <= NEIGHBOUR_R; y++)
			//	{
			//		for (int z = -NEIGHBOUR_R; z <= NEIGHBOUR_R; z++)
			for (int x = -NEIGHBOUR_R; x <= NEIGHBOUR_R; x++)
			{
				for (int y = -NEIGHBOUR_R; y <= NEIGHBOUR_R; y++)
				{
					for (int z = -NEIGHBOUR_R; z <= NEIGHBOUR_R; z++)
					{
						density += fluidInfoSum(sum_gradient, sum_gradient2, n,
							num_voxel, diameter, particle_id,
							voxel_id + z * 1 + y * resolution.z + x * resolution.y * resolution.z,
							predictPositions, deltaPositions, particles, grid, dev_n);
					}
				}
			}


			float rho_0 = getRHO0(diameter);

			// when density / rho_0 -1.0f > 0 , move
			// i.e. when lambda < 0, move
			
			//float lambda_i =  -(density / rho_0 - 1.0f) / (sum_gradient2 + glm::dot(sum_gradient, sum_gradient));

			
			
			//for testing
			float ci = density / rho_0 - 1.0f;
			float denominator = sum_gradient2 + glm::dot(sum_gradient, sum_gradient) + LAMBDA_EPSILON;
			float lambda_i = - ci / denominator;

			lambda[particle_id] = min(0.0f,lambda_i);
			//lambda_i = -lambda_i;
			//if (/*particles[particle_id].y < -1.0f &&*/ particle_id == 955)
			//{
			//	printf("%f,%f,%f,lambda=%f\n", density, rho_0, denominator, lambda_i);
			//}
				
			

			//sum_gradient *= min(0.0f, lambda_i) / rho_0;

			//if (/*particles[particle_id].y < -1.0f &&*/ particle_id == 1000)
			//{
			//	printf("self -- lambda=%f,g.x=%f,g.y=%f,g.z=%f\n", lambda_i, sum_gradient.x, sum_gradient.y, sum_gradient.z);
			//}
			//deltaPositions[particle_id] += sum_gradient;
			

			//deltaPositions[particle_id] += min(0.0f, lambda_i) * sum_gradient / rho_0;
			//dev_n[particle_id] += 1;



			////second loop used to calculate delta pos for neighbour particle
			//for (int x = -NEIGHBOUR_R; x <= NEIGHBOUR_R; x++)
			//{
			//	for (int y = -NEIGHBOUR_R; y <= NEIGHBOUR_R; y++)
			//	{
			//		for (int z = -NEIGHBOUR_R; z <= NEIGHBOUR_R; z++)
			//		{
			//			fluidNeighbourEnforce(lambda_i,rho_0, lambda,//mutex,
			//				num_voxel, diameter, particle_id,
			//				voxel_id + z * 1 + y * resolution.z + x * resolution.y * resolution.z,
			//				predictPositions, deltaPositions, particles, grid, dev_n);
			//		}
			//	}
			//}
			


			//deltaPositions[particle_id] += delta_pos / (n > 0 ? (float)n: 100.0f) * 1.0f * lambda;
			//if (n > 0){
			//	printf("%f,%f,%f\t%f,%f,%d\n", delta_pos.x, delta_pos.y, delta_pos.z, density, gradient, n);
			//}


			//naive way
			//n = min(n, 1);
			//deltaPositions[particle_id] += max( -1.0f * diameter, min(0.0f, density - 6.0f / (diameter*diameter)) ) * delta_pos / (float)n;
			//dev_n[particle_id] += 1;

		}
	}
}



//fluid apply delta
__global__
void FluidApplyLambdaDelta(int N, int num_voxel, float diameter, glm::ivec3 resolution, float* lambda,//int * mutex,
glm::vec3 * predictPositions, glm::vec3 * deltaPositions, Particle * particles, Voxel * grid, int * ids, float delta_t, int * dev_n)
{
	int particle_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (particle_id < N)
	{
		int voxel_id = ids[particle_id];

		// Collision detection & reaction; simplified SDF constraint
		// hitTest particles in neighbour voxel


		if (particles[particle_id].type == FLUID)
		{
			for (int x = -NEIGHBOUR_R; x <= NEIGHBOUR_R; x++)
			{
				for (int y = -NEIGHBOUR_R; y <= NEIGHBOUR_R; y++)
				{
					for (int z = -NEIGHBOUR_R; z <= NEIGHBOUR_R; z++)
					{
						fluidNeighbourEnforce(lambda,//mutex,
											num_voxel, diameter, particle_id,
											voxel_id + z * 1 + y * resolution.z + x * resolution.y * resolution.z,
											predictPositions, deltaPositions, particles, grid, dev_n);
					}
				}
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
void shapeMatching(int base, int size, glm::vec3 * delta_positions, glm::vec3 * predictions, glm::vec3 *x0, glm::vec3 cm0, glm::vec3 cm, glm::mat3 * dev_Apq_ptr, int * dev_n){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < size){
		threadId += base;

		glm::mat3 Apq(0.0f), R(0.0f), ROOT(0.0f);

		// TODO extract to one single pre-calculation (as it was before)
		for (int i = 0; i < size; i++){
			Apq += dev_Apq_ptr[i];
		}

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
		float nd = n[threadId + base] == 0 ? 1.0 : (float)n[threadId + base];
		predictions[threadId] += delta[threadId + base] / nd;
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
		if (particles[threadId].type == SOLID)
		{
			particles[threadId].x = particles[threadId].x + glm::trunc((predictions[threadId] - particles[threadId].x)*SPLEEFING_COFF) / SPLEEFING_COFF;
		}
		else
		{
			particles[threadId].x = particles[threadId].x + (predictions[threadId] - particles[threadId].x);
		}

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
	// FIXME below memset breaks updateVoxelIndex; not sure why
	//cudaMemset(dev_particle_voxel_id, 0, num_voxel * sizeof(Voxel));

	//update
	updateVoxelIndex << <blockCountr, blockSizer >> >(num_particles, grid_resolution, grid_min_x, grid_length, dev_predictPosition, dev_grid, dev_particle_voxel_id);
	//updateVoxelIndexBefore << <blockCountr, blockSizer >> >(num_particles, grid_resolution, grid_min_x, grid_length, dev_positions, dev_grid, dev_particle_voxel_id);
	checkCUDAError("ERROR: updateVoxelIndex");

	//detect collisions and generate collision constraints
	//for fluid, get density and constraints
	handleCollision << <blockCountr, blockSizer >> >(num_particles, num_voxel, grid_length * 0.99f, grid_resolution, dev_lambda,//dev_mutex,
		dev_predictPosition, dev_deltaPosition, dev_particles, dev_grid, dev_particle_voxel_id, delta_t, dev_n);
	checkCUDAError("ERROR: handle collision");

	cudaDeviceSynchronize();

	FluidApplyLambdaDelta << <blockCountr, blockSizer >> >(num_particles, num_voxel, grid_length * 0.99f, grid_resolution, dev_lambda,//dev_mutex,
		dev_predictPosition, dev_deltaPosition, dev_particles, dev_grid, dev_particle_voxel_id, delta_t, dev_n);
	checkCUDAError("ERROR: handle collision");

	cudaDeviceSynchronize();

	//---- Shape matching constraint --------
	int base = 0;
	for (int i = 0; i < num_rigidBodies; i++)
	{
		ParticleType body_type = rigidBody[i].getType();
		int size = rigidBody[i].m_particles.size();
		dim3 blockCountrPerRigidBody((size + blockSizer - 1) / blockSizer);

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

			//modify predict positions
			// Also find A and R within the kernel
			shapeMatching << <blockCountrPerRigidBody, blockSizer >> >(base, size, dev_deltaPosition, dev_predictPosition, dev_particle_x0, hst_cm0[i], cm, dev_Apq_ptr, dev_n);

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