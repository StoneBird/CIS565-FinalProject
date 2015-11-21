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
static Particle * dev_particle_cache;
static float * dev_particle_pos_cache;
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
	cudaMalloc(&dev_triangles, num_triangles * sizeof(Triangle));

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
		int v1_id = threadId * 3;
		unsigned int p1_id = indices[v1_id] * 3;
		unsigned int p2_id = indices[v1_id + 1] * 3;
		unsigned int p3_id = indices[v1_id + 2] * 3;
		triangles[threadId].v[0] = glm::vec3(position[p1_id], position[p1_id + 1], position[p1_id + 2]);
		triangles[threadId].v[1] = glm::vec3(position[p2_id], position[p2_id + 1], position[p2_id + 2]);
		triangles[threadId].v[2] = glm::vec3(position[p3_id], position[p3_id + 1], position[p3_id + 2]);
	}
}

void samplingSetBuffers(float * hst_positions, unsigned int * hst_indices)
{
	cudaMemcpy(dev_positions, hst_positions, num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_indices, hst_indices, num_vertices * sizeof(unsigned int), cudaMemcpyHostToDevice);
	checkCUDAError("Obj index copy");
	
	//triangle assembly
	const int blockSize = 192;
	dim3 blockCount((num_vertices + blockSize - 1) / blockSize);

	kernTriangleAssembly << <blockCount, blockSize >> >(num_triangles, dev_triangles, dev_positions, dev_indices);
	checkCUDAError("Obj assemble");
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
void intersect(RayPeel * rp, Triangle * tri, const int tri_count, const glm::vec3 resolution, const float diameter, const glm::vec3 ray){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = x + y*resolution.x;
	if (x < resolution.x && y < resolution.y){
		coordRemap(x, y, resolution);
		bool hasIntersect = false;
		float xReal = x*diameter, yReal = y*diameter;
		float minZ = resolution.z / 2 * diameter, maxZ = -resolution.z / 2*diameter;
		for (int t = 0; t < tri_count; t++){
			AABB tbox = getAABBForTriangle(tri[t]);
			if (
				xReal >= tbox.min.x && xReal <= tbox.max.x &&
				yReal >= tbox.min.y && yReal <= tbox.max.y
				){
				glm::vec3 bcc = calculateBarycentricCoordinate(tri[t], glm::vec2(xReal, yReal));
				if (isBarycentricCoordInBounds(bcc)){
					hasIntersect = true;
					float z = getZAtCoordinate(bcc, tri[t]);
					if (z < minZ){
						minZ = z;
					}
					if (z > maxZ){
						maxZ = z;
					}
				}
			}
		}
		glm::vec2 peel = glm::vec2(minZ, maxZ);
		if (!hasIntersect){
			peel = glm::vec2(0.0f);
		}
		rp[idx].peel = peel/diameter;
	}
}

__global__
void fillPeel(ParticleWrapper * p_out, RayPeel * rp, const glm::vec3 resolution, const float diameter){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int idx = x + y*resolution.x;
	if (x < resolution.x && y < resolution.y){
		glm::vec2 p = rp[idx].peel;
		int depth = ceil(abs(p.y - p.x));
		coordRemap(x, y, resolution);
		for (int z = 0; z < depth; z++){
			p_out[idx*(int)resolution.z + z].x = glm::vec3(x, y, z+p.x)*diameter;
			p_out[idx*(int)resolution.z + z].isEmpty = false;
		}
		for (int z = depth; z < resolution.z; z++){
			p_out[idx*(int)resolution.z + z].isEmpty = true;
		}
	}
}

__global__
void transformParticle(float *pos_out, Particle *p_out, ParticleWrapper *p_in, int size, const glm::mat4 mat, const glm::vec3 body_init_velocity, const float body_mass_scale){
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < size){
		glm::vec3 pos = p_in[threadId].x;
		int pIdx = threadId * 3;

		//p_out[threadId].x = pos+offset;
		//pos_out[pIdx] = pos.x+offset.x;
		//pos_out[pIdx + 1] = pos.y+offset.y;
		//pos_out[pIdx + 2] = pos.z+offset.z;
		glm::vec4 tmp = mat * glm::vec4(pos, 1.0f);
		tmp /= tmp.w;
		p_out[threadId].x = glm::vec3( tmp.x,tmp.y,tmp.z );

		// Use rigid body's velocity
		p_out[threadId].v = body_init_velocity;

		// Scale particle mass (so that static objects can work)
		// TODO assign invmass somewhere else
		p_out[threadId].invmass = 1.0f;
		p_out[threadId].invmass = p_out[threadId].invmass * body_mass_scale;

		pos_out[pIdx] = tmp.x;
		pos_out[pIdx + 1] = tmp.y;
		pos_out[pIdx + 2] = tmp.z;
	}
}

void sampleParticles(std::vector<Particle> &hst_p, std::vector<float> &hst_pos, const glm::mat4 mat, const glm::vec3 body_init_velocity, const float body_mass_scale){
	const int blockSideLength = 8;
	const dim3 blockSize(blockSideLength, blockSideLength);
	dim3 blocksPerGrid(
		(resolution.x + blockSize.x - 1) / blockSize.x,
		(resolution.y + blockSize.y - 1) / blockSize.y);

	thrust::device_vector<ParticleWrapper> dev_grid(num_particles);
	ParticleWrapper * dev_particles = thrust::raw_pointer_cast(&dev_grid[0]);
	checkCUDAError("Malloc thrust");

	// Depth peeling
		// Intersection test
	intersect << <blocksPerGrid, blockSize >> >(dev_peels, dev_triangles, num_triangles, resolution, voxel_diam, ray);
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
	cudaMalloc(&dev_particle_cache, newSize * sizeof(Particle));
	checkCUDAError("Malloc 1");
	cudaMalloc(&dev_particle_pos_cache, newSize * 3 * sizeof(float));
	checkCUDAError("Malloc 2");

	const int blockSizer = 192;
	dim3 blockCountr((newSize + blockSizer - 1) / blockSizer);
	transformParticle << <blockCountr, blockSizer >> >(dev_particle_pos_cache, dev_particle_cache, dev_particles, newSize, mat, body_init_velocity, body_mass_scale);
	checkCUDAError("Wrapper translation");

	// Copy to host
	Particle *p_out = (Particle *)malloc(newSize * sizeof(Particle));
	float *pos_out = (float *)malloc(newSize * 3 * sizeof(float));

	cudaMemcpy(p_out, dev_particle_cache, newSize * sizeof(Particle), cudaMemcpyDeviceToHost);
	checkCUDAError("Memcpy particle");
	cudaMemcpy(pos_out, dev_particle_pos_cache, newSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("Memcpy particle pos");

	hst_p.resize(newSize);
	hst_pos.resize(newSize * 3);

	std::copy(p_out, p_out + newSize, hst_p.begin());
	std::copy(pos_out, pos_out + newSize*3, hst_pos.begin());

	cudaFree(dev_particle_cache);
	cudaFree(dev_particle_pos_cache);
	free(p_out);
	free(pos_out);
}



