#pragma once

#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

enum ParticleType
{
	SOLID,FLUID
};


struct Particle
{
	glm::vec3 x;	//position
	glm::vec3 v;	//velocity

	float invmass;	//1/mass, 0 means static object, mass = infinity

	int phase;		//group

	ParticleType type;	//type: fluid, solid...
};

struct ParticleWrapper
{
	glm::vec3 x;
	bool isEmpty;
};

struct Triangle
{
	glm::vec3 v[3];
};

struct RayPeel
{
	glm::vec2 peel;
};

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};

struct is_empty{
	__host__ __device__
		bool operator()(const ParticleWrapper &p)
	{
		return p.isEmpty;
	}
};