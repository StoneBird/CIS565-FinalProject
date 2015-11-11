#pragma once

#include <glm/vec3.hpp>

struct Particle
{
	glm::vec3 x;	//position
	glm::vec3 v;	//velocity

	float invmass;

	int phase;		//group
};