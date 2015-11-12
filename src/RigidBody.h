#pragma once

#include "src/Particle.h"
#include "util/tiny_obj_loader.h"
#include <vector>
#include <string>


using namespace std;

class RigidBody
{
protected:
	glm::ivec3 m_resolution;

	glm::vec3 m_min;
	glm::vec3 m_max;

	//particle
	vector<Particle> m_particles;

	//obj geometry
	vector<tinyobj::shape_t> m_shapes;
	vector<tinyobj::material_t> m_materials;
public:

	bool initObj(const string & filename);

	void initBoundingBox();

	//Call this after Obj is loaded
	void initParticles(glm::ivec3 resolution);
};