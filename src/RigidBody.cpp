#include "RigidBody.h"
#include <algorithm>

#include "particleSampling.h"

void RigidBody::translate(glm::vec3 offset){
	m_offset = offset;
}

bool RigidBody::initObj(const string & filename)
{
	string err;
	bool ret = tinyobj::LoadObj(m_shapes, m_materials, err,filename.c_str());

	initBoundingBox();
	//TODO: handle error cases

	return ret;
}

void RigidBody::initBoundingBox()
{
	m_min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	m_max = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	//get bounding box
	for (auto shape : m_shapes)
	{
		for (int i = 0; i < shape.mesh.indices.size() / 3; i++)
		{
			//for each tri
			for (int j = 0; j < 3; j++)
			{
				//for each vertex
				int idx = shape.mesh.indices.at(3 * i + j);
				glm::vec3 pos(shape.mesh.positions[3 * idx + 0]
					, shape.mesh.positions[3 * idx + 1]
					, shape.mesh.positions[3 * idx + 2]);
				m_min.x = min(m_min.x, pos.x);
				m_min.y = min(m_min.y, pos.y);
				m_min.z = min(m_min.z, pos.z);

				m_max.x = max(m_max.x, pos.x);
				m_max.y = max(m_max.y, pos.y);
				m_max.z = max(m_max.z, pos.z);
			}


		}

	}
}



void RigidBody::initParticles(int x_res)
{
	// Per object bounding box
	glm::vec3 tmp = m_max - m_min;

	// Use resolution on one axis to compute voxel grid side length
	// Use side length to find # of particles on the other 2 axes
	m_grid_length = tmp.x / ((float)x_res);
	m_resolution = glm::ceil((tmp) / m_grid_length);

	// Transfer model data from host to device; init properties
	// --> Per object particle count
	samplingInit(m_shapes.at(0).mesh.indices.size(), m_resolution, m_grid_length);
	samplingSetBuffers(m_shapes.at(0).mesh.positions.data(), m_shapes.at(0).mesh.indices.data());

	// --> Per object depth peeling
	// --> Stream compaction
	// --> Copy array of Particle to host
	sampleParticles(m_particles, m_particle_pos, m_offset);

	/*
	printf("%f %f %f\n", m_particles[0].x.x, m_particles[0].x.y, m_particles[0].x.z);
	printf("%f %f %f\n", m_particle_pos[0], m_particle_pos[1], m_particle_pos[2]);
	*/
}