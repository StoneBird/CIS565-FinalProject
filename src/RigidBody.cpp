#include "RigidBody.h"
#include <algorithm>
#include <iostream>
#include "particleSampling.h"

RigidBody::RigidBody()
	:m_scale(1.0), m_translate(0.0)
	, m_type(SOLID)
{
}

void RigidBody::setTranslate(glm::vec3 translate)
{
	m_translate = translate;
}

void RigidBody::setScale(glm::vec3 scale)
{
	m_scale = scale;
}

void RigidBody::setRotation(glm::mat4 rot)
{
	m_rotation = rot;
}

void RigidBody::setInitVelocity(glm::vec3 iv){
	// Rigid body initial velocity;
	m_init_velocity = iv;
}

void RigidBody::setMassScale(float s){
	m_mass_scale = s;
}

void RigidBody::setPhase(int p){
	m_phase = p;
}

void RigidBody::setType(ParticleType t)
{
	m_type = t;
}

glm::mat4 RigidBody::getTransformMatrix()
{
	return (glm::translate(m_translate) * m_rotation);
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
	//m_min = glm::vec3(9999.0f);
	//m_max = glm::vec3(-9999.0f);

	//get bounding box
	//for (auto shape : m_shapes)
	for (tinyobj::shape_t& shape : m_shapes)
	{
		for (int i = 0; i < shape.mesh.indices.size() / 3; i++)
		{
			//for each tri
			for (int j = 0; j < 3; j++)
			{
				//for each vertex
				int idx = shape.mesh.indices.at(3 * i + j);

				//scale
				shape.mesh.positions[3 * idx + 0] *= m_scale.x;
				shape.mesh.positions[3 * idx + 1] *= m_scale.y;
				shape.mesh.positions[3 * idx + 2] *= m_scale.z;


				glm::vec3 pos(shape.mesh.positions[3 * idx + 0]
					, shape.mesh.positions[3 * idx + 1]
					, shape.mesh.positions[3 * idx + 2]);

				//all particle mass is identical
				//m_cm += pos;


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
	float grid_size = tmp.x / ((float)x_res);

	initParticles(grid_size);
}


void RigidBody::initParticles(float grid_size)
{
	m_grid_length = grid_size;

	m_resolution = glm::ceil((m_max - m_min) / m_grid_length);

	// Transfer model data from host to device; init properties
	// --> Per object particle count
	samplingInit(m_shapes.at(0).mesh.indices.size(), m_resolution, m_grid_length);
	samplingSetBuffers(m_shapes.at(0).mesh.positions.data(), m_shapes.at(0).mesh.indices.data());

	// --> Per object depth peeling
	// --> Stream compaction
	// --> Copy array of Particle to host
	// Apply transformation before calculating center of mass
	//sampleParticles(m_particles, m_particle_pos, getTransformMatrix(), m_init_velocity, m_mass_scale, m_phase,m_type);
	sampleParticles(m_particles, m_particle_pos, glm::mat4(), m_init_velocity, m_mass_scale, m_phase, m_type);

	int num_particle = m_particles.size();

	// calculate center of mass
	// cpu version
	m_cm = glm::vec3(0.0);
	for (int i = 0; i < num_particle; i++)
	{
		glm::vec3 p(m_particle_pos.at(3 * i + 0), m_particle_pos.at(3 * i + 1), m_particle_pos.at(3 * i + 2));
		m_cm += p;

		// Initialize x0 (positions in rest config) at the same time
		m_x0.push_back(m_particles[i].x);
	}
	m_cm /= (float)num_particle;

	
}



void RigidBody::setColor(const glm::vec4 & c)
{
	m_color[0] = c.r;
	m_color[1] = c.g;
	m_color[2] = c.b;
	m_color[3] = c.a;
}