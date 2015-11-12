#include "RigidBody.h"
#include <algorithm>

#include "particleSampling.h"


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



void RigidBody::initParticles(glm::ivec3 resolution)
{
	m_resolution = resolution;
	



	//TODO
	//transfer model data from host to device by
	//Calling cuda functions
	//test
	samplingInit(m_shapes.at(0).mesh.indices.size());
	samplingSetBuffers(m_shapes.at(0).mesh.positions.data(), m_shapes.at(0).mesh.indices.data());
	samplingFree();
	//cuda part return 2 depth 2D textures (float[][])


	//cpu build m_particles vector due to this 2 depth texture
}