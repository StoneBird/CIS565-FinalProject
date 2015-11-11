#include "RigidBody.h"


bool RigidBody::initObj(const string & filename)
{
	string err;
	bool ret = tinyobj::LoadObj(m_shapes, m_materials, err,filename.c_str());

	//TODO: handle error cases

	return ret;
}



void RigidBody::initParticles(glm::ivec3 resolution)
{
	m_resolution = resolution;

	//TODO
	//transfer model data from host to device by
	//Calling cuda functions


	//cuda part return 2 depth 2D texture (float[][])


	//cpu build m_particles vector due to this 2 depth texture
}