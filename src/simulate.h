#pragma once

#include <glm/glm.hpp>
#include <util/utilityCore.hpp>
#include "util/tiny_obj_loader.h"
#include "Particle.h"
#include "rasterizeTools.h"
#include "RigidBody.h"


//CUDA-accelerated particle sampling of rigid body obj models

#define NUM_PARTICLE_VOXEL (4)

struct Voxel
{
	int num;
	int particle_id[NUM_PARTICLE_VOXEL];
};



//init simulation
//void initSimulate(int num_rigidBody, RigidBody * rigidbodys, glm::vec3 bmin, glm::vec3 bmax, float particle_diameter);


//assamble particle arrays from all rigidbodys into one array
void assembleParticleArray(int num_rigidBody, RigidBody * rigidbodys);

void initUniformGrid(glm::vec3 bmin, glm::vec3 bmax, float diameter);


/////////////////////////



//----------simulate loop functions-------------------

void simulate(const glm::vec3 forces, const float delta_t, float * opengl_buffer, RigidBody * rigidBody);




//----------end Simulation free----------------------

void endSimulation();