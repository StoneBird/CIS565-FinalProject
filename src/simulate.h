#pragma once

#include <glm/glm.hpp>
#include <util/utilityCore.hpp>
#include "util/tiny_obj_loader.h"
#include "Particle.h"
#include "rasterizeTools.h"
#include "RigidBody.h"

//CUDA-accelerated particle sampling of rigid body obj models


//init simulation
//void initSimulate();
//assamble particle arrays from all rigidbodys into one array
void assembleParticleArray(int num_rigidBody, RigidBody * rigidbodys);



/////////////////////////



//simulate loop functions

void simulate(const glm::vec3 forces, const float delta_t, float * opengl_buffer);




void updateParticles();

void endSimulation();