#pragma once

#include <glm/glm.hpp>
#include <util/utilityCore.hpp>
#include "util/tiny_obj_loader.h"
#include "Particle.h"
#include "rasterizeTools.h"

//CUDA-accelerated particle sampling of rigid body obj models

void applyForces(int num_particles, const glm::vec3 forces, const float delta_t)
