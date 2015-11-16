#pragma once

#include <glm/glm.hpp>
#include <util/utilityCore.hpp>
#include "util/tiny_obj_loader.h"
#include "Particle.h"
#include "rasterizeTools.h"

//CUDA-accelerated particle sampling of rigid body obj models


void samplingInit(int num_v, glm::vec3, float);
void samplingSetBuffers(float * hst_positions, unsigned int * hst_indices);
void samplingFree();
void sampleParticles(std::vector<Particle> &, std::vector<float> &, const glm::vec3);


