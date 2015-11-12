#pragma once

#include <glm/glm.hpp>
#include <util/utilityCore.hpp>
#include "util/tiny_obj_loader.h"

//CUDA-accelerated particle sampling of rigid body obj models


void samplingInit(int num_v);
void samplingSetBuffers(float * hst_positions, unsigned int * hst_indices);
void samplingFree();


