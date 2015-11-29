#include <cmath>
#include <cstdio>
#include <cuda.h>

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include <vector>

#include <cuda_runtime.h>
#include <util/checkCUDAError.h>
#include <glm/gtx/transform.hpp>
#include "fluid.h"
#include "simulate.h"
