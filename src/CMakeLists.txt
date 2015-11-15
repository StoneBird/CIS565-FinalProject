set(SOURCE_FILES
    "rasterize.cu"
    "rasterize.h"
    "Particle.h"
    "RigidBody.h"
    "RigidBody.cpp"
    "particleSampling.h"
    "particleSampling.cu"
	"rasterizeTools.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
