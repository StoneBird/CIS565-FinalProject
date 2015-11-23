set(SOURCE_FILES
	"checkCUDAError.h"
    "stb.cpp"
    "utilityCore.hpp"
    "utilityCore.cpp"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "obj.hpp"
    "obj.cpp"
    "objloader.hpp"
    "objloader.cpp"
    "tiny_obj_loader.h"
    "tiny_obj_loader.cc"
	"svd3_cuda.h"
   )

cuda_add_library(util
   ${SOURCE_FILES}
   OPTIONS -arch=sm_20
   )
