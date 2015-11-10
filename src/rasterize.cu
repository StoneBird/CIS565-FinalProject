/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <util/checkCUDAError.h>
#include <vector>
#include <glm/gtx/transform.hpp>
#include "rasterize.h"

static int width = 0;
static int height = 0;
__constant__ static int *dev_bufIdx = NULL;
__constant__ static int *dev_depth = NULL;
__constant__ static glm::vec3 *dev_framebuffer = NULL;

// Temp variables for stream compaction
__constant__ static int *dv_f_tmp = NULL;
__constant__ static int *dv_idx_tmp = NULL;
__constant__ static int *dv_c_tmp = NULL;

__global__ void sendImageToPBO(uchar4 *pbo, int w, int h) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		pbo[index].w = 0;
		glm::vec3 color = glm::vec3(255.0f);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].x = color.x*glm::clamp(0.8f, 0.0f, 1.0f);
		pbo[index].y = color.y*glm::clamp(0.8f, 0.0f, 1.0f);
		pbo[index].z = color.z*glm::clamp(0.8f, 0.0f, 1.0f);
	}
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
	height = h;
    //cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

    checkCUDAError("rasterizeInit");
}

/**
 * Set all of the buffers necessary for rasterization.
 */
void rasterizeSetBuffers(
        int _bufIdxSize, int *bufIdx,
        int _vertCount, float *bufPos, float *bufNor, float *bufCol) {
	/*
    bufIdxSize = _bufIdxSize;
    vertCount = _vertCount;

    //cudaFree(dev_bufIdx);
    cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
    cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);

    VertexIn *bufVertex = new VertexIn[_vertCount];
    for (int i = 0; i < vertCount; i++) {
        int j = i * 3;
        bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
        bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
        bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
    }
    //cudaFree(dev_bufVertex);
    cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
    cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	// Allocate temp vars
	cudaMalloc((void**)&dv_f_tmp, triCount * geomShaderLimit *sizeof(int));
	cudaMalloc((void**)&dv_idx_tmp, triCount * geomShaderLimit *sizeof(int));
	cudaMalloc((void**)&dv_c_tmp, sizeof(int));
	*/

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo) {
	dim3 blockSize2d2(16, 16);

	dim3 blockCount2d2((width + blockSize2d2.x - 1) / blockSize2d2.x,
		(height + blockSize2d2.y - 1) / blockSize2d2.y);

	sendImageToPBO << <blockCount2d2, blockSize2d2 >> >(pbo, width, height);
    checkCUDAError("rasterize");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {
    cudaFree(dev_bufIdx);
    cudaFree(dev_framebuffer);
	cudaFree(dev_depth);

	cudaFree(dv_f_tmp);
	cudaFree(dv_idx_tmp);
	cudaFree(dv_c_tmp);

    checkCUDAError("rasterizeFree");
}