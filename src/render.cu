#include "render.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>      // for FLT_MAX
#include <cmath>       // for std::sqrt, std::floor
#include <algorithm>   // for std::min

// Static device memory pointers and configuration parameters
static uint32_t* d_colorBuffer = nullptr;  // Device pointer for ARGB pixel framebuffer
static float*    d_depthBuffer = nullptr;  // Device pointer for depth buffer (float)
static int       frameWidth = 0;
static int       frameHeight = 0;
static dim3      threadsPerBlock(16, 16);   // Thread block dimensions (will be adjusted in initCuda)
static dim3      blocksPerGrid(1, 1);       // Grid dimensions (computed based on frame size)

/**
 * @brief CUDA kernel to clear the color and depth buffers.
 *
 * Each thread sets one pixel in the color buffer to the specified clear color and 
 * the corresponding depth value to clearDepth. Threads outside the framebuffer range do nothing.
 */
__global__ static void clearBuffers(uint32_t* colorBuf, float* depthBuf,
                                    int width, int height,
                                    uint32_t clearColor, float clearDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return; // Skip threads outside of buffer bounds
    int idx = y * width + x;
    colorBuf[idx] = clearColor;
    depthBuf[idx] = clearDepth;
}

/**
 * @brief CUDA kernel to rasterize a single triangle with depth testing.
 *
 * Each thread corresponds to a pixel. It calculates barycentric coordinates (w0, w1, w2) for that pixel 
 * relative to the triangle vertices. If the pixel lies inside the triangle (all weights >= 0), 
 * it interpolates the depth value at that pixel and performs a depth test. If the pixel is closer 
 * (depth smaller) than the current value in the depth buffer, the color and depth buffers are updated.
 *
 * @param colorBuf Output color buffer (device memory, ARGB32 format).
 * @param depthBuf Output depth buffer (device memory, float).
 * @param width    Framebuffer width in pixels.
 * @param height   Framebuffer height in pixels.
 * @param x0,y0    Screen-space X and Y of triangle vertex 0.
 * @param x1,y1    Screen-space X and Y of triangle vertex 1.
 * @param x2,y2    Screen-space X and Y of triangle vertex 2.
 * @param z0,z1,z2 Depth values at vertices 0, 1, 2.
 * @param color    Fill color for the triangle (in ARGB32 format).
 */
__global__ static void rasterizeTriDepth(uint32_t* colorBuf, float* depthBuf,
                                         int width, int height,
                                         float x0, float y0,
                                         float x1, float y1,
                                         float x2, float y2,
                                         float z0, float z1, float z2,
                                         uint32_t color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    // Compute barycentric coordinates for point (x, y) with respect to the triangle
    float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (denom == 0.0f) {
        // Degenerate triangle (zero area), skip any drawing
        return;
    }
    float w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom;
    float w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom;
    float w2 = 1.0f - w0 - w1;

    // Check if the pixel lies inside the triangle (using barycentric weights)
    if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
        // Interpolate depth at this pixel
        float pixelDepth = w0 * z0 + w1 * z1 + w2 * z2;
        // Depth test: update pixel only if it is closer (smaller depth value) than current depth
        if (pixelDepth < depthBuf[idx]) {
            colorBuf[idx] = color;
            depthBuf[idx] = pixelDepth;
        }
    }
}

// Initialize CUDA resources: allocate buffers and configure kernel launch parameters
extern "C" bool initCuda(int width, int height) {
    frameWidth = width;
    frameHeight = height;
    size_t colorBufSize = size_t(width) * size_t(height) * sizeof(uint32_t);
    size_t depthBufSize = size_t(width) * size_t(height) * sizeof(float);

    // Allocate device memory for the framebuffer and depth buffer
    cudaError_t errC = cudaMalloc(&d_colorBuffer, colorBufSize);
    cudaError_t errD = cudaMalloc(&d_depthBuffer, depthBufSize);
    if (errC != cudaSuccess || errD != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc for frame buffers failed: %s\n",
                    cudaGetErrorString(errC != cudaSuccess ? errC : errD));
        return false;
    }

    // Determine an optimal thread block size (square block up to 32x32) based on device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    // Choose block dimension as largest power-of-two <= min(32, sqrt(maxThreads))
    int maxDim = std::min(32, (int)std::floor(std::sqrt((float) maxThreads)));
    int blockSide = 1;
    while (blockSide * 2 <= maxDim) {
        blockSide *= 2;
    }
    threadsPerBlock = dim3(blockSide, blockSide);
    // Calculate grid dimensions to cover entire framebuffer (using ceiling division)
    blocksPerGrid = dim3((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::printf("CUDA init successful: Device \"%s\", maxThreadsPerBlock=%d, using block=(%d,%d), grid=(%d,%d)\n",
                prop.name, maxThreads,
                threadsPerBlock.x, threadsPerBlock.y,
                blocksPerGrid.x, blocksPerGrid.y);
    return true;
}

// Rasterize triangles on the GPU and copy the result back to host memory
extern "C" void renderFrame(const CudaTri* tris, int numTris, uint32_t* hostPix, float* hostDepth) {
    // 1. Clear the GPU buffers to a default color and depth
    uint32_t clearColor = 0xFF000000u; // ARGB format: 0xFF000000 = opaque black
    float    clearDepth = FLT_MAX;     // Use a very large depth (far plane)
    clearBuffers<<<blocksPerGrid, threadsPerBlock>>>(d_colorBuffer, d_depthBuffer,
                                                     frameWidth, frameHeight,
                                                     clearColor, clearDepth);

    // 2. Rasterize each triangle (in white color) with depth testing
    uint32_t triColor = 0xFFFFFFFFu; // White color (ARGB: 0xFFFFFFFF). 
    // In a more advanced renderer, this could vary per triangle or be computed via a shader.
    for (int i = 0; i < numTris; ++i) {
        const CudaTri& t = tris[i];
        rasterizeTriDepth<<<blocksPerGrid, threadsPerBlock>>>(d_colorBuffer, d_depthBuffer,
                                                              frameWidth, frameHeight,
                                                              t.x0, t.y0,
                                                              t.x1, t.y1,
                                                              t.x2, t.y2,
                                                              t.z0, t.z1, t.z2,
                                                              triColor);
    }

    // 3. Synchronize and copy the results back to host memory
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during rasterization: %s\n", cudaGetErrorString(err));
        // (If a kernel failed, we still attempt to copy back what was rendered for debugging)
    }
    // Copy color buffer from device to host
    err = cudaMemcpy(hostPix, d_colorBuffer,
                     size_t(frameWidth) * size_t(frameHeight) * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy (color buffer) failed: %s\n", cudaGetErrorString(err));
    }
    // Copy depth buffer from device to host
    err = cudaMemcpy(hostDepth, d_depthBuffer,
                     size_t(frameWidth) * size_t(frameHeight) * sizeof(float),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy (depth buffer) failed: %s\n", cudaGetErrorString(err));
    }
}

// Release CUDA resources
extern "C" void cleanupCuda() {
    if (d_colorBuffer) {
        cudaFree(d_colorBuffer);
        d_colorBuffer = nullptr;
    }
    if (d_depthBuffer) {
        cudaFree(d_depthBuffer);
        d_depthBuffer = nullptr;
    }
}
