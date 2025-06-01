/**********************************************************************
 *  IzerRaster - CUDA backend                                         *
 *  - Rasterizare triunghiuri cu Z-buffer                              *
 *  - OPTIONAL: sampling textură RGBA8                                 *
 *********************************************************************/
#include "render.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>      // FLT_MAX
#include <cmath>       // sqrt, floor
#include <algorithm>   // min
#include <cstdint>  

/* ====================================================================
   -- Buffere de frame                                                 */
static uint32_t* d_colorBuffer = nullptr;   // ARGB8   [width * height]
static float*    d_depthBuffer = nullptr;   // float32 [width * height]
static int       frameWidth  = 0;
static int       frameHeight = 0;

/* ==== Config pentru lansare kernel ================================= */
static dim3 threadsPerBlock(16, 16);
static dim3 blocksPerGrid (1 , 1 );

/* ====================================================================
   -- Textură globală (read-only)                                      */
__device__ uint32_t* d_texture = nullptr;   // pixel array RGBA8
__device__ int       d_texW    = 0;
__device__ int       d_texH    = 0;
__device__ bool      d_useTex  = false;     // switch sampling ON/OFF

extern "C" void uploadTexture(const uint32_t* devPixels, int w, int h)
{
    cudaMemcpyToSymbol(d_texture, &devPixels, sizeof(uint32_t*));
    cudaMemcpyToSymbol(d_texW,    &w,         sizeof(int));
    cudaMemcpyToSymbol(d_texH,    &h,         sizeof(int));
}


extern "C" void setTexturing(bool enable)
{
    cudaMemcpyToSymbol(d_useTex, &enable, sizeof(bool));
}

/* ====================================================================
   Device helpers                                                      */
__device__ __forceinline__
uint32_t sampleTexture(float u, float v)
{
    /* clamp / wrap la [0,1] */
    u -= floorf(u);          // wrap (repeat)
    v -= floorf(v);

    int x = int(u * d_texW);
    int y = int((1.f - v) * d_texH);
    x = ::max(0, ::min(x, d_texW  - 1));   // ::max = built-in device func
    y = ::max(0, ::min(y, d_texH  - 1));
    return d_texture[y * d_texW + x];
}

/* ====================================================================
   Kernel: clear color + depth                                         */
__global__ static void clearBuffers(uint32_t* colorBuf, float* depthBuf,
                                    int width, int height,
                                    uint32_t clearColor, float clearDepth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    colorBuf[idx] = clearColor;
    depthBuf[idx] = clearDepth;
}

/* ====================================================================
   Kernel: rasterize triunghi cu Z-buffer + opţional textură           */
__global__ static void rasterizeTri(uint32_t* colorBuf, float* depthBuf,
                                    int width, int height,
                                    float x0, float y0,
                                    float x1, float y1,
                                    float x2, float y2,
                                    float z0, float z1, float z2,
                                    float u0, float v0,
                                    float u1, float v1,
                                    float u2, float v2,
                                    uint32_t flatColor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;

    /* --- barycentric ------------------------------------------------ */
    float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (denom == 0.f) return;                         // degenerate

    float w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom;
    float w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom;
    float w2 = 1.f - w0 - w1;

    if (w0 < 0.f || w1 < 0.f || w2 < 0.f) return;     // pixel în afara

    /* --- depth ------------------------------------------------------ */
    float z = w0 * z0 + w1 * z1 + w2 * z2;
    if (z >= depthBuf[idx]) return;                   // test Z

    /* --- culoare ---------------------------------------------------- */
    uint32_t outColor = flatColor;
    if (d_useTex && d_texture)
    {
        float u = w0 * u0 + w1 * u1 + w2 * u2;
        float v = w0 * v0 + w1 * v1 + w2 * v2;
        outColor = sampleTexture(u, v);
    }

    colorBuf[idx] = outColor;
    depthBuf[idx] = z;
}

/* ====================================================================
   Init CUDA – alocă buffere                                            */
extern "C" bool initCuda(int width, int height)
{
    frameWidth  = width;
    frameHeight = height;

    size_t colorBytes = size_t(width) * height * sizeof(uint32_t);
    size_t depthBytes = size_t(width) * height * sizeof(float);

    cudaError_t ec = cudaMalloc(&d_colorBuffer, colorBytes);
    cudaError_t ed = cudaMalloc(&d_depthBuffer, depthBytes);
    if (ec != cudaSuccess || ed != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc failed: %s\n",
                     cudaGetErrorString(ec != cudaSuccess ? ec : ed));
        return false;
    }

    /* calcul blocuri */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxT = prop.maxThreadsPerBlock;
    int maxDim = std::min(32, int(std::floor(std::sqrt(float(maxT)))));
    int side = 1; while (side * 2 <= maxDim) side *= 2;

    threadsPerBlock = dim3(side, side);
    blocksPerGrid   = dim3( (width  + side - 1) / side,
                            (height + side - 1) / side );

    std::printf("[CUDA] init OK – \"%s\", block %dx%d, grid %dx%d\n",
                prop.name, side, side, blocksPerGrid.x, blocksPerGrid.y);
    return true;
}

/* ====================================================================
   Render frame: clear + loop triunghiuri                               */
extern "C" void renderFrame(const CudaTri* tris, int numTris,
                            uint32_t* hostPix, float* hostDepth)
{
    /* --- clear ------------------------------------------------------ */
    const uint32_t clr = 0xFF000000u;
    clearBuffers<<<blocksPerGrid, threadsPerBlock>>>(d_colorBuffer,
                                                     d_depthBuffer,
                                                     frameWidth,
                                                     frameHeight,
                                                     clr, FLT_MAX);

    /* --- rasterizare ------------------------------------------------ */
    const uint32_t white = 0xFFFFFFFFu;
    for (int i = 0; i < numTris; ++i)
    {
        const CudaTri& t = tris[i];
        rasterizeTri<<<blocksPerGrid, threadsPerBlock>>>(
            d_colorBuffer, d_depthBuffer,
            frameWidth, frameHeight,
            t.x0, t.y0,  t.x1, t.y1,  t.x2, t.y2,
            t.z0, t.z1,  t.z2,
            t.u0, t.v0,  t.u1, t.v1,  t.u2, t.v2,
            white);
    }

    /* --- copy back -------------------------------------------------- */
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));

    size_t colorBytes = size_t(frameWidth) * frameHeight * sizeof(uint32_t);
    size_t depthBytes = size_t(frameWidth) * frameHeight * sizeof(float);

    cudaMemcpy(hostPix,   d_colorBuffer, colorBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDepth, d_depthBuffer, depthBytes, cudaMemcpyDeviceToHost);
}

/* ====================================================================
   Cleanup                                                             */
extern "C" void cleanupCuda()
{
    if (d_colorBuffer) { cudaFree(d_colorBuffer); d_colorBuffer = nullptr; }
    if (d_depthBuffer) { cudaFree(d_depthBuffer); d_depthBuffer = nullptr; }
    setTexturing(false); 
}
