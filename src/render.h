#pragma once
#include <cstdint>

/**
 * @file render.h
 * @brief Public interface for the CUDA rasterization renderer.
 *
 * Defines the data structures and functions for initializing the renderer, 
 * rendering a frame (rasterizing triangles with depth buffering), and cleaning up.
 * Uses CUDA for rendering, SDL for window/texture (in main), and GLM for math (in main).
 */

// Structure representing a triangle with screen-space coordinates and per-vertex depth values.
// Note: This struct can be extended with additional per-vertex attributes (e.g. color, texture UVs) 
//       for more complex shading or texturing.
#pragma once
#include "renderer2D.hpp"
#include <cstdint>

// ==================  geometrie GPU ==================
struct CudaTri
{
    // poziţii 2-D în pixeli
    float x0, y0;
    float x1, y1;
    float x2, y2;
    // adâncimi
    float z0, z1, z2;
    // coordonate UV
    float u0, v0;
    float u1, v1;
    float u2, v2;
};

#ifdef __cplusplus
extern "C" {
#endif



void uploadTexture(const uint32_t* devPixels, int w, int h);
void setTexturing(bool enable);

// void uploadLighting(const Light& light, const glm::vec3& camPos, const Material& material);



/**
 * @brief Initialize the CUDA renderer.
 *
 * Allocates GPU memory for the color buffer (ARGB32 pixels) and depth buffer (float)
 * for a framebuffer of the given width and height. Must be called once at startup.
 *
 * @param width   Framebuffer width in pixels.
 * @param height  Framebuffer height in pixels.
 * @return true if initialization succeeded, false if CUDA memory allocation failed.
 */
bool initCuda(int width, int height);

/**
 * @brief Rasterize triangles and produce a rendered frame.
 *
 * Rasterizes an array of triangles (provided in screen-space coordinates with depth)
 * into the GPU color and depth buffers using CUDA. Performs depth testing per pixel.
 * After rasterization, this function copies the resulting color and depth buffers back to host memory.
 *
 * @param tris      Pointer to an array of CudaTri triangles to render.
 * @param numTris   Number of triangles in the array.
 * @param hostPix   Pointer to a host buffer (size = width*height) to receive 32-bit ARGB pixel data.
 * @param hostDepth Pointer to a host buffer (size = width*height) to receive float depth values.
 */
void renderFrame(const CudaTri* tris, int numTris, uint32_t* hostPix, float* hostDepth);

/**
 * @brief Clean up the CUDA renderer.
 *
 * Frees the GPU memory allocated for color and depth buffers. 
 * Call this once at application shutdown to release GPU resources.
 */
void cleanupCuda();

#ifdef __cplusplus
}
#endif
