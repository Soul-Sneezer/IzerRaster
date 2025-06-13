# IzerRaster
IzerRaster is a rasterizer. What is a rasterizer? A rasterizer takes 3D models and projects them to a 2D surface, that 2D surface being the computer screen.
https://www.canva.com/design/DAGmfsr3YA4/k_NagS48sFp2AQYWm-HcGw/view?utm_content=DAGmfsr3YA4&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h4b41e2cd2e


Currently the library has two main features: 
- it can load and render 3D models and basic 2D shapes(triangles, rectangles, circles)
- it can handle input events from the mouse or keyboard

## Setup
This project requires the following libraries:

- GLM: A header-only C++ mathematics library for graphics software.
- SDL3: The Simple DirectMedia Layer library for handling graphics, input, and audio.
- SDL3_ttf: An SDL extension for handling TrueType fonts.

Make sure to install these dependencies before building the project:
 - SDL3 : https://github.com/libsdl-org/SDL/releases/tag/release-3.2.12
 - GLM : https://github.com/g-truc/glm/releases
 - SDL3_ttf : https://github.com/libsdl-org/SDL_ttf/releases
 - Pybind11 : make sure to have python installed after that pip/pipx install pybind11

Windows only:
- Build Tools for Visual Studio 2022 (.Net and C++ development tools): https://visualstudio.microsoft.com/downloads/?q=build+tools 
 
Optional dependencies: GLEW, NINJA.

## Building and running

After installing the project requirements (look above if not), build the **CMakeLists.txt** from the project and after the build folder successfully creates use: `cmake --build. --Config <BuildType>`. This command will create a .Pyd file that must be copied in `/src`. After this step the application is ready to run. To load your objects add the *path* in the ``loadObj`` function and run `main.py`. Use **W** to zoom out if you are too close to the object and **Esc** to close the window.

## Documentation

The IzerRaster library has a core loop, named UserUpdate(), that runs once per frame. And a UserInit() function, which runs right after the basic Init() function that initializes everything. The user should override the "UserUpdate", "UserInit" function when defining the renderer class.

The drawing/rendering and input handling functions should be called from UserUpdate.

In order to render an object you need to:
- Load the object, using the _loadObj_ function. It takes a path as argument, and can only load .obj files. It then returns a _mesh_ object
- You can then modify the _mesh_ object using the _applyRenderMatrix_ function, which takes 2 arguments: a render matrix and a mesh
- Then you render the object using the _drawObj_ and passing the mesh as an argument
  
**IzerRaster** also has a few drawing functions:
- *drawPoint* - takes 3 arguments, the first is the X coordinate, the second is the Y coordinate and the last one is the color in RGBA format
- *drawLine* - takes 5 arguments, the first two are the X,Y coordinates of the first point of the line, then the X,Y coordinates of the second point of the line, and the color
- *drawRect* - takes 5 arguments, the first two are the X,Y coordinates of the lower left corner point, the following two are the X,Y coordinates of the upper right corner point, and then the color
- *drawTriangle* - takes 7 arguments, the first six are coordinates of the points that describe the triangle: X1,Y1, X2,Y2, X3,Y3, the last one is the color 
- *drawCircle* - takes 4 arguments, the first two are the X,Y coordinates of the center of the triangle, then the radius and lastly, the color

For almost every drawing function(with the exception of *drawPoint* and *drawLine*) there is an equivalent _fill_ function that takes the same arguments. As the name may say, a _fill_ function fills the shape, while the _draw_ functions draws the outline. The fill functions are the following: _fillRect_, _fillTriangle_, _fillCircle_.

Input handling is done in the following way:
- You call _poolInputEvents_ or _detectInputEvent_. The difference between them is that _poolInputEvents_ returns a list of all the events detected in a frame, while _detectInputEvent_ returns the last event detected in the frame
- Each input event has four fields: its type, the key that was pressed, and if it was a mouse event, the current coordinates of the mouse pointer. The user can then use this information to handle each event.

The event types are the following:
- *KEYUP* - detects if the user stopped pressing a key
- *KEYDOWN* - detects if the user pressed a key
- *MOUSEUP* - detects if the user stopped pressing a mouse key
- *MOUSEDOWN* - detects if the user pressed a mouse key
- *MOUSEWHEEL* - detects if the user moved the mouse wheel
- *MOUSEMOTION* - detects if the user moved the mouse

Most keyboard and mouse keys are supported and their values are marked with the following aliases:
- KEY_UNKNOWN
- KEY_RETURN
- KEY_ESCAPE
- KEY_BACKSPACE
- KEY_TAB
- KEY_SPACE
- KEY_EXCLAIM
- KEY_DBLAPOSTROPHE
- KEY_HASH
- KEY_DOLLAR
- KEY_PERCENT
- KEY_AMPERSAND
- KEY_APOSTROPHE
- KEY_LEFTPAREN
- KEY_RIGHTPAREN
- KEY_ASTERISK
- KEY_PLUS
- KEY_COMMA
- KEY_MINUS
- KEY_PERIOD
- KEY_SLASH
- KEY_LEFTBRACE
- KEY_PIPE
- KEY_RIGHTBRACE
- KEY_TILDE
- KEY_DELETE
- KEY_PLUSMINUS
- KEY_CAPSLOCK
- KEY_UP
- KEY_DOWN
- KEY_LEFT
- KEY_RIGHT
- KEY_1
- KEY_2
- KEY_3
- KEY_4
- KEY_5
- KEY_6
- KEY_7
- KEY_8
- KEY_9
- KEY_0
- KEY_A
- KEY_B
- KEY_C
- KEY_D
- KEY_E
- KEY_F
- KEY_G
- KEY_H
- KEY_I
- KEY_J
- KEY_K
- KEY_L
- KEY_M
- KEY_N
- KEY_O
- KEY_P
- KEY_Q
- KEY_R
- KEY_S
- KEY_T
- KEY_U
- KEY_V
- KEY_W
- KEY_X
- KEY_Y
- KEY_Z

# CUDA Rasterization Implementation

IzerRaster's CUDA implementation provides GPU-accelerated triangle rasterization with depth buffering for high-performance 3D rendering. This documentation covers the GPU-side rendering pipeline that complements the CPU-based 2D drawing functions.

## Architecture Overview

The CUDA rasterizer operates on a **one thread per pixel** architecture, where each CUDA thread is responsible for determining if a pixel lies within a triangle and handling depth testing. This approach maximizes GPU parallelism by utilizing thousands of cores simultaneously.

### Core Components

- **GPU Memory Management**: Dedicated CUDA buffers for color (ARGB32) and depth (float) data
- **Triangle Rasterization**: Barycentric coordinate-based inside/outside testing
- **Depth Testing**: Per-pixel hidden surface removal using Z-buffering
- **Optimized Thread Configuration**: Automatic block size optimization based on GPU capabilities

## CUDA Data Structures

### CudaTri Structure
```cpp
struct CudaTri {
    float x0, y0;     // Screen-space coordinates for vertex 0 (pixels)
    float x1, y1;     // Screen-space coordinates for vertex 1 (pixels) 
    float x2, y2;     // Screen-space coordinates for vertex 2 (pixels)
    float z0, z1, z2; // Depth values for each vertex (post-projection)
};
```

This structure represents a triangle ready for GPU rasterization, with all coordinates already transformed from 3D world space to 2D screen space.

## GPU Kernels

### clearBuffers Kernel
Initializes the framebuffer for each frame by setting every pixel to a default color and maximum depth value.

**Execution Model**: Each thread handles one pixel
**Purpose**: Prepare clean buffers before triangle rasterization
**Performance**: Highly parallel, memory-bandwidth limited

### rasterizeTriDepth Kernel
The core rasterization kernel that determines triangle coverage and performs depth testing.

**Algorithm**:
1. Calculate barycentric coordinates for the current pixel
2. Test if pixel lies inside triangle (all barycentric weights ≥ 0)
3. Interpolate depth value using barycentric coordinates
4. Perform depth test against existing Z-buffer value
5. Update color and depth buffers if pixel passes depth test

**Mathematical Foundation**:
For a triangle with vertices A, B, C and a point P, barycentric coordinates (w0, w1, w2) satisfy:
- P = w0×A + w1×B + w2×C
- w0 + w1 + w2 = 1
- If all weights ≥ 0, point P lies inside the triangle

## API Functions

### initCuda(int width, int height)
Initializes the CUDA rendering system by allocating GPU memory and configuring optimal thread block dimensions.

**Memory Allocation**:
- Color buffer: `width × height × sizeof(uint32_t)` bytes
- Depth buffer: `width × height × sizeof(float)` bytes

**Thread Configuration**:
- Automatically determines optimal block size based on GPU capabilities
- Uses square thread blocks (typically 16×16 or 32×32)
- Calculates grid dimensions to cover entire framebuffer

**Returns**: `true` on success, `false` if GPU memory allocation fails

### renderFrame(const CudaTri* tris, int numTris, uint32_t* hostPix, float* hostDepth)
Executes the complete GPU rendering pipeline for a frame.

**Rendering Pipeline**:
1. **Clear Phase**: Reset color buffer to black, depth buffer to maximum distance
2. **Rasterization Phase**: Process each triangle with depth testing
3. **Transfer Phase**: Copy results from GPU memory back to CPU

**Performance Characteristics**:
- Scales linearly with triangle count
- Each triangle launches a full-screen kernel
- Memory bandwidth typically becomes the bottleneck for high triangle counts

### cleanupCuda()
Releases all GPU memory allocations and resets the system to an uninitialized state.

**Resource Management**:
- Frees color and depth buffers
- Sets device pointers to nullptr
- Safe to call multiple times

## Performance Considerations

### Thread Block Optimization
The system automatically selects thread block dimensions based on:
- GPU's maximum threads per block capability
- Square block layout for optimal memory coalescing
- Power-of-two dimensions for warp efficiency

### Memory Access Patterns
- **Coalesced Access**: Thread blocks are arranged to ensure adjacent threads access adjacent memory locations
- **Bank Conflicts**: Avoided through careful thread indexing schemes
- **Occupancy**: Thread block size chosen to maximize GPU occupancy

### Scalability
The rasterizer performance scales with:
- **Resolution**: O(width × height) for buffer operations
- **Triangle Count**: O(numTriangles) for rasterization
- **GPU Cores**: Near-linear scaling with available CUDA cores

## Integration with CPU Pipeline

The CUDA rasterizer operates as part of a hybrid CPU/GPU rendering system:

1. **CPU Tasks**:
   - 3D model loading and transformation
   - Back-face culling
   - Triangle sorting for transparency
   - Wireframe overlay rendering

2. **GPU Tasks**:
   - Solid triangle fill with depth testing
   - High-throughput parallel pixel processing

3. **Data Flow**:
   - CPU prepares triangle data in screen space
   - GPU rasterizes triangles into framebuffer
   - CPU reads back results for display and overlay rendering

## Error Handling

The CUDA implementation includes comprehensive error checking:
- **Allocation Failures**: Detected during initialization
- **Kernel Launch Errors**: Checked after each kernel call
- **Memory Transfer Errors**: Validated during host/device copies
- **Graceful Degradation**: Continues rendering even after non-fatal errors

## Future Extension Points

The current architecture supports future enhancements:
- **Texture Mapping**: CudaTri structure can be extended with UV coordinates
- **Per-Vertex Colors**: Additional color attributes for gradient fills
- **Multi-pass Rendering**: Support for transparency and effects
- **Compute Shaders**: Integration with custom CUDA kernels for advanced effects

This CUDA implementation provides the high-performance foundation for IzerRaster's 3D rendering capabilities, handling the computationally intensive triangle rasterization while maintaining the flexibility for CPU-based post-processing and overlay effects.

# Trello

We used Trello in order to manage and organise our tasks during the development of the application. This is the [link](https://trello.com/b/ZR0p4Yfg/rasterizer) to our backlog creation.

# User stories

User stories are short, simple descriptions of a feature told from the perspective of the person who desires the new capability, usually a user or customer of the system.

User story 1: As a Blender enthusiast, I want to import 3D models in common file formats so I can load and convert my designs without hassle.

User story 2: As a designer, I want to set the output resolution myself so I can ensure my images meet quality standards for client work.

User story 3: As a lighting artist, I want to tweak lighting settings so my rasterized scenes look dynamic and eye-catching.

User story 4: As a 3D animator, I want to switch between rendering modes to experiment with different visual styles for my portfolio.

User story 5: As a 3D enthusiast, I want to have simple command for different drawing shapes and have them displayed in a buffer for convenience.

User story 6: As a shader developer, I want to apply custom GLSL/HLSL shaders to add unique textures and effects to my models.

User story 7: As a virtual photographer, I want to adjust the camera’s position, angle, and zoom to frame my 3D scenes perfectly for renders.

User story 8: As a detail-focused tester, I want anti-aliasing enabled to eliminate jagged edges and ensure my final images look polished.

User story 9: As a freelancer with tight deadlines, I want a back face culling option before committing to a full render.

User story 10: As a game developer, I want GPU acceleration support to speed up rendering for complex models and large projects.

User story 11: As a newbie to 3D software, I want an intuitive interface with helpful tooltips so I can learn the app without getting overwhelmed.
