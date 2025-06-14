# IzerRaster

IzerRaster is a hybrid CPU/GPU rasterizer for 3D models and 2D shapes, supporting interactive input and high-performance CUDA-accelerated rendering. It projects 3D models onto a 2D surface (the computer screen) and provides a flexible API for both drawing and input handling.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Setup](#setup)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [API Reference](#api-reference)
- [CUDA Rasterization](#cuda-rasterization)
- [Input Handling](#input-handling)
- [Error Handling](#error-handling)
- [Extensibility](#extensibility)
- [Project Management](#project-management)
- [User Stories](#user-stories)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **3D Model Loading:** Supports OBJ and STL formats, including vertex normals and face indices.
- **2D Shape Rendering:** Draw and fill primitives (points, lines, rectangles, triangles, circles).
- **Hybrid Rendering Pipeline:** CPU for 2D and overlays, GPU (CUDA) for triangle rasterization.
- **Depth Buffering:** Z-buffer for correct hidden surface removal.
- **Multiple Rendering Modes:** Wireframe, flat-shaded, textured (planned), overlays.
- **Interactive Input:** Real-time mouse and keyboard event handling.
- **Extensible Event System:** Register custom input handlers for advanced interactivity.
- **Modular Architecture:** Easily extend with new features (texture mapping, shaders, etc.).
- **Python Bindings:** Expose core API to Python via Pybind11 for rapid prototyping and scripting.
- **Cross-Platform:** Linux and Windows (with Visual Studio 2022 Build Tools).
- **Performance:** CUDA acceleration for high triangle counts and resolutions.

---

## Architecture Overview

- **Core:** C++ engine for model loading, transformation, and rasterization.
- **GPU Module:** CUDA kernels for triangle rasterization and depth testing.
- **Python Bindings:** Pybind11 exposes C++/CUDA API to Python.
- **Frontend:** Python scripts (e.g., `main.py`) for application logic, event loop, and UI.
- **Input System:** SDL3-based, supports polling and event-driven input.
- **Rendering Loop:** CPU prepares frame, GPU rasterizes, CPU overlays and presents.

---

## Setup

### Dependencies

- [GLM](https://github.com/g-truc/glm/releases): Header-only C++ math library (vector/matrix math)
- [SDL3](https://github.com/libsdl-org/SDL/releases/tag/release-3.2.12): Graphics, input, and audio
- [SDL3_ttf](https://github.com/libsdl-org/SDL_ttf/releases): TrueType font support
- [Pybind11](https://github.com/pybind/pybind11): Python bindings (install via pip/pipx)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): For GPU acceleration (Linux/Windows, NVIDIA GPU required)
- **Windows only:** Visual Studio 2022 Build Tools (.NET and C++)

*Optional:* GLEW (OpenGL extension wrangler), Ninja (faster CMake builds)

### Recommended Environment

- Linux (tested on Ubuntu 22.04+)
- Python 3.8+
- NVIDIA GPU with CUDA 11.0+ support

---

## Build Instructions

1. **Install all dependencies** (see above).
2. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/IzerRaster.git
   cd IzerRaster
   ```
3. **Configure and build with CMake:**
   ```sh
   cmake -S . -B build
   cmake --build build --config Release
   ```
   - On Windows, use the "x64 Native Tools Command Prompt for VS 2022".
   - On Linux, ensure CUDA and SDL3 development headers are installed.

4. **Copy the generated Python extension:**
   - The `.so` (Linux) or `.pyd` (Windows) file will be in `build/`.
   - Copy it to the `src/` directory.

5. **Run the application:**
   ```sh
   cd src
   python3 main.py
   ```

---

## Usage

### Loading and Rendering Models

- **Load OBJ:**  
  `mesh = loadObj("path/to/model.obj")`
- **Load STL:**  
  `mesh = loadStl("path/to/model.stl")`
- **Apply Transformation:**  
  `applyRenderMatrix(matrix, mesh)`
- **Render Mesh:**  
  `drawObj(mesh)`

### Drawing 2D Primitives

- `drawPoint(x, y, color)`
- `drawLine(x1, y1, x2, y2, color)`
- `drawRect(x, y, w, h, color)`
- `drawTriangle(x1, y1, x2, y2, x3, y3, color)`
- `drawCircle(x, y, r, color)`
- `fillRect(...)`, `fillTriangle(...)`, `fillCircle(...)`

### Input Handling

- **Polling:**  
  `events = poolInputEvents()`
- **Event Detection:**  
  `event = detectInputEvent()`

**Supported Event Types:**  
`KEYUP`, `KEYDOWN`, `MOUSEUP`, `MOUSEDOWN`, `MOUSEWHEEL`, `MOUSEMOTION`

**Key Aliases:**  
Standard SDL key names (e.g., `LEFT`, `RIGHT`, `ESCAPE`, `SPACE`, `A`, `B`, ...)

---

## API Reference

### Model Functions

- `loadObj(path: str) -> Mesh`
- `loadStl(path: str) -> Mesh`
- `applyRenderMatrix(matrix: np.ndarray, mesh: Mesh)`
- `drawObj(mesh: Mesh)`

### Drawing Functions

- `drawPoint(x, y, color)`
- `drawLine(x1, y1, x2, y2, color)`
- `drawRect(x, y, w, h, color)`
- `drawTriangle(x1, y1, x2, y2, x3, y3, color)`
- `drawCircle(x, y, r, color)`
- `fillRect(...)`, `fillTriangle(...)`, `fillCircle(...)`

### CUDA Functions

- `initCuda(width: int, height: int)`
- `renderFrame(tris, numTris, hostPix, hostDepth)`
- `cleanupCuda()`

### Input Functions

- `poolInputEvents() -> List[Event]`
- `detectInputEvent() -> Optional[Event]`

---

## CUDA Rasterization

- **initCuda(width, height):**  
  Allocates GPU buffers for color and depth, configures optimal thread blocks.
- **renderFrame(tris, numTris, hostPix, hostDepth):**  
  Runs the full GPU pipeline: clears buffers, rasterizes triangles with depth testing, copies results to CPU.
- **cleanupCuda():**  
  Frees all GPU memory and resets device pointers.

**Performance:**  
- Scales with triangle count and resolution.
- Uses one thread per pixel for maximum parallelism.
- CPU prepares triangle data, GPU rasterizes, CPU overlays wireframes and handles input.

---

## Input Handling

- **Event Types:**  
  - Keyboard: `KEYUP`, `KEYDOWN`
  - Mouse: `MOUSEUP`, `MOUSEDOWN`, `MOUSEWHEEL`, `MOUSEMOTION`
- **Custom Handlers:**  
  Register Python callbacks for specific events.
- **Example:**
  ```python
  def on_key(event):
      if event.key == 'ESCAPE':
          quit()
  registerInputHandler('KEYDOWN', on_key)
  ```

---

## Error Handling

- All CUDA calls are checked for errors; exceptions are raised on failure.
- Graceful fallback to CPU rendering if GPU memory allocation fails.
- Input and file loading errors are reported with descriptive messages.

---

## Extensibility

- **Planned Features:**  
  - Texture mapping (UV support)
  - Per-vertex color and lighting
  - Custom shaders (GLSL/CUDA)
  - Anti-aliasing and post-processing
  - Scene graph and camera controls
- **Modular Design:**  
  - Add new rendering modes or input devices with minimal changes.
  - Extend Python API via Pybind11.

---

## Project Management

- [Trello board](https://trello.com/b/ZR0p4Yfg/rasterizer) for backlog and task tracking.
- Issues and feature requests: [GitHub Issues](https://github.com/yourusername/IzerRaster/issues)

---

## User Stories

- User story 1: As a Blender enthusiast, I want to import 3D models in common file formats so I can load and visualise my designs without hassle.
- User story 2: As a designer, I want to set the output resolution myself so I can ensure my images meet quality standards for client work.
- User story 3: As an artist, I want to have lighting on the models so my rasterized scenes look dynamic and eye-catching.
- User story 4: As a 3D software developer, I want to switch between CPU and GPU rendering modes to free my GPU VRAM.
- User story 5: As a 3D enthusiast, I want to have a simple command for different drawing shapes and have them displayed in a buffer for convenience.
- User story 6: As a shader developer, I want to apply custom textures and effects to my models.
- User story 7: As a virtual photographer, I want to adjust the camera’s position, angle, and zoom to frame my 3D scenes perfectly for renders.
- User story 8: As a detail-focused tester, I want anti-aliasing enabled to eliminate jagged edges and ensure my final images look polished.
- User story 9: As a freelancer with tight deadlines, I want a back face culling option before committing to a full render.
- User story 10: As a game developer, I want GPU acceleration support to speed up rendering for complex models and large projects.
- User story 11: As a newbie to 3D software, I want an intuitive interface with helpful tooltips so I can learn the app without getting overwhelmed.

---

## Troubleshooting

- **Build fails:**  
  - Ensure all dependencies are installed and on your PATH.
  - For CUDA errors, check your GPU driver and CUDA toolkit version.
- **No display/output:**  
  - Verify SDL3 is installed and working.
  - Run with `python3 -m pdb main.py` for debugging.
- **Python import errors:**  
  - Ensure the `.so`/`.pyd` file is in the `src/` directory.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**For more details, see the in-file documentation and comments.**
