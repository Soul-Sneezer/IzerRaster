# CMakeLists.txt

# Minimum CMake version required (adjust if needed, pybind11 might need newer)
cmake_minimum_required(VERSION 3.15)

# Project Name (should match the output module base name)
project(IzerRaster LANGUAGES CXX CUDA)

# Set C++ standard (C++17 as requested)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard C++
set(CMAKE_CUDA_ARCHITECTURES 86)

# --- Find Dependencies ---

find_package(CUDAToolkit REQUIRED)  


# 1. Find Python (Interpreter and Development Libraries)
#    REQUIRED ensures CMake stops with an error if not found.
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Found Python Development Libraries: ${Python_LIBRARIES}")
message(STATUS "Found Python Include Dirs: ${Python_INCLUDE_DIRS}")

# 2. Find pybind11
#    Option A: Use find_package (requires pybind11 to be installed where CMake can find it,
#              e.g., via pip install pybind11 and setting CMAKE_PREFIX_PATH, or system install)
# find_package(pybind11 REQUIRED)
# message(STATUS "Found pybind11 (find_package): ${pybind11_INCLUDE_DIRS}")

#    Option B: Fetch pybind11 using FetchContent (often more reliable, downloads if needed)
#              Requires CMake 3.11+ (already satisfied by 3.15 minimum)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0 # Or choose the latest stable tag
)
# You might need policy CMP0135 NEW if using CMake 3.24+ with FetchContent_MakeAvailable
# if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
#   cmake_policy(SET CMP0135 NEW)
# endif()
FetchContent_MakeAvailable(pybind11)
message(STATUS "Using pybind11 (FetchContent): ${pybind11_INCLUDE_DIRS}")

# 3. Find SDL3 and SDL3_ttf
#    This relies on SDL3 and SDL3_ttf being installed in a way CMake can find them.
#    Users might need to set the CMAKE_PREFIX_PATH environment variable or CMake variable
#    to point to the SDL installation directories (e.g., the root dir containing include/, lib/).
#    Example: cmake .. -DCMAKE_PREFIX_PATH="/path/to/sdl3_install;/path/to/sdl3_ttf_install"
find_package(SDL3 REQUIRED)
find_package(SDL3_ttf REQUIRED)
message(STATUS "Found SDL3 Library: ${SDL3_LIBRARIES} (Using Target: SDL3::SDL3)")
message(STATUS "Found SDL3 Include Dirs: ${SDL3_INCLUDE_DIRS}")
message(STATUS "Found SDL3_ttf Library: ${SDL3_TTF_LIBRARIES} (Using Target: SDL3_ttf::SDL3_ttf)")
message(STATUS "Found SDL3_ttf Include Dirs: ${SDL3_TTF_INCLUDE_DIRS}")
# --- Define the Python Module Target ---

# Use pybind11's helper function to create the Python extension module.
# It automatically handles the correct naming conventions (.so/.pyd) and links Python.
# The first argument "IzerRaster" MUST match the name in PYBIND11_MODULE(IzerRaster, m)
pybind11_add_module(IzerRaster SHARED
    src/texture.cpp   
    src/renderer2D.cpp
    src/bindings.cpp
    src/render.cu
)
# --- stb_image ca INTERFACE ---
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(IzerRaster PRIVATE stb_image)

# --- Link Dependencies to the Module ---

set_target_properties(IzerRaster PROPERTIES
    CUDA_ARCHITECTURES "86"
)

# Link the necessary libraries. pybind11_add_module handles Python itself.
# We use the modern CMake targets provided by SDL3 and pybind11 if available.
# These targets usually propagate include directories automatically.
target_link_libraries(IzerRaster PRIVATE
    pybind11::module # Link pybind11 core module support
    SDL3::SDL3       # Link SDL3 library (target name from SDL3's CMake config)
    SDL3_ttf::SDL3_ttf # Link SDL3_ttf library (target name from SDL3_ttf's CMake config)
    CUDA::cudart
)

# Explicitly add include directory for local headers like "renderer2D.hpp"
target_include_directories(IzerRaster PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} # Directory containing this CMakeLists.txt
)

find_path(GLM_INCLUDE_DIR glm/glm.hpp)
if(NOT GLM_INCLUDE_DIR)
  message(FATAL_ERROR "GLM not found!")
endif()

target_include_directories(IzerRaster PRIVATE ${GLM_INCLUDE_DIR})


# --- Configure Build Options ---

# Add common warning flags (adjust as needed)
if(MSVC)
  # Microsoft Visual C++ specific flags
  target_compile_options(IzerRaster PRIVATE /W3) # Warning level 3
else()
  # GCC/Clang specific flags (-Wall corresponds to the original command)
  target_compile_options(IzerRaster PRIVATE -Wall -Wextra -g)
  # -fPIC is generally handled automatically by CMake for SHARED libraries on relevant platforms
endif()

# Build type configuration (Debug, Release, etc.)
# The original command used -O3, which corresponds to Release optimization.
# CMake handles this via build types. Setting a default if none is specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as default.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# --- Optional: Installation ---
# You can add installation rules here if you want to install the module system-wide
# or into a specific location using `cmake --install`.
# Example (adjust paths as needed):
# include(GNUInstallDirs)
# install(TARGETS IzerRaster
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages
#         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # For DLLs on Windows
#         )

message(STATUS "Configuration complete. To build, run: cmake --build . --config <BuildType>")
message(STATUS "Replace <BuildType> with Debug, Release, RelWithDebInfo, or MinSizeRel.")
