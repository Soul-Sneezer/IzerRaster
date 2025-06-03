# CMakeLists.txt

# Minimum CMake version required (adjust if needed, pybind11 might need newer)
cmake_minimum_required(VERSION 3.15)

# Project Name (should match the output module base name)
project(IzerRaster LANGUAGES CXX)

# Set C++ standard (C++17 as requested)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard C++

# Set binary output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Find Dependencies ---

# 1. Find Python (Interpreter and Development Libraries)
#    REQUIRED ensures CMake stops with an error if not found.
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Found Python Development Libraries: ${Python_LIBRARIES}")
message(STATUS "Found Python Include Dirs: ${Python_INCLUDE_DIRS}")

# 2. Find pybind11
#    Option B: Fetch pybind11 using FetchContent (often more reliable, downloads if needed)
#              Requires CMake 3.11+ (already satisfied by 3.15 minimum)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0 # Or choose the latest stable tag
)
FetchContent_MakeAvailable(pybind11)
message(STATUS "Using pybind11 (FetchContent): ${pybind11_INCLUDE_DIRS}")

# 3. Find SDL3 and SDL3_ttf
find_package(SDL3 REQUIRED)
message(STATUS "Found SDL3 Library: ${SDL3_LIBRARIES} (Using Target: SDL3::SDL3)")
message(STATUS "Found SDL3 Include Dirs: ${SDL3_INCLUDE_DIRS}")

# --- Define the Python Module Target ---

# Use pybind11's helper function to create the Python extension module.
pybind11_add_module(IzerRaster SHARED
    src/renderer2D.cpp
    src/bindings.cpp
)

# --- Link Dependencies to the Module ---

target_link_libraries(IzerRaster PRIVATE
    pybind11::module # Link pybind11 core module support
    SDL3::SDL3       # Link SDL3 library (target name from SDL3's CMake config)
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
endif()

# Build type configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as default.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# Get Python site-packages directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Installation rules
install(TARGETS IzerRaster
    LIBRARY DESTINATION "bin"
    RUNTIME DESTINATION "bin"
    ARCHIVE DESTINATION "bin"
)

# For development/testing: Also install to Python site-packages
install(TARGETS IzerRaster
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
    RUNTIME DESTINATION "${PYTHON_SITE_PACKAGES}"
)

# On Windows, install required runtime DLLs
if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:IzerRaster>
            DESTINATION "bin")
endif()

message(STATUS "Configuration complete. To build, run: cmake --build . --config <BuildType>")
message(STATUS "Replace <BuildType> with Debug, Release, RelWithDebInfo, or MinSizeRel.")
