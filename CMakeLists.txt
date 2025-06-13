cmake_minimum_required(VERSION 3.15)

# Add option for CUDA support (default to OFF for CI/CD compatibility)
option(ENABLE_CUDA "Enable CUDA support" OFF)

# Project Name (should match the output module base name)

if(ENABLE_CUDA)
    project(IzerRaster LANGUAGES CXX CUDA)
else()
    project(IzerRaster LANGUAGES CXX)
endif()

# Set C++ standard (C++17 as requested)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard C++
set(CMAKE_CUDA_ARCHITECTURES 86)

# Only set CUDA architecture if CUDA is enabled
if(ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Set binary output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Find Dependencies ---

# Only find CUDA if enabled
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# 1. Find Python (Interpreter and Development Libraries)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Found Python Development Libraries: ${Python_LIBRARIES}")
message(STATUS "Found Python Include Dirs: ${Python_INCLUDE_DIRS}")

# 2. Find pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)
message(STATUS "Using pybind11 (FetchContent): ${pybind11_INCLUDE_DIRS}")

# 3. Find SDL3
find_package(SDL3 REQUIRED)
message(STATUS "Found SDL3 Library: ${SDL3_LIBRARIES} (Using Target: SDL3::SDL3)")
message(STATUS "Found SDL3 Include Dirs: ${SDL3_INCLUDE_DIRS}")

# --- Define source files ---
set(COMMON_SOURCES
    src/texture.cpp
    src/renderer2D.cpp
    src/bindings.cpp
)
# --- stb_image ca INTERFACE ---
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(IzerRaster PRIVATE stb_image)

# Conditionally add CUDA sources
if(ENABLE_CUDA)
    set(SOURCES ${COMMON_SOURCES} src/render.cu)
else()
    set(SOURCES ${COMMON_SOURCES})
endif()

# --- Define the Python Module Target ---
pybind11_add_module(IzerRaster SHARED ${SOURCES})

# --- stb_image as INTERFACE ---
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(IzerRaster PRIVATE stb_image)

# --- Link Dependencies to the Module ---
# Only set CUDA properties if enabled
if(ENABLE_CUDA)
    set_target_properties(IzerRaster PROPERTIES
        CUDA_ARCHITECTURES "86"
    )
endif()

# Base dependencies
set(DEPENDENCIES
    pybind11::module
    SDL3::SDL3
)

# Add CUDA dependencies if enabled
if(ENABLE_CUDA)
    list(APPEND DEPENDENCIES CUDA::cudart)
endif()

target_link_libraries(IzerRaster PRIVATE ${DEPENDENCIES})

# Explicitly add include directory for local headers
target_include_directories(IzerRaster PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# First try find_package for GLM
find_package(GLM QUIET)

# If find_package fails, try find_path as fallback
if(NOT TARGET GLM::GLM)
    find_path(GLM_INCLUDE_DIR glm/glm.hpp
        PATHS 
            ${CMAKE_PREFIX_PATH}/include
            ${GLM_DIR}/include
            ${GLM_ROOT_DIR}/include
            $ENV{GLM_DIR}/include
            $ENV{GLM_ROOT_DIR}/include
            $ENV{VCPKG_ROOT}/installed/x64-windows/include
    )
    if(NOT GLM_INCLUDE_DIR)
        message(FATAL_ERROR "GLM not found! Neither find_package nor find_path could locate GLM. Searched in: \n"
            "  ${CMAKE_PREFIX_PATH}/include\n"
            "  ${GLM_DIR}/include\n"
            "  ${GLM_ROOT_DIR}/include\n"
            "  $ENV{GLM_DIR}/include\n"
            "  $ENV{GLM_ROOT_DIR}/include\n"
            "  $ENV{VCPKG_ROOT}/installed/x64-windows/include")
    endif()

    # Create GLM::GLM target
    add_library(GLM::GLM INTERFACE IMPORTED)
    set_target_properties(GLM::GLM PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}"
    )
endif()

target_link_libraries(IzerRaster PRIVATE GLM::GLM)

# --- Configure Build Options ---

# Add common warning flags
if(MSVC)
    target_compile_options(IzerRaster PRIVATE /W3)
else()
    target_compile_options(IzerRaster PRIVATE -Wall -Wextra -g)
endif()

# Build type configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as default.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# --- Installation Rules ---

if(DEFINED ENV{CI})
    # In CI environment, only install to local artifacts directory
    install(TARGETS IzerRaster
        LIBRARY DESTINATION "bin"
        RUNTIME DESTINATION "bin"
        ARCHIVE DESTINATION "bin"
    )
else()
    # For local development, install both to artifacts and Python site-packages
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    install(TARGETS IzerRaster
        LIBRARY DESTINATION "bin"
        RUNTIME DESTINATION "bin"
        ARCHIVE DESTINATION "bin"
    )

    # Only install to Python site-packages during local development
    install(TARGETS IzerRaster
        LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
        RUNTIME DESTINATION "${PYTHON_SITE_PACKAGES}"
    )
endif()

# On Windows, install required runtime DLLs
if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:IzerRaster>
            DESTINATION "bin")
endif()

# Add CUDA status message
if(ENABLE_CUDA)
    message(STATUS "CUDA support is enabled")
else()
    message(STATUS "CUDA support is disabled")
endif()

message(STATUS "Configuration complete. To build, run: cmake --build . --config <BuildType>")
message(STATUS "Replace <BuildType> with Debug, Release, RelWithDebInfo, or MinSizeRel.")
