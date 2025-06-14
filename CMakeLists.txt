cmake_minimum_required(VERSION 3.15)

option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    project(IzerRaster LANGUAGES CXX CUDA)
else()
    project(IzerRaster LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Found Python Development Libraries: ${Python_LIBRARIES}")
message(STATUS "Found Python Include Dirs: ${Python_INCLUDE_DIRS}")

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)
message(STATUS "Using pybind11 (FetchContent): ${pybind11_INCLUDE_DIRS}")

find_package(SDL3 REQUIRED)
message(STATUS "Found SDL3 Library: ${SDL3_LIBRARIES} (Using Target: SDL3::SDL3)")
message(STATUS "Found SDL3 Include Dirs: ${SDL3_INCLUDE_DIRS}")

set(COMMON_SOURCES
    src/texture.cpp
    src/renderer2D.cpp
    src/bindings.cpp
)

if(ENABLE_CUDA)
    set(SOURCES ${COMMON_SOURCES} src/render.cu)
else()
    set(SOURCES ${COMMON_SOURCES})
endif()

pybind11_add_module(IzerRaster SHARED ${SOURCES})

# Add HAS_CUDA preprocessor definition if CUDA is enabled
if(ENABLE_CUDA)
    target_compile_definitions(IzerRaster PRIVATE HAS_CUDA=1)
endif()

add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(IzerRaster PRIVATE stb_image)

if(ENABLE_CUDA)
    set_target_properties(IzerRaster PROPERTIES
        CUDA_ARCHITECTURES "86"
    )
endif()

set(DEPENDENCIES
    pybind11::module
    SDL3::SDL3
)

if(ENABLE_CUDA)
    list(APPEND DEPENDENCIES CUDA::cudart)
endif()

target_link_libraries(IzerRaster PRIVATE ${DEPENDENCIES})

target_include_directories(IzerRaster PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(GLM QUIET)
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
        message(FATAL_ERROR "GLM not found! Neither find_package nor find_path could locate GLM.")
    endif()

    add_library(GLM::GLM INTERFACE IMPORTED)
    set_target_properties(GLM::GLM PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}"
    )
endif()

target_link_libraries(IzerRaster PRIVATE GLM::GLM)

if(MSVC)
    target_compile_options(IzerRaster PRIVATE /W3)
else()
    target_compile_options(IzerRaster PRIVATE -Wall -Wextra -g)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as default.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

if(DEFINED ENV{CI})
    install(TARGETS IzerRaster
        LIBRARY DESTINATION "bin"
        RUNTIME DESTINATION "bin"
        ARCHIVE DESTINATION "bin"
    )
else()
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

    install(TARGETS IzerRaster
        LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
        RUNTIME DESTINATION "${PYTHON_SITE_PACKAGES}"
    )
endif()

if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:IzerRaster>
            DESTINATION "bin")
endif()

if(ENABLE_CUDA)
    message(STATUS "CUDA support is enabled")
else()
    message(STATUS "CUDA support is disabled")
endif()

message(STATUS "Configuration complete. To build, run: cmake --build . --config <BuildType>")
