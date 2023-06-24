cmake_minimum_required(VERSION 3.14)

project(flower)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
find_package(Vulkan REQUIRED)

# Cpp version set.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Add dependency library.
add_subdirectory("dependency/spdlog")
add_subdirectory("dependency/vma")
add_subdirectory("dependency/stduuid")
add_subdirectory("dependency/json")
add_subdirectory("dependency/inipp") 
add_subdirectory("dependency/utfcpp")
add_subdirectory("dependency/nativefiledialogs")
add_subdirectory("dependency/entt")
add_subdirectory("dependency/tinyexr")
add_subdirectory("dependency/lz4/build/cmake")  

SET(USE_MSVC_RUNTIME_LIBRARY_DLL ON CACHE BOOL "Use MSVC Runtime Library DLL (/MD or /MDd)" FORCE)
add_subdirectory("dependency/bullet3")  

SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
SET(ASSIMP_BUILD_TESTS OFF CACHE BOOL "Assimp build tests" FORCE)
SET(ASSIMP_INSTALL OFF CACHE BOOL "Assimp install" FORCE) 
SET(ASSIMP_INSTALL_PDB OFF CACHE BOOL "Assimp install PDB" FORCE)
add_subdirectory("dependency/assimp")

SET(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW build document" FORCE) 
SET(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW build tests" FORCE) 
SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW build examples" FORCE)  
add_subdirectory("dependency/glfw")  
 
SET(GFX_API_DX12 OFF CACHE BOOL "Build with DX12" FORCE)
SET(GFX_API_VK ON CACHE BOOL "Build with Vulkan" FORCE)  
add_subdirectory("dependency/fsr2") 

SET(LIBTYPE "STATIC")
SET(ALSOFT_EXAMPLES  OFF CACHE BOOL "Build example programs"  FORCE)
add_subdirectory("dependency/openal")
 
# Prepare imgui path.
set(CEREAL_DIR ${CMAKE_SOURCE_DIR}/dependency/cereal/include)

# Include path config.
include_directories("dependency")  
include_directories("dependency/assimp/include") 
include_directories("source/engine") 
 
# Applications.  
add_subdirectory("source")    