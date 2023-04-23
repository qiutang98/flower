cmake_minimum_required(VERSION 3.14)

project(island)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Cpp version set.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Add external library.
add_subdirectory("external/spdlog")
add_subdirectory("external/vma")
add_subdirectory("external/glm")
add_subdirectory("external/stduuid")
add_subdirectory("external/json")
add_subdirectory("external/inipp") 
add_subdirectory("external/utfcpp")
add_subdirectory("external/nativefiledialogs")
add_subdirectory("external/entt")
add_subdirectory("external/tinyexr")

add_subdirectory("external/lz4/build/cmake")  

SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
SET(ASSIMP_BUILD_TESTS OFF CACHE BOOL "Assimp build tests" FORCE)
SET(ASSIMP_INSTALL OFF CACHE BOOL "Assimp install" FORCE) 
SET(ASSIMP_INSTALL_PDB OFF CACHE BOOL "Assimp install PDB" FORCE)
add_subdirectory("external/assimp")

SET(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW build document" FORCE) 
SET(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW build tests" FORCE) 
SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW build examples" FORCE) 
add_subdirectory("external/glfw")  
 
SET(GFX_API_DX12 OFF CACHE BOOL "Build with DX12" FORCE)
SET(GFX_API_VK ON CACHE BOOL "Build with Vulkan" FORCE)  
add_subdirectory("external/fsr2") 

# Prepare imgui path.
set(CEREAL_DIR     ${CMAKE_SOURCE_DIR}/external/cereal/include)

# Include path config.
include_directories("external")  
include_directories("external/assimp/include") 
include_directories("source/engine") 
 
# Applications.  
add_subdirectory("source")    