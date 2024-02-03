cmake_minimum_required(VERSION 3.14)

project(dark)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(LIBRARY_OUTPUT_PATH    ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})


SET(GLFW_INSTALL                   OFF CACHE BOOL "" FORCE) 
SET(GLFW_BUILD_DOCS                OFF CACHE BOOL "" FORCE) 
SET(GLFW_BUILD_TESTS               OFF CACHE BOOL "" FORCE) 
SET(GLFW_BUILD_EXAMPLES            OFF CACHE BOOL "" FORCE)  
add_subdirectory("${PROJECT_SOURCE_DIR}/external/glfw")

SET(BUILD_RTTR_DYNAMIC             OFF CACHE BOOL "" FORCE) 
SET(BUILD_UNIT_TESTS               OFF CACHE BOOL "" FORCE) 
SET(BUILD_STATIC                   ON  CACHE BOOL "" FORCE) 
SET(BUILD_WITH_STATIC_RUNTIME_LIBS OFF CACHE BOOL "" FORCE) 
SET(BUILD_WITH_RTTI                ON  CACHE BOOL "" FORCE) 
SET(BUILD_BENCHMARKS               OFF CACHE BOOL "" FORCE) 
SET(BUILD_EXAMPLES                 OFF CACHE BOOL "" FORCE) 
SET(BUILD_DOCUMENTATION            OFF CACHE BOOL "" FORCE) 
SET(BUILD_INSTALLER                OFF CACHE BOOL "" FORCE) 
SET(BUILD_PACKAGE                  OFF CACHE BOOL "" FORCE) 
SET(USE_PCH                        ON  CACHE BOOL "" FORCE) 
SET(CUSTOM_DOXYGEN_STYLE           OFF CACHE BOOL "" FORCE) 
SET(BUILD_WEBSITE_DOCU             OFF CACHE BOOL "" FORCE) 
add_subdirectory("${PROJECT_SOURCE_DIR}/external/rttr")

add_subdirectory("${PROJECT_SOURCE_DIR}/external/assimp")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/nativefiledialogs")
add_subdirectory("${PROJECT_SOURCE_DIR}/external/lz4/cmake") 

file(GLOB_RECURSE shaderHeaders CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/install/shader/*.h")

file(GLOB_RECURSE engineHeaders CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/source/engine/*.h")
file(GLOB_RECURSE engineSource CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/source/engine/*.cpp")

file(GLOB_RECURSE editorHeaders CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/source/editor/*.h")
file(GLOB_RECURSE editorSource CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/source/editor/*.cpp")

add_executable(dark ${editorSource} ${editorHeaders} ${engineSource} ${engineHeaders} ${shaderHeaders})

find_package(Vulkan REQUIRED)

source_group(TREE "${PROJECT_SOURCE_DIR}/source/engine" PREFIX "engine" FILES ${engineSource} ${engineHeaders}) 
source_group(TREE "${PROJECT_SOURCE_DIR}/source/editor" PREFIX "editor" FILES ${editorSource} ${editorHeaders}) 
source_group(TREE "${PROJECT_SOURCE_DIR}/install/shader" PREFIX "shader" FILES ${shaderHeaders}) 

target_precompile_headers(dark PRIVATE 
    "${PROJECT_SOURCE_DIR}/source/engine/utils/utils.h" 
)

# Visual studio incremental build.
if(MSVC)
    target_compile_options(dark PUBLIC "/ZI")
    target_link_options(dark PUBLIC "/INCREMENTAL")   
endif()  

# Big obj flag.
if(MSVC)
    target_compile_options(dark PRIVATE /bigobj)
else()
    target_compile_options(dark PRIVATE -Wa,-mbig-obj)
endif()  

target_link_libraries(dark PUBLIC glfw Vulkan::Vulkan nativefiledialog RTTR::Core_Lib lz4_static assimp)

target_include_directories(dark PUBLIC 
    "${PROJECT_SOURCE_DIR}/external/include" 
    "${PROJECT_SOURCE_DIR}/source" 
    "${PROJECT_SOURCE_DIR}/install/shader" 
    "${PROJECT_SOURCE_DIR}/source/engine"
    "${PROJECT_SOURCE_DIR}/external/tracy")

set_target_properties(dark PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/install") 