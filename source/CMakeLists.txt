MACRO(SOURCE_GROUP_BY_FOLDER target)
    SET(SOURCE_GROUP_DELIMITER "/")
    SET(last_dir "")
    SET(files "")
    FOREACH(file ${${target}_SRC} ${${target}_HEADERS})
        file(RELATIVE_PATH relative_file "${PROJECT_SOURCE_DIR}/source/" ${file})
        GET_FILENAME_COMPONENT(dir "${relative_file}" PATH)
        IF (NOT "${dir}" STREQUAL "${last_dir}")
        IF (files)
            SOURCE_GROUP("${last_dir}" FILES ${files})
        ENDIF (files)
        SET(files "") 
        ENDIF (NOT "${dir}" STREQUAL "${last_dir}")
        SET(files ${files} ${file})
        SET(last_dir "${dir}")
    ENDFOREACH(file)
    IF (files)
        SOURCE_GROUP("${last_dir}" FILES ${files})
    ENDIF (files)
ENDMACRO(SOURCE_GROUP_BY_FOLDER)

file(GLOB_RECURSE flower_SRC CONFIGURE_DEPENDS "*.h" "*.cpp")

add_executable(flower ${flower_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/icon.rc")

SOURCE_GROUP_BY_FOLDER(flower)

target_include_directories(flower PRIVATE 
${CEREAL_DIR} 
${PROJECT_SOURCE_DIR}/dependency/assimp/include
${BULLET_INCLUDE_DIR}
)
 
target_precompile_headers(flower PRIVATE
    "engine/asset/asset.h"
    "engine/imgui/ui.h"
    "engine/renderer/renderer.h"
    "engine/rhi/rhi.h" 
    "engine/scene/scene.h" 
    "engine/util/util.h"
    "editor/editor.h") 


 
if(MSVC)
  target_compile_options(flower PUBLIC "/ZI")
  target_link_options(flower PUBLIC "/INCREMENTAL")   
endif()  

add_definitions(${BULLET_DEFINITIONS})
  
target_link_libraries(flower PUBLIC 
    Vulkan::Vulkan 
    assimp
    VulkanMemoryAllocator
    EnTT::EnTT
    utf8cpp 
    nativefiledialog
    spdlog::spdlog 
    glm::glm 
    stduuid 
    nlohmann_json 
    inipp::inipp  
    glfw 
    utf8cpp 
    EnTT::EnTT 
    lz4_static
    ffx_fsr2_api_x64 
    ffx_fsr2_api_vk_x64
    tinyexr
    ${BULLET_LIBRARY_DIRS}
)

set_target_properties(flower PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/install") 

if (MSVC)
    target_compile_options(flower PRIVATE /bigobj)
else ()
    target_compile_options(flower PRIVATE -Wa,-mbig-obj)
endif ()