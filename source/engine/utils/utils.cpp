#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#define STB_DXT_IMPLEMENTATION
#include <stb/stb_dxt.h>

#include <nlohmann/json.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION 
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <tinygltf/tiny_gltf.h>