#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "vk_engine.h"
int main(int argc, char *argv[])
{
    VulkanEngine engine;

    engine.init();

    engine.run();

    engine.cleanup();

    return 0;
}
