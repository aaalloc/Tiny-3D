#pragma once

#include <vk_descriptors.h>
#include <vk_initializers.h>
#include <vk_types.h>

// bootstrap library
#include "VkBootstrap.h"

/*
Doing callbacks like this is inneficient at scale, because we are storing whole std::functions for every object we are
deleting, which is not going to be optimal. For the amount of objects we will use in this tutorial, its going to be
fine. but if you need to delete thousands of objects and want them deleted faster, a better implementation would be to
store arrays of vulkan handles of various types such as VkImage, VkBuffer, and so on. And then delete those from a
loop.
*/

constexpr unsigned int FRAME_OVERLAP = 2;

struct ComputePushConstants
{
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
    float time;
};

struct ComputeEffect
{
    const char *name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

class VulkanEngine
{
  public:
    DeletionQueue _mainDeletionQueue;
    FrameData _frames[FRAME_OVERLAP];
    VkInstance _instance;                      // Vulkan library handle
    VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
    VkPhysicalDevice _chosenGPU;               // GPU chosen as the default device
    VkDevice _device;                          // Vulkan device for commands
    VkSurfaceKHR _surface;
    VkQueue _graphicQueue;
    uint32_t _graphicsQueueFamily;

    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent; // Vulkan window surface

    AllocatedImage _drawImage;
    VkExtent2D _drawExtent;

    DescriptorAllocator globalDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    VmaAllocator _allocator;

    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;

    void init_triangle_pipeline();

    void draw_geometry(VkCommandBuffer cmd);

    // draw resources

    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{1700, 900};

    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{0};

    struct SDL_Window *_window{nullptr};

    static VulkanEngine &Get();
    FrameData &get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }

    void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

    // initializes everything in the engine
    void init();

    // shuts down the engine
    void cleanup();

    // draw loop
    void draw();

    // run main loop
    void run();

  private:
    void init_vulkan();
    void init_swapchain();
    void init_commands();
    void init_sync_structures();
    void init_descriptors();
    void init_pipelines();
    void init_background_pipelines();
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void init_imgui();
    void draw_background(VkCommandBuffer cmd);
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
};
