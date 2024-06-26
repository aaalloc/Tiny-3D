#pragma once

#include "camera.h"
#include <vk_descriptors.h>
#include <vk_initializers.h>
#include <vk_loader.h>
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

struct FrameData
{
    VkSemaphore swapchainSemaphore; // for render commands wait on the swapchain image request
    VkSemaphore renderSemaphore;    // control presenting the image to the OS after drawing finish
    VkFence renderFence;            // let us wait for the draw command of a given frame to be finished
    VkCommandPool commandPool;
    VkCommandBuffer mainCommandBuffer;
    DeletionQueue deletionQueue;
    DescriptorAllocatorGrowable frameDescriptors;
};

struct GLTFMetallic_Roughness
{
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    VkDescriptorSetLayout materialLayout;

    struct MaterialConstants
    {
        glm::vec4 colorFactors;
        glm::vec4 metal_rough_factors;
        // padding, we need it anyway for uniform buffers
        glm::vec4 extra[14];
    };

    struct MaterialResources
    {
        AllocatedImage colorImage;
        VkSampler colorSampler;
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler;
        VkBuffer dataBuffer;
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void build_pipelines(VulkanEngine *engine);
    void clear_resources(VkDevice device);

    MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources &resources,
                                    DescriptorAllocatorGrowable &descriptorAllocator);
};

struct MeshNode : public Node
{

    std::shared_ptr<MeshAsset> mesh;

    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

struct RenderObject
{
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    MaterialInstance *material;
    Bounds bounds;

    glm::mat4 transform;
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<RenderObject> TransparentSurfaces;
};

struct EngineStats
{
    float frametime;
    int triangle_count;
    int drawcall_count;
    float scene_update_time;
    float mesh_draw_time;
};

class VulkanEngine
{

  public:
    EngineStats stats;
    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;
    MaterialInstance defaultData;
    GLTFMetallic_Roughness metalRoughMaterial;
    Camera mainCamera;
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
    AllocatedImage _depthImage;
    VkExtent2D _drawExtent;

    DescriptorAllocatorGrowable globalDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;

    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    VmaAllocator _allocator;

    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;
    GPUMeshBuffers rectangle;

    std::vector<std::shared_ptr<MeshAsset>> testMeshes;

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer &buffer);
    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    void draw_geometry(VkCommandBuffer cmd);
    void init_default_data();

    VkDescriptorSetLayout _singleImageDescriptorLayout;
    GPUSceneData sceneData;
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;

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

    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;

    void update_scene();

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
    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();
    void init_imgui();
    void draw_background(VkCommandBuffer cmd);
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
};
