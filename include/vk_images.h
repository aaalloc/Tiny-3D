
#pragma once
#include "vk_types.h"
#include <vulkan/vulkan.h>

class VulkanEngine;
namespace vkutil
{

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize,
                         VkExtent2D dstSize);
void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);
AllocatedImage create_image(VulkanEngine &engine, void *data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                            bool mipmapped = false);
AllocatedImage create_image(VulkanEngine &engine, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                            bool mipmapped = false);
void destroy_image(VulkanEngine &engine, const AllocatedImage &img);
}; // namespace vkutil