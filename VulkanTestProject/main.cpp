#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <array>
#include <fstream>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


int main1() {
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);

	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	std::cout << extensionCount << " extensions supported\n";

	glm::mat4 matrix;
	glm::vec4 vec;
	auto test = matrix * vec;

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	glfwTerminate();

	return 0;
}

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

struct Vertex
{
	glm::vec2 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}


	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};


const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}


class HelloTriangleApplication {
public:
	void initWindow();

	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:


	static const int MAX_FRAMES_IN_FLIGHT = 2;


	void	checkExtensions(const char** glfwExtensions, uint32_t glfwExtensionCount);
	bool	checkValidationLayerSupport();

	std::vector<const char*>	getRequiredExtensions();


	static VKAPI_ATTR VkBool32	VKAPI_CALL	VkDebugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData);

	static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);


	void setupVkDebugMessenger();


	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily{};
		std::optional<uint32_t> presentFamily{};


		bool isComplete()
		{
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	QueueFamilyIndices  findQueueFamilies(VkPhysicalDevice bestCandidate);
	bool				checkDeviceExtensionSupport(VkPhysicalDevice device);
	int					rateDeviceSuitability(VkPhysicalDevice device);





	void createInstance(const char** glfwExtensions, uint32_t glfwExtensionCount);

	void pickPhysicalDevice();

	void createLogicalDevice();

	void createSurface();

	struct SwapChainSupportDetails
	{
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};
	SwapChainSupportDetails	querySwapChainSupport(VkPhysicalDevice device);

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	VkSurfaceFormatKHR		chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

	VkPresentModeKHR		chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

	VkExtent2D				chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	void					createSwapChain();
	void					retrieveSwapChainImages(uint32_t imageCount);

	void					createImageViews();

	void					createDescriptorSetLayout();

	void					createDescriptorPool();

	void					createDescriptorSets();

	void					createGraphicsPipeline();

	void createFramebuffers();

	void createCommandPool();

	VkShaderModule	createShaderModule(const std::vector<char>& bytecode);

	void	createVertexBuffer();

	void	createIndexBuffer();

	void	createUniformBuffers();

	void	updateUniformBuffer(uint32_t currentImage);

	void	createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

	void	copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
	void	copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

	void	createRenderPass();

	void createCommandBuffers();

	VkCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(VkCommandBuffer commandBuffer);

	void createSyncObjects();

	void createTextureImage();

	void createTextureImageView();

	void createTextureSampler();

	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

	VkImageView	createImageView(VkImage image, VkFormat format);

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);


	void recreateSwapChain();
	void cleanupSwapChain();

	void initVulkan();

	void drawFrame();
	void mainLoop();

	void cleanup();

	static std::vector<char> readFile(const std::string& filename);

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);


	GLFWwindow* window;

	VkInstance  instance;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue graphicsQueue;

	VkQueue presentQueue;

	VkSurfaceKHR surface;

	VkSwapchainKHR swapChain;

	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;

	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;

	VkShaderModule vertShaderModule;

	VkShaderModule fragShaderModule;

	VkRenderPass renderPass;

	VkDescriptorSetLayout			descriptorSetLayout;

	VkDescriptorPool				descriptorPool;

	std::vector<VkDescriptorSet>	descriptorSets;

	VkPipelineLayout pipelineLayout;

	VkPipeline graphicsPipeline;

	std::vector<VkFramebuffer> swapChainFramebuffers;

	std::vector<VkCommandBuffer> commandBuffers;

	VkCommandPool commandPool;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<VkFence> imagesInFlight;

	size_t currentFrame = 0;

	bool framebufferResized = false;
	bool minimized = false;

	const std::vector<Vertex> vertices =
	{
		{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
		{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
		{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
		{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
	};

	const std::vector<uint16_t> indices =
	{
		0, 1, 2, 2, 3, 0
	};


	struct UniformBufferObject
	{
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};



	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;

	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;


	VkImage			textureImage;
	VkDeviceMemory	textureImageMemory;
	VkImageView		textureImageView;

	VkSampler		textureSampler;

	VkDebugUtilsMessengerEXT debugMessenger;
};


VkCommandBuffer HelloTriangleApplication::beginSingleTimeCommands()
{
	// TODO: You may wish to create a separate command pool for these kinds of short-lived buffers,
	// because the implementation may be able to apply memory allocation optimizations.
	// You should use the VK_COMMAND_POOL_CREATE_TRANSIENT_BIT flag during command pool generation in that case.
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = commandPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	// We're only going to use the command buffer once and wait with returning from the function until the copy operation has finished executing.
	// It's good practice to tell the driver about our intent using VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT.
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}


void HelloTriangleApplication::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

	// TODO:
	// We could use a fence and wait with vkWaitForFences,
	// or simply wait for the transfer queue to become idle with vkQueueWaitIdle.
	// A fence would allow you to schedule multiple transfers simultaneously and wait for all of them complete,
	// instead of executing one at a time. That may give the driver more opportunities to optimize.
	vkQueueWaitIdle(graphicsQueue); // TODO : yuk !!

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}


void HelloTriangleApplication::initWindow()
{
	int ok = glfwInit();
	assert(ok == GLFW_TRUE);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

}


bool HelloTriangleApplication::checkValidationLayerSupport()
{
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	// check if all the validation layers we want are available
	for (const char* layerName : validationLayers)
	{
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers)
		{
			if (strcmp(layerName, layerProperties.layerName) == 0)
			{
				std::cout << "Found available validation layer: " << layerName << ".\n";
				layerFound = true;
				break;
			}
		}

		assert(layerFound); // false not supposed to happen, I'm asserting.
	}

	return false;
}

std::vector<const char*> HelloTriangleApplication::getRequiredExtensions()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}


VkBool32 HelloTriangleApplication::VkDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	std::cerr << "Vk validation layer debug callback message: " << pCallbackData->pMessage << std::endl;

	// The callback returns a boolean that indicates if the Vulkan call that triggered the validation layer message should be aborted.
	// If the callback returns true, then the call is aborted with the VK_ERROR_VALIDATION_FAILED_EXT error.
	// This is normally only used to test the validation layers themselves, so you should always return VK_FALSE.
	return VK_FALSE;
}

void HelloTriangleApplication::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = VkDebugCallback;
}

void HelloTriangleApplication::setupVkDebugMessenger()
{
	// No debug ? Nothing to do
	if constexpr (!enableValidationLayers)
		return;

	VkDebugUtilsMessengerCreateInfoEXT createInfo{};
	populateDebugMessengerCreateInfo(createInfo);
	createInfo.pUserData = nullptr; // Optional

	bool ok = CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger);
	assert(ok == VK_SUCCESS);


}




HelloTriangleApplication::QueueFamilyIndices  HelloTriangleApplication::findQueueFamilies(VkPhysicalDevice device)
{
	// Logic to find graphics queue family
	QueueFamilyIndices indices;

	// Logic to find queue family indices to populate struct with
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int iFam = 0;
	for (const auto& queueFamily : queueFamilies)
	{
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			indices.graphicsFamily = iFam;
		}

		// Check for present support because we want to present images to screen
		// TODO; this could be factored out because we dont care about that if we're rendering offscreen.
		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, iFam, surface, &presentSupport);

		if (presentSupport)
		{
			indices.presentFamily = iFam;
		}


		if (indices.isComplete())
			break;

		iFam++;
	}

	return indices;
}


bool HelloTriangleApplication::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
	// Check that the provided device can present images using a swap chain.
	// TODO: Refactor that in configuration
	const std::vector<const char*> unsupportedDeviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	for (auto& neededExtension : unsupportedDeviceExtensions)
	{
		auto found = std::find_if(
			availableExtensions.begin(),
			availableExtensions.end(),
			[neededExtension](auto& extensionProperties) { return (strcmp(neededExtension, extensionProperties.extensionName) == 0); });

		if (found == availableExtensions.end())
			return false; // One needed extension was not found
	}

	return true;
}


int HelloTriangleApplication::rateDeviceSuitability(VkPhysicalDevice device)
{
	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(device, &deviceProperties);

	std::cout << "Device ID [" << deviceProperties.deviceID
		<< "] | Device name [" << deviceProperties.deviceName
		<< "] | Vulkan version [" << deviceProperties.apiVersion
		<< "] | Device type [" << (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "GPU" : "Not GPU")
		<< "]\n";

	int score = 0;

	// Discrete GPUs have a significant performance advantage
	if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		score += 1000;
	}

	// Maximum possible size of textures affects graphics quality
	score += deviceProperties.limits.maxImageDimension2D;
	score += (int)deviceProperties.limits.maxSamplerAnisotropy;
	// TODO : add more here...

	// List of features the application cannot live without

	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

	bool ok = (
		deviceFeatures.geometryShader
		&& deviceFeatures.tessellationShader
		&& deviceFeatures.multiDrawIndirect
		&& deviceFeatures.multiViewport
		&& deviceFeatures.depthClamp
		&& deviceFeatures.depthBiasClamp
		&& deviceFeatures.fillModeNonSolid
		&& deviceFeatures.samplerAnisotropy);

	if constexpr (enableValidationLayers)
	{
		ok &= (bool)deviceFeatures.pipelineStatisticsQuery; // for Debug only
	}

	if (!ok)
	{
		score = 0;
	}
	else // Do additional checks that could disqualify this device
	{
		// Then we want to make sure our device also has all the right queues for our needs
		QueueFamilyIndices familyIndices = findQueueFamilies(device);
		if (false == familyIndices.isComplete())
			score = 0;
		else if (familyIndices.presentFamily == familyIndices.graphicsFamily)
		{
			// add a bonus to explicitly prefer a physical device that supports drawing and presentation in the same queue for improved performance.
			score += 100;
		}

		const bool extensionsAreSupported = checkDeviceExtensionSupport(device);
		if (!extensionsAreSupported)
			score = 0;
		else
		{
			// Swap chain is supported, but does the swap chain support all the settings we need ?
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			if (swapChainSupport.formats.empty() || swapChainSupport.presentModes.empty())
				score = 0; // device is not suitable
		}
	}

	std::cout << "Device score: " << score << std::endl;

	return score;
}

void HelloTriangleApplication::pickPhysicalDevice()
{
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	// If there are 0 devices with Vulkan support then there is no point going further.
	assert(deviceCount > 0);

	// Now of all available devices pick the best one.
	std::cout << "Selecting appropriate Vulkan device...\n";

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	VkPhysicalDevice bestCandidate = VK_NULL_HANDLE;
	int bestCandidateScore = 0;
	int bestCandidateIndex = 0;
	int devIndex = 0;

	for (const auto& device : devices)
	{
		std::cout << "Device " << devIndex << ":\n";

		int score = rateDeviceSuitability(device);
		if (bestCandidateScore < score)
		{
			bestCandidate = device;
			bestCandidateScore = score;
			bestCandidateIndex = devIndex;
		}
		devIndex++;
	}

	// Check if the best candidate is suitable at all
	assert(bestCandidateScore > 0);

	physicalDevice = bestCandidate;

	std::cout << "Picked device " << bestCandidateIndex << std::endl;

	findQueueFamilies(bestCandidate);

	assert(physicalDevice != VK_NULL_HANDLE);
}


void HelloTriangleApplication::createLogicalDevice()
{
	// TODO improvement : use findQueueFamilies only once... store that information somewhere from when we picked the device
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

	// Create one device queue per unique family index we need...
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(uniqueQueueFamilies.size());

	static const float queuePriority = 1.0f;

	int deviceInfoIndex = 0;
	for (uint32_t queueFamily : uniqueQueueFamilies)
	{
		VkDeviceQueueCreateInfo& queueCreateInfo = queueCreateInfos[deviceInfoIndex];
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		deviceInfoIndex++;
	}

	// Fill here all the device features we want. TODO : refactor that with the rateDeviceSuitability function!
	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.geometryShader = true;
	deviceFeatures.tessellationShader = true;
	deviceFeatures.multiDrawIndirect = true;
	deviceFeatures.multiViewport = true;
	deviceFeatures.depthClamp = true;
	deviceFeatures.depthBiasClamp = true;
	deviceFeatures.fillModeNonSolid = true;
	deviceFeatures.samplerAnisotropy = true;

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.queueCreateInfoCount = (uint32_t) queueCreateInfos.size();

	createInfo.pEnabledFeatures = &deviceFeatures;

	// TODO: Refactor that with checkDeviceExtensionSupport
	const std::vector<const char*> neededDeviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	createInfo.enabledExtensionCount = (uint32_t) neededDeviceExtensions.size();
	createInfo.ppEnabledExtensionNames = neededDeviceExtensions.data();

	createInfo.enabledLayerCount = 0;
	if constexpr (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}

	bool ok = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
	assert(ok == VK_SUCCESS);

	vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
	vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

}


void HelloTriangleApplication::createSurface()
{
	VkResult ok = glfwCreateWindowSurface(instance, window, nullptr, &surface);
	assert(ok == VK_SUCCESS);
}


HelloTriangleApplication::SwapChainSupportDetails HelloTriangleApplication::querySwapChainSupport(
	VkPhysicalDevice device)
{
	SwapChainSupportDetails details{};

	//There are basically three kinds of properties we need to check:
	// - Basic surface capabilities (min / max number of images in swap chain, min / max width and height of images)
	// - Surface formats (pixel format, color space)
	// - Available presentation modes

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0)
	{
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	if (presentModeCount != 0)
	{
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

VkSurfaceFormatKHR HelloTriangleApplication::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
	assert(availableFormats.empty() == false); // not supposed to happen

	// For the color space we'll use SRGB if it is available, because it results in more accurate perceived colors.
	// It is also pretty much the standard color space for images, like the textures we'll use later on.
	// Because of that we should also use an SRGB color format, of which one of the most common ones is VK_FORMAT_B8G8R8A8_SRGB.
	for (const auto& availableFormat : availableFormats)
	{
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			return availableFormat;
		}
	}

	// just settle for the first available one
	return availableFormats[0];
}

VkPresentModeKHR HelloTriangleApplication::chooseSwapPresentMode(
	const std::vector<VkPresentModeKHR>& availablePresentModes)
{
	// VK_PRESENT_MODE_MAILBOX_KHR is typically used to implement triple buffering.
	// Triple buffering is a very nice trade-off because it allows us to avoid tearing
	// while still maintaining a fairly low latency by rendering new images that are
	// as up-to-date as possible right until the vertical blank.
	// So, let's look through the list to see if it's available
	for (const auto& availablePresentMode : availablePresentModes)
	{
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			return availablePresentMode;
		}
	}

	// Only the VK_PRESENT_MODE_FIFO_KHR mode is guaranteed to be available,
	// So choose it as the fallback (it's double buffering).
	return VK_PRESENT_MODE_FIFO_KHR;
}


VkExtent2D HelloTriangleApplication::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != UINT32_MAX)
	{
		return capabilities.currentExtent;
	}
	else
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent;
	}
}


void HelloTriangleApplication::createSwapChain()
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

	// We have to decide how many images we would like to have in the swap chain.
	// Simply sticking to the minimum means that we may sometimes have to wait on the driver
	// to complete internal operations before we can acquire another image to render to.
	// Therefore it is recommended to request at least one more image than the minimum
	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

	// We should also make sure to not exceed the maximum number of images while doing this. Note that 0 is a special value that means that there is no maximum.
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1; // This is always 1 unless you are developing a stereoscopic 3D application.
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;


	// Next, we need to specify how to handle swap chain images that will be used across multiple queue families.
	// That will be the case in our application if the graphics queue family is different from the presentation queue.
	// If the graphics queue familyand presentation queue family are the same, which will be the case on most hardware,
	// then we should stick to exclusive mode, because concurrent mode requires you to specify at least two distinct queue families.
	QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
	uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

	if (indices.graphicsFamily != indices.presentFamily)
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0; // Optional
		createInfo.pQueueFamilyIndices = nullptr; // Optional
	}

	// Don't apply any transform to the swap chain
	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

	// The compositeAlpha field specifies if the alpha channel should be used
	// for blending with other windows in the window system.
	// You'll almost always want to simply ignore the alpha channel, hence VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR.
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;


	createInfo.presentMode = presentMode;

	// clipped = VK_TRUE means that we don't care about the color of pixels that are obscured,
	// for example because another window is in front of them.
	createInfo.clipped = VK_TRUE;

	// TODO : change that during swap chain recreation (when the window is resized for example)
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	VkResult ok = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain);
	assert(ok == VK_SUCCESS);

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;

	retrieveSwapChainImages(imageCount);
}


void HelloTriangleApplication::retrieveSwapChainImages(uint32_t imageCount)
{
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
	swapChainImages.resize(imageCount);
	vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());


}


void HelloTriangleApplication::createImageViews()
{
	swapChainImageViews.resize(swapChainImages.size());

	int imgIndex = 0;

	for (VkImage& swapImage : swapChainImages)
	{
		swapChainImageViews[imgIndex] = createImageView(swapImage, swapChainImageFormat);
		imgIndex++;
	}
}


void HelloTriangleApplication::createDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding uboLayoutBinding{};
	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;

	// The stageFlags field can be a combination of VkShaderStageFlagBits values or the value VK_SHADER_STAGE_ALL_GRAPHICS.
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &uboLayoutBinding;

	VkResult ok = vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
	assert(ok == VK_SUCCESS);

}


void HelloTriangleApplication::createDescriptorPool()
{
	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size());

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;

	poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

	// The structure has an optional flag similar to command pools
	// that determines if individual descriptor sets can be freed or not:
	// VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT.
	// We're not going to touch the descriptor set after creating it, so we don't need this flag.
	poolInfo.flags = 0;

	VkResult result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
	assert(result == VK_SUCCESS);

}


void HelloTriangleApplication::createDescriptorSets()
{
	// In our case we will create one descriptor set for each swap chain image, all with the same layout.
	// Unfortunately we do need all the copies of the layout because the next function expects an array matching the number of sets.
	std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
	allocInfo.pSetLayouts = layouts.data();


	descriptorSets.resize(swapChainImages.size());
	VkResult result = vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data());
	assert(result == VK_SUCCESS);
	// You don't need to explicitly clean up descriptor sets, because they will be automatically freed when the descriptor pool is destroyed.

	// The descriptor sets have been allocated now, but the descriptors within still need to be configured.
	for (size_t i = 0; i < swapChainImages.size(); i++)
	{
		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = uniformBuffers[i];
		// overwriting the whole buffer, like we are in this case
		bufferInfo.offset = 0;
		bufferInfo.range = VK_WHOLE_SIZE;

		VkWriteDescriptorSet descriptorWrite{};
		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = descriptorSets[i];
		descriptorWrite.dstBinding = 0;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo = &bufferInfo;
		descriptorWrite.pImageInfo = nullptr; // Optional
		descriptorWrite.pTexelBufferView = nullptr; // Optional

		vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
	}


}


void HelloTriangleApplication::createGraphicsPipeline()
{
	auto vertShaderCode = readFile("vert.spv");
	vertShaderModule = createShaderModule(vertShaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	auto fragShaderCode = readFile("frag.spv");
	fragShaderModule = createShaderModule(fragShaderCode);

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };


	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	// we want to use a viewport that covers the entire framebuffer :
	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)swapChainExtent.width;
	viewport.height = (float)swapChainExtent.height;
	viewport.minDepth = 0.0f; // In VK, normalized depth is [0;1] like in D3D, but this can be changed here.
	viewport.maxDepth = 1.0f;

	// we simply want to draw to the entire framebuffer, so we'll specify a scissor rectangle that covers it entirely:
	VkRect2D scissor{};
	scissor.offset = { 0, 0 };
	scissor.extent = swapChainExtent;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	// If depthClampEnable = VK_TRUE, fragments that are beyond the near and far planes are clamped
	// to them as opposed to discarding them. This is useful in some special cases like shadow maps.
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;

	// if rasterizerDiscardEnable = true, geometry never passes through the rasterizer stage. This basically disables any output to the framebuffer.
	rasterizer.rasterizerDiscardEnable = VK_FALSE;

	// display full poygons (use VK_POLYGON_MODE_LINE for wireframe, VK_POLYGON_MODE_POINT for point cloud)
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;

	// thickness of lines in terms of fragments.
	rasterizer.lineWidth = 1.0f;

	// Activate back-face culling and use clockwise order to determine what is the front face.
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f; // Optional
	rasterizer.depthBiasClamp = 0.0f; // Optional
	rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

	// Let's keep MSAA disabled for now
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f; // Optional
	multisampling.pSampleMask = nullptr; // Optional
	multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
	multisampling.alphaToOneEnable = VK_FALSE; // Optional

	// Add depth and stencil configuration later here...

	// Use alpha blending mode for now
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_TRUE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f; // Optional
	colorBlending.blendConstants[1] = 0.0f; // Optional
	colorBlending.blendConstants[2] = 0.0f; // Optional
	colorBlending.blendConstants[3] = 0.0f; // Optional


	// A limited amount of the state that we've specified in the previous structs can actually be changed without recreating the pipeline.
	// Examples are the size of the viewport, line width and blend constants.
	std::array<VkDynamicState, 2> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_LINE_WIDTH
	};

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = (uint32_t) dynamicStates.size();
	dynamicState.pDynamicStates = dynamicStates.data();

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
	pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

	VkResult ok = vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);
	assert(ok == VK_SUCCESS);

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;

	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = nullptr; // Optional
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = nullptr; // Optional

	pipelineInfo.layout = pipelineLayout;

	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;

	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
	pipelineInfo.basePipelineIndex = -1; // Optional

	ok = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);
	assert(ok == VK_SUCCESS);

}


void HelloTriangleApplication::createFramebuffers()
{
	swapChainFramebuffers.resize(swapChainImageViews.size());

	for (size_t i = 0; i < swapChainImageViews.size(); i++)
	{
		VkImageView attachments[] = {
			swapChainImageViews[i]
		};

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = swapChainExtent.width;
		framebufferInfo.height = swapChainExtent.height;
		framebufferInfo.layers = 1;

		VkResult ok = vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
		assert(ok == VK_SUCCESS);
	}
}


void HelloTriangleApplication::createCommandPool()
{
	// TODO : refactor to avoid having to recall that each time
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

	// Command buffers are executed by submitting them on one of the device queues, like the graphics and presentation queues we retrieved.
	// Each command pool can only allocate command buffers that are submitted on a single type of queue.
	// We're going to record commands for drawing, which is why we've chosen the graphics queue family.
	VkCommandPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

	// We will only record the command buffers at the beginning of the program
	// and then execute them many times in the main loop, so we're not going to use either of these flags.
	poolInfo.flags = 0; // Optional

	VkResult ok = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
	assert(ok == VK_SUCCESS);


}


VkShaderModule HelloTriangleApplication::createShaderModule(const std::vector<char>& bytecode)
{
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = bytecode.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(bytecode.data());

	VkShaderModule shaderModule;
	VkResult ok = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
	assert(ok == VK_SUCCESS);

	return shaderModule;
}


void HelloTriangleApplication::createVertexBuffer()
{
	VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* data;
	VkResult ok = vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
	assert(ok == VK_SUCCESS);
	memcpy(data, vertices.data(), (size_t)bufferSize);
	vkUnmapMemory(device, stagingBufferMemory);

	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

	copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
}



void HelloTriangleApplication::createIndexBuffer()
{
	VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void* data;

	VkResult ok = vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
	assert(ok == VK_SUCCESS);
	memcpy(data, indices.data(), (size_t)bufferSize);
	vkUnmapMemory(device, stagingBufferMemory);

	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

	copyBuffer(stagingBuffer, indexBuffer, bufferSize);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
}


void HelloTriangleApplication::createUniformBuffers()
{
	VkDeviceSize bufferSize = sizeof(UniformBufferObject);

	uniformBuffers.resize(swapChainImages.size());
	uniformBuffersMemory.resize(swapChainImages.size());

	for (size_t i = 0; i < swapChainImages.size(); i++)
	{
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
	}


}


void HelloTriangleApplication::updateUniformBuffer(uint32_t currentImage)
{
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	UniformBufferObject ubo{};
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

	ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

	ubo.proj[1][1] *= -1;

	void* data;
	VkResult ok = vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
	assert(ok == VK_SUCCESS);
	memcpy(data, &ubo, sizeof(ubo));
	vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
}


void HelloTriangleApplication::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult ok = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
	assert(ok == VK_SUCCESS);


	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

	ok = vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory);
	assert(ok == VK_SUCCESS);


	vkBindBufferMemory(device, buffer, bufferMemory, 0);
}


void HelloTriangleApplication::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	endSingleTimeCommands(commandBuffer);
}


void HelloTriangleApplication::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;

	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;

	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = {
		width,
		height,
		1
	};

	vkCmdCopyBufferToImage(
		commandBuffer,
		buffer,
		image,
		//  I'm assuming here that the image has already been transitioned to the layout that is optimal for copying pixels to.
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		// Right now we're only copying one chunk of pixels to the whole image
		1,
		&region
	);

	endSingleTimeCommands(commandBuffer);
}


uint32_t HelloTriangleApplication::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (uint32_t iMemType = 0; iMemType < memProperties.memoryTypeCount; iMemType++)
	{
		if (typeFilter & (1 << iMemType) && (memProperties.memoryTypes[iMemType].propertyFlags & properties) == properties)
		{
			return iMemType;
		}
	}

	assert(false); // not supposed to happen
	return 0;
}


void HelloTriangleApplication::createRenderPass()
{
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = swapChainImageFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT; // not doing anything with MSAA yet so sticking to one sample

	// clear the framebuffer to black before drawing a new frame.
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;

	// Rendered contents will be stored in memory and can be read later
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

	// Our application won't do anything with the stencil buffer, so the results of loading and storing are irrelevant.
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	// means that we don't care what previous layout the image was in.
	// doesn't matter since we're going to clear it anyway.
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	// We want the image to be ready for presentation using the swap chain after rendering
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;


	VkAttachmentReference colorAttachmentRef{};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{};
	// Vulkan may also support compute subpasses in the future, so we have to be explicit about this being a graphics subpass.
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	// Create an explicit dependency on the subpass to make sure we acquire the image at the right time
	VkSubpassDependency dependency{};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;

	// We need to wait for the swap chain to finish reading from the image before we can access it.
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;

	// The operations that should wait on this are in the color attachment stage and involve the writing of the color attachment.
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;


	VkResult ok = vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
	assert(ok == VK_SUCCESS);

}


void HelloTriangleApplication::createCommandBuffers()
{
	commandBuffers.resize(swapChainFramebuffers.size());

	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // We won't make use of the secondary command buffer functionality here
	allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

	VkResult ok = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());
	assert(ok == VK_SUCCESS);

	for (size_t i = 0; i < commandBuffers.size(); i++)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0; // Optional
		beginInfo.pInheritanceInfo = nullptr; // Optional

		VkResult ok = vkBeginCommandBuffer(commandBuffers[i], &beginInfo);
		assert(ok == VK_SUCCESS);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[i];

		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		//  the clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR
		VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo,
			//  The render pass commands will be embedded in the primary command buffer itself and no secondary command buffers will be executed.
			VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkBuffer vertexBuffers[] = { vertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

		vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

		vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffers[i]);

		ok = vkEndCommandBuffer(commandBuffers[i]);
		assert(ok == VK_SUCCESS);


	}



}


void HelloTriangleApplication::createSyncObjects()
{

	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
	imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	// Create the fences in 'signaled' state or waiting for them for the first time will hang forever.
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
			vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
		{
			assert(false); //failed to create semaphores for a frame!
		}
	}
}


void HelloTriangleApplication::createTextureImage()
{
	int texWidth, texHeight, texChannels;
	// The STBI_rgb_alpha value forces the image to be loaded with an alpha channel,
	// even if it doesn't have one, which is nice for consistency with other textures in the future.
	stbi_uc* pixels = stbi_load("assets/textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	VkDeviceSize imageSize = texWidth * texHeight * 4;

	assert(pixels != nullptr);

	// We'll start by creating a staging resource and filling it with pixel data and then we copy this to the final image object that we'll use for rendering.
	// Although it is possible to create a staging image for this purpose, Vulkan also allows you to copy pixels from a VkBuffer to an image
	// and the API for this is actually faster on some hardware.
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	// The buffer should be in host visible memory so that we can map itand it should be usable as a transfer source so that we can copy it to an image later on :
	createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	// We can then directly copy the pixel values that we got from the image loading library to the buffer:
	void* data;
	VkResult ok = vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
	assert(ok == VK_SUCCESS);
	memcpy(data, pixels, (size_t)imageSize);
	vkUnmapMemory(device, stagingBufferMemory);

	// Don't forget to clean up the original pixel array now
	stbi_image_free(pixels);

	createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

	transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

	transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
}


void HelloTriangleApplication::createTextureImageView()
{
	textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
}


void HelloTriangleApplication::createTextureSampler()
{
	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;

	// the repeat mode is probably the most common mode, because it can be used to tile textures like floors and walls.
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

	// There is no reason not to use this unless performance is a concern.
	// Instead of enforcing the availability of anisotropic filtering, it's also possible to simply not use it by conditionally setting:
	//		samplerInfo.anisotropyEnable = VK_FALSE;
	//		samplerInfo.maxAnisotropy = 1.0f;
	samplerInfo.anisotropyEnable = VK_TRUE;

	// The maxAnisotropy field limits the amount of texel samples that can be used to calculate the final color.
	// A lower value results in better performance, but lower quality results.
	// To figure out which value we can use, we need to retrieve the properties of the physical device.
	// TODO: refactor this...
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(physicalDevice, &properties);
	samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

	// The borderColor field specifies which color is returned when sampling beyond the image with clamp to border addressing mode.
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

	// the texels are addressed using the [0, 1) range on all axes.
	samplerInfo.unnormalizedCoordinates = VK_FALSE;

	// This is mainly used for percentage-closer filtering on shadow maps.
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = 0.0f;

	VkResult ok = vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler);
	assert(ok == VK_SUCCESS);


}


void HelloTriangleApplication::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                                           VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = static_cast<uint32_t>(width);
	imageInfo.extent.height = static_cast<uint32_t>(height);
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;

	// we should use the same format for the texels as the pixels in the buffer, otherwise the copy operation will fail.
	imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;

	// The tiling field can have one of two values :
	// VK_IMAGE_TILING_LINEAR: Texels are laid out in row - major order like our pixels array
	// VK_IMAGE_TILING_OPTIMAL : Texels are laid out in an implementation defined order for optimal access
	// Unlike the layout of an image, the tiling mode cannot be changed at a later time.
	// If you want to be able to directly access texels in the memory of the image,
	// then you must use VK_IMAGE_TILING_LINEAR.
	// We will be using a staging buffer instead of a staging image, so this won't be necessary.
	// We will be using VK_IMAGE_TILING_OPTIMAL for efficient access from the shader.
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;

	// There are only two possible values for the initialLayout of an image :
	// - VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPUand the very first transition will discard the texels.
	// - VK_IMAGE_LAYOUT_PREINITIALIZED : Not usable by the GPU, but the first transition will preserve the texels.
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	// We want to be able to access the image from the shader to color our mesh, so the usage should include VK_IMAGE_USAGE_SAMPLED_BIT.
	imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

	// The image will only be used by one queue family: the one that supports graphics (and therefore also) transfer operations.
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	// Related to multisampling.This is only relevant for images that will be used as attachments
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

	imageInfo.flags = 0; // Optional

	VkResult ok = vkCreateImage(device, &imageInfo, nullptr, &textureImage);
	assert(ok == VK_SUCCESS);

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, textureImage, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	ok = vkAllocateMemory(device, &allocInfo, nullptr, &textureImageMemory);
	assert(ok == VK_SUCCESS);

	vkBindImageMemory(device, textureImage, textureImageMemory, 0);
}


VkImageView HelloTriangleApplication::createImageView(VkImage image, VkFormat format)
{
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView imageView;
	VkResult ok = vkCreateImageView(device, &viewInfo, nullptr, &imageView);
	assert(ok == VK_SUCCESS);

	return imageView;
}


void HelloTriangleApplication::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;

	// If you are using the barrier to transfer queue family ownership, then these two fields should be the indices of the queue families.
	// They must be set to VK_QUEUE_FAMILY_IGNORED if you don't want to do this (not the default value!).
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	// TODO: To configure depending on the actual type of image (color, depth, stencil, cubemap, mip number...)
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	// There are two transitions we need to handle :
	// Undefined -> transfer destination : transfer writes that don't need to wait on anything
	// Transfer destination -> shader reading : shader reads should wait on transfer writes,
	// specifically the shader reads in the fragment shader, because that's where we're going to use the texture
	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; // the earliest possible pipeline stage for the pre-barrier operations.
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT; // not a real stage within the graphics and compute pipelines. It is more of a pseudo-stage where transfers happen.
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else
	{
		assert(false); // unmanaged transition !
	}


	// The first parameter after the command buffer specifies in which pipeline stage the operations occur that should happen before the barrier.
	// The second parameter specifies the pipeline stage in which operations will wait on the barrier.
	// The pipeline stages that you are allowed to specify before and after the barrier depend on how you use the resource before and after the barrier.
	// The third parameter is either 0 or VK_DEPENDENCY_BY_REGION_BIT.
	// The latter turns the barrier into a per - region condition.
	// That means that the implementation is allowed to already begin reading from the parts of a resource that were written so far, for example.
	// The last three pairs of parameters reference arrays of pipeline barriers of the three available types :
	// memory barriers, buffer memory barriers, and image memory barriers like the one we're using here.
	vkCmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

	endSingleTimeCommands(commandBuffer);
}


void HelloTriangleApplication::recreateSwapChain()
{
	// TODO: This minimization-catching trick is ugly as hell ! To refactor !
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(device);

	cleanupSwapChain();

	createSwapChain();
	createImageViews();
	createRenderPass();
	createGraphicsPipeline();
	createFramebuffers();
	createUniformBuffers();
	createDescriptorPool();
	createDescriptorSets();
	createCommandBuffers();

	if (framebufferResized)
	{
		framebufferResized = false; // we took care of the resizing.
	}
}

void HelloTriangleApplication::cleanupSwapChain()
{
	for (size_t i = 0; i < swapChainFramebuffers.size(); i++)
	{
		vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
	}

	for (size_t i = 0; i < swapChainImages.size(); i++)
	{
		vkDestroyBuffer(device, uniformBuffers[i], nullptr);
		vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
	}

	vkDestroyDescriptorPool(device, descriptorPool, nullptr);

	// We could recreate the command pool from scratch, but that is rather wasteful.
	// Instead, clean up the existing command buffers with vkFreeCommandBuffers.
	// This way we can reuse the existing pool to allocate the new command buffers.
	vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());


	vkDestroyShaderModule(device, vertShaderModule, nullptr);
	vkDestroyShaderModule(device, fragShaderModule, nullptr);

	vkDestroyPipeline(device, graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	vkDestroyRenderPass(device, renderPass, nullptr);

	for (size_t i = 0; i < swapChainImageViews.size(); i++)
	{
		vkDestroyImageView(device, swapChainImageViews[i], nullptr);
	}

	vkDestroySwapchainKHR(device, swapChain, nullptr);
}


void HelloTriangleApplication::initVulkan()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	checkExtensions(glfwExtensions, glfwExtensionCount);
	createInstance(glfwExtensions, glfwExtensionCount);

	// Don't need this one anymore but leaving it on the code for reference
	//setupVkDebugMessenger();

	// The window surface needs to be created right after the instance creation, because it can actually influence the physical device selection.
	createSurface();

	pickPhysicalDevice();

	createLogicalDevice();

	createSwapChain();

	createImageViews();

	createRenderPass();

	createDescriptorSetLayout();

	createGraphicsPipeline();

	createFramebuffers();

	createCommandPool();

	createTextureImage();

	createTextureImageView();

	createVertexBuffer();

	createIndexBuffer();

	createUniformBuffers();

	createDescriptorPool();

	createDescriptorSets();

	createCommandBuffers();

	createSyncObjects();
}


void HelloTriangleApplication::drawFrame()
{
	vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

	// If the swap chain turns out to be out of date when attempting to acquire an image, then it is no longer possible to present to it.
	// Therefore we should immediately recreate the swap chain and try again in the next drawFrame call.
	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		std::cout << "Swap chain out of date - recreation in progress\n";
		recreateSwapChain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) // don't manage subobtimal here because we successfully acquired the image : do it after present
	{
		assert(false);
	}

	// Check if a previous frame is using this image (i.e. there is its fence to wait on)
	if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
		vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
	}
	// Mark the image as now being in use by this frame
	imagesInFlight[imageIndex] = inFlightFences[currentFrame];


	updateUniformBuffer(imageIndex);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

	VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;


	vkResetFences(device, 1, &inFlightFences[currentFrame]);

	VkResult ok = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
	assert(ok == VK_SUCCESS);

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { swapChain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;

	// It's not necessary if you're only using a single swap chain,
	// because you can simply use the return value of the present function.
	presentInfo.pResults = nullptr; // Optional

	result = vkQueuePresentKHR(presentQueue, &presentInfo);

	// The vkQueuePresentKHR function returns the same values than vkAcquireNextImageKHR with the same meaning.
	// In this case we will also recreate the swap chain if it is suboptimal, because we want the best possible result.
	// Regarding framebuffer resized : It is important to check this after vkQueuePresentKHR,
	// to ensure the semaphores are in a consistent state, otherwise a signalled semaphore may never be properly waited upon.
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
	{
		std::cout << "Swap chain out of date or suboptimal - recreation in progress\n";
		recreateSwapChain();
	}
	else
	{
		assert(result == VK_SUCCESS);
	}

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;


}

void HelloTriangleApplication::mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		if (minimized)
		{
			glfwWaitEvents();
		}
		else
		{
			glfwPollEvents();
			drawFrame();
		}

	}

}

void HelloTriangleApplication::cleanup()
{
	std::cout << "Vulkan shutting down...\n";

	// Wait for the device to be idle to ensure I don't destroy resources while queues have pending work.
	vkDeviceWaitIdle(device);

	cleanupSwapChain();

	vkDestroySampler(device, textureSampler, nullptr);

	vkDestroyImageView(device, textureImageView, nullptr);

	vkDestroyImage(device, textureImage, nullptr);
	vkFreeMemory(device, textureImageMemory, nullptr);

	vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

	vkDestroyBuffer(device, indexBuffer, nullptr);
	vkFreeMemory(device, indexBufferMemory, nullptr);

	vkDestroyBuffer(device, vertexBuffer, nullptr);
	vkFreeMemory(device, vertexBufferMemory, nullptr);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
		vkDestroyFence(device, inFlightFences[i], nullptr);
	}

	vkDestroyCommandPool(device, commandPool, nullptr);

	vkDestroySurfaceKHR(instance, surface, nullptr);

	vkDestroyDevice(device, nullptr);

	vkDestroyInstance(instance, nullptr);

	glfwDestroyWindow(window);

	glfwTerminate();
}


std::vector<char> HelloTriangleApplication::readFile(const std::string& filename)
{
	// The advantage of starting to read at the end of the file is that we can use the read position to determine the size of the file and allocate a buffer
	// Only works with binary flag to avoid text transformations.
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	assert(file.is_open());

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	// After that, we can seek back to the beginning of the file and read all of the bytes at once:
	file.seekg(0);
	file.read(buffer.data(), fileSize);

	return buffer;
}


void HelloTriangleApplication::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));

	if (width == 0 && height == 0) // Minimized ? Stop everything and wait to be given a good size...
	{
		app->minimized = true;
	}
	else
	{
		app->framebufferResized = true;
		app->minimized = false;
	}

}


void HelloTriangleApplication::createInstance(const char** glfwExtensions, uint32_t glfwExtensionCount)
{
	VkApplicationInfo appInfo{};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Hello Triangle";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	std::vector<const char*> extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	// Use a debug messenger specifically to be able to trace errors during the vkCreateInstance and vkDestroyInstance calls.
	// Placed it outside the if statement to ensure that it is not destroyed before the vkCreateInstance call.
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
	if (enableValidationLayers)
	{
		checkValidationLayerSupport();
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();

		populateDebugMessengerCreateInfo(debugCreateInfo);
		createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		// It will be cleaned up on vkDestroyInstance
	}
	else
	{
		createInfo.enabledLayerCount = 0;
		createInfo.ppEnabledLayerNames = nullptr; //TODO : make PR to Vulkan Tutorial for this one
	}


	VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
	assert(result == VK_SUCCESS);

}


void HelloTriangleApplication::checkExtensions(const char** glfwExtensions, uint32_t glfwExtensionCount)
{
	uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

	std::cout << "available extensions:\n";

	bool allNeededExtensionsAvailable = true; // start optimistic
	std::vector<const char*> glfwExtensionsVec(glfwExtensionCount);
	for (auto iExt = 0u; iExt < glfwExtensionCount; iExt++)
	{
		glfwExtensionsVec[iExt] = glfwExtensions[iExt];
	}

	// display available extensions and make sure all GLFW required ones are present
	for (const auto& extension : extensions)
	{
		std::cout << '\t' << extension.extensionName << '\n';
		for (int iGlfwExt = 0; iGlfwExt < glfwExtensionsVec.size(); iGlfwExt++)
		{
			if (strcmp(extension.extensionName, glfwExtensionsVec[iGlfwExt]) == 0)
			{
				// swap-with-last erase trick
				std::iter_swap(glfwExtensionsVec.begin() + iGlfwExt, glfwExtensionsVec.end() - 1);
				glfwExtensionsVec.pop_back();
				if (glfwExtensionsVec.empty())
					break;
			}
		}

		if (glfwExtensionsVec.empty())
			break;
	}

	assert(glfwExtensionsVec.empty());

	std::cout << "All GLFW extensions are supported!\n";

}

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
