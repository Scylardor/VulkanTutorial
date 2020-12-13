#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <fstream>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <vector>

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

	VkSurfaceFormatKHR		chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

	VkPresentModeKHR		chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

	VkExtent2D				chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	void					createSwapChain();
	void					retrieveSwapChainImages(uint32_t imageCount);

	void					createImageViews();

	void					createGraphicsPipeline();


	VkShaderModule createShaderModule(const std::vector<char>& bytecode);

	void initVulkan();

	void mainLoop();

	void cleanup();

	static std::vector<char> readFile(const std::string& filename);

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

	VkDebugUtilsMessengerEXT debugMessenger;
};


void HelloTriangleApplication::initWindow()
{
	int ok = glfwInit();
	assert(ok == GLFW_TRUE);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);


	window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);


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
		VkImageViewCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = swapImage;

		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.format = swapChainImageFormat;

		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

		// The subresourceRange field describes what the image's purpose is
		// and which part of the image should be accessed.
		// Our images will be used as color targets without any mipmapping levels or multiple layers.
		createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = 1;

		// If you were working on a stereographic 3D application,
		// then you would create a swap chain with multiple layers.
		// You could then create multiple image views for each image
		// representing the views for the left and right eyes by accessing different layers.
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;

		VkResult ok = vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[imgIndex]);
		assert(ok == VK_SUCCESS);

		imgIndex++;
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

	createGraphicsPipeline();
}

void HelloTriangleApplication::mainLoop()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
	}
}

void HelloTriangleApplication::cleanup()
{
	std::cout << "Vulkan shutting down...\n";

	//if constexpr (enableValidationLayers)
	//{
	//	//DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	//}

	vkDestroyShaderModule(device, fragShaderModule, nullptr);
	vkDestroyShaderModule(device, vertShaderModule, nullptr);

	for (auto imageView : swapChainImageViews)
	{
		vkDestroyImageView(device, imageView, nullptr);
	}

	vkDestroySwapchainKHR(device, swapChain, nullptr);

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
