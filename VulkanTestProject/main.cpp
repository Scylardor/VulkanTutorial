#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <map>
#include <optional>
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


	void createInstance(const char** glfwExtensions, uint32_t glfwExtensionCount);


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
		std::optional<uint32_t> graphicsFamily = 0;


		bool isComplete()
		{
			return graphicsFamily.has_value();
		}
	};

	QueueFamilyIndices  findQueueFamilies(VkPhysicalDevice bestCandidate);
	int rateDeviceSuitability(VkPhysicalDevice device);


	void pickPhysicalDevice();

	void createLogicalDevice();

	void initVulkan() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		checkExtensions(glfwExtensions, glfwExtensionCount);
		createInstance(glfwExtensions, glfwExtensionCount);

		// Don't need this one anymore but leaving it on the code for reference
		//setupVkDebugMessenger();

		pickPhysicalDevice();

		createLogicalDevice();


	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
		}
	}

	void cleanup() {

		std::cout << "Vulkan shutting down...\n";

		//if constexpr (enableValidationLayers)
		//{
		//	//DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		//}

		vkDestroyDevice(device, nullptr);

		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	GLFWwindow* window;

	VkInstance  instance;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue graphicsQueue;



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

		if (indices.isComplete())
			break;

		iFam++;
	}

	return indices;
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

	// Then we want to make sure our device also has all the right queues for our needs
	QueueFamilyIndices familyIndices = findQueueFamilies(device);
	if (false == familyIndices.isComplete())
		score = 0;

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

	VkDeviceQueueCreateInfo queueCreateInfo{};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
	queueCreateInfo.queueCount = 1;

	float queuePriority = 1.0f;
	queueCreateInfo.pQueuePriorities = &queuePriority;

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
	createInfo.pQueueCreateInfos = &queueCreateInfo;
	createInfo.queueCreateInfoCount = 1;

	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = 0;

	createInfo.enabledLayerCount = 0;
	if constexpr (enableValidationLayers)
	{
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}

	bool ok = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device);
	assert(ok == VK_SUCCESS);

	vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);

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
