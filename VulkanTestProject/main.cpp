#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
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
    void createInstance();


    void initVulkan() {
        createInstance();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {

        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    GLFWwindow* window;

    VkInstance  instance;

};


void HelloTriangleApplication::initWindow()
{
	int ok = glfwInit();
    assert(ok == GLFW_TRUE);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);


    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);


}

void HelloTriangleApplication::createInstance()
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

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr; //TODO : make PR to Vulkan Tutorial for this one

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    assert(result == VK_SUCCESS);

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::cout << "available extensions:\n";

    bool allNeededExtensionsAvailable = true; // start optimistic
    std::vector<const char*> glfwExtensionsVec(glfwExtensionCount);
	for (int iExt = 0; iExt < glfwExtensionCount; iExt++)
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
