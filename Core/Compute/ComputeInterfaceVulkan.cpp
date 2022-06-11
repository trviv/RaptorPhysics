#include "ComputeInterface.h"
#include <unordered_set>
#include "DeviceArray.h"

#ifdef USE_VULKAN_COMPUTE

const char* getStatusMessage(ComputeStatus status)
{
  switch (status)
  {
#define STR(r) case VK_ ##r: return #r
    STR(NOT_READY);
    STR(TIMEOUT);
    STR(EVENT_SET);
    STR(EVENT_RESET);
    STR(INCOMPLETE);
    STR(ERROR_OUT_OF_HOST_MEMORY);
    STR(ERROR_OUT_OF_DEVICE_MEMORY);
    STR(ERROR_INITIALIZATION_FAILED);
    STR(ERROR_DEVICE_LOST);
    STR(ERROR_MEMORY_MAP_FAILED);
    STR(ERROR_LAYER_NOT_PRESENT);
    STR(ERROR_EXTENSION_NOT_PRESENT);
    STR(ERROR_FEATURE_NOT_PRESENT);
    STR(ERROR_INCOMPATIBLE_DRIVER);
    STR(ERROR_TOO_MANY_OBJECTS);
    STR(ERROR_FORMAT_NOT_SUPPORTED);
    STR(ERROR_FRAGMENTED_POOL);
    STR(ERROR_UNKNOWN);
    STR(ERROR_OUT_OF_POOL_MEMORY);
    STR(ERROR_INVALID_EXTERNAL_HANDLE);
    STR(ERROR_FRAGMENTATION);
    STR(ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
    STR(PIPELINE_COMPILE_REQUIRED);
    STR(ERROR_SURFACE_LOST_KHR);
    STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
    STR(SUBOPTIMAL_KHR);
    STR(ERROR_OUT_OF_DATE_KHR);
    STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
    STR(ERROR_VALIDATION_FAILED_EXT);
    STR(ERROR_INVALID_SHADER_NV);
    STR(ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
    STR(ERROR_NOT_PERMITTED_KHR);
    STR(ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
    STR(THREAD_IDLE_KHR);
    STR(THREAD_DONE_KHR);
    STR(OPERATION_DEFERRED_KHR);
//    STR(OPERATION_NOT_DEFERRED_KHR);
//    STR(ERROR_OUT_OF_POOL_MEMORY_KHR);
//    STR(ERROR_INVALID_EXTERNAL_HANDLE_KHR);
//    STR(ERROR_FRAGMENTATION_EXT);
//    STR(ERROR_NOT_PERMITTED_EXT);
//    STR(ERROR_INVALID_DEVICE_ADDRESS_EXT);
//    STR(ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR);
//    STR(PIPELINE_COMPILE_REQUIRED_EXT);
//    STR(ERROR_PIPELINE_COMPILE_REQUIRED_EXT);
    STR(RESULT_MAX_ENUM);
#undef STR
  default:
    return "UNKNOWN_ERROR";
  }
  return "No status";
}


ComputeMemory::ComputeMemory(ComputeMemoryIdentifier ref, ComputeBufferIdentifier buf, size_t offset, size_t size)
{
  this->ref = ref;
  this->buf = buf;
  this->offset = offset;
  this->size = size;
}


ComputeMemory* ComputeHeap::alloc(size_t sizeInBytes, void* data, ComputeMemoryFlag flag)
{
  sizeInBytes = mAlignBy(sizeInBytes, 16) * 16;
  ComputeMemory* ret;

  if (bypass)
  {
#pragma mark -
#pragma mark CreateBufferHandle

    VkBufferCreateInfo bufferCreateInfo {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.usage = flag | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.size  = sizeInBytes;
//    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    ComputeBufferIdentifier buffer {};
    getStatusMessage(vkCreateBuffer(compute->getDevice(), &bufferCreateInfo, nullptr, &buffer));

#pragma mark -
#pragma mark AllocateMemory

    VkMemoryPropertyFlags memoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkMemoryRequirements memoryReqs;
    vkGetBufferMemoryRequirements(compute->getDevice(), buffer, &memoryReqs);
    sizeInBytes = memoryReqs.size;

    VkMemoryAllocateInfo memoryAllocInfo {};
    memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocInfo.allocationSize = sizeInBytes;
    memoryAllocInfo.memoryTypeIndex = compute->getMemoryType(memoryReqs.memoryTypeBits, memoryPropertyFlags);

    VkMemoryAllocateFlagsInfoKHR allocFlagsInfo {};
    if (flag & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
    {
      allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
      allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
      memoryAllocInfo.pNext = &allocFlagsInfo;
    }

    VkDeviceMemory memory;
    checkError(vkAllocateMemory(compute->getDevice(), &memoryAllocInfo, nullptr, &memory));
    checkError(vkBindBufferMemory(compute->getDevice(), buffer, memory, 0));
    ret = new ComputeMemory(memory, buffer, 0, sizeInBytes);
  }
  else
  {
#pragma mark -
#pragma mark CreateBufferHandle

    size_t offset = 0;
    if (childs.size())
    {
      const ComputeMemory* last = childs.back();
      offset = last->getOffset() + last->getSize();
    }

    ret = new ComputeMemory(*heap, offset, sizeInBytes);
  }

  childs.push_back(ret);

  return ret;
}

void ComputeHeap::free(ComputeMemory* memory)
{
  memory->ref = NULL;

  vector<ComputeMemory*>::iterator pos = find(childs.begin(), childs.end(), memory);
  if (pos != childs.end())
  {
    childs.erase(pos);
  }
  else if (memory != heap) // not during destructor
  {
    logComputeError("Trying to free outside memory!");
  }
}


ComputeKernel::ComputeKernel(const ComputeKernel& ref)
{
  this->ref = ref.ref;
  this->compute = ref.compute;

  this->constantData = ref.constantData;
  this->shaderModule = ref.shaderModule;
  this->descriptorSetLayout = ref.descriptorSetLayout;
  this->descriptorSet = ref.descriptorSet;
  this->pipelineLayout = ref.pipelineLayout;
  this->computeWriteDescriptorSets = ref.computeWriteDescriptorSets;
  this->descriptorBufferInfos = ref.descriptorBufferInfos;
  this->name = ref.name;

  this->args = ref.args;
  this->setArgumentBuffer = ref.setArgumentBuffer;
  this->argumentBufferRange = ref.argumentBufferRange;
  this->argumentBuffers = ref.argumentBuffers;
  this->usePushConstants = false;
}

ComputeKernel::ComputeKernel(ComputeKernelIdentifier ref, ComputeInterface* compute)
{
  this->ref = ref;
  this->compute = compute;
  this->setArgumentBuffer = false;
  this->constantData = NULL;
  this->shaderModule = {};
  this->descriptorSetLayout = {};
  this->descriptorSet = {};
  this->pipelineLayout = {};
  this->usePushConstants = false;
}

ComputeKernel::~ComputeKernel()
{
  if (constantData)
  {
    delete ((DeviceArray<Byte>*)constantData);
    constantData = NULL;
  }
}

void ComputeKernel::createPipeline()
{
#pragma mark -
#pragma mark CreateKernelPipelineLayout

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo {};
  pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.setLayoutCount = 1;
  pipelineLayoutCreateInfo.pSetLayouts    = &descriptorSetLayout;

  if (pushConstantRanges.size() > 0)
  {
    pipelineLayoutCreateInfo.pushConstantRangeCount = pushConstantRanges.size();
    pipelineLayoutCreateInfo.pPushConstantRanges    = pushConstantRanges.data();
  }

  checkError(vkCreatePipelineLayout(compute->getDevice(), &pipelineLayoutCreateInfo, nullptr,  &pipelineLayout));

#pragma mark -
#pragma mark CreateKernelPipeline

  VkPipelineShaderStageCreateInfo shaderStage {};
  shaderStage.sType   = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStage.stage   = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStage.module  = this->shaderModule;
  shaderStage.pName   = name.c_str();

  VkComputePipelineCreateInfo computePipelineCreateInfo {};
  computePipelineCreateInfo.sType   = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.layout  = pipelineLayout;
  computePipelineCreateInfo.flags   = 0;
  computePipelineCreateInfo.stage   = shaderStage;

  ComputeKernelIdentifier& pipeline = ref;
  checkError(vkCreateComputePipelines(compute->getDevice(), compute->pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));
}

void ComputeKernel::addArgumentBufferRange(uint startIndex, uint inclusiveEndIndex)
{}

void ComputeKernel::setArg(void* valuePtr, size_t valueSize, uint index)
{
  args.push_back({ARG_DATA_HOST_PTR, (uint)valueSize, index, valuePtr});
}

void ComputeKernel::setArg(const void* valuePtr, size_t valueSize, uint index)
{
  args.push_back({ARG_DATA_CONST_HOST_PTR, (uint)valueSize, index, valuePtr});
}

void ComputeKernel::setArg(ComputeMemory* buffer, uint index)
{
  args.push_back({ARG_DATA_DEVICE_PTR, 0, index, buffer});
}

void ComputeKernel::setArg(const ComputeMemory* buffer, uint index)
{
  args.push_back({ARG_DATA_CONST_DEVICE_PTR, 0, index, buffer});
}

void ComputeKernel::setSharedMemArg(const size_t valueSize, uint index)
{
  args.push_back({ARG_DATA_CONST_HOST_PTR, (uint)valueSize, index, NULL});
}

void ComputeKernel::setArgs(VkCommandBuffer commandBuffer)
{
  pushConstantRanges.clear();

  if (!constantData)
  {
    constantData = new DeviceArray<Byte>(compute, NULL, true);
  }

  DeviceArray<Byte>* constantDeviceData = (DeviceArray<Byte>*)constantData;

  bool setDescriptorBufferInfos = (descriptorBufferInfos.size() == 0);

  uint constantOffset = 0;
  const uint constantDataAlignment = 32;

  for (int i=0; i<args.size(); i++)
  {
    const auto& arg = args[i];
    switch (arg.type)
    {
      case ARG_DATA_HOST_PTR:
      case ARG_DATA_CONST_HOST_PTR:
      case ARG_DATA_SHARED_PTR:
        if (!usePushConstants)
        {
          if (setDescriptorBufferInfos)
          {
            VkDescriptorBufferInfo descriptorBufferInfo {};
            descriptorBufferInfo.offset = constantOffset;
            descriptorBufferInfo.range = mAlignBy(arg.size, constantDataAlignment) * constantDataAlignment;
            descriptorBufferInfos.push_back(descriptorBufferInfo);
          }
          constantOffset += mAlignBy(arg.size, constantDataAlignment) * constantDataAlignment;
        }
        else
        {
          VkPushConstantRange pushConstantRange {};
          pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; //VK_SHADER_STAGE_ALL
          pushConstantRange.offset = constantOffset;
          pushConstantRange.size = arg.size;
          pushConstantRanges.push_back(pushConstantRange);

          constantOffset += pushConstantRange.size;
        }
        break;
      case ARG_DATA_DEVICE_PTR:
      case ARG_DATA_CONST_DEVICE_PTR:
        if (usePushConstants) constantOffset += 8;
        break;
      default:
        logComputeError("Unknown Argument type!");
        break;
    }
  }

  if (!usePushConstants && constantOffset > constantDeviceData->host()->size())
  {
    constantDeviceData->resize(constantOffset, false);
    constantDeviceData->host()->resize(constantOffset);
  }

  if (ref == VK_NULL_HANDLE)
  {
    createPipeline();
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)*this);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

  auto& constantHostData = *constantDeviceData->host();
  uint constantDeviceDataOffset = 0;
  uint constantCounter = 0;
  bool updateConstantDeviceData = false;

  for (uint i=0; i<args.size(); i++)
  {
    const auto &arg = args[i];
    switch (arg.type)
    {
      case ARG_DATA_HOST_PTR:
      case ARG_DATA_CONST_HOST_PTR:
      case ARG_DATA_SHARED_PTR:
        if (!usePushConstants)
        {
          if (memcmp(constantHostData.data() + constantDeviceDataOffset, arg.cptr, arg.size) != 0)
          {
            memcpy(constantHostData.data() + constantDeviceDataOffset, arg.cptr, arg.size);
            updateConstantDeviceData = true;
          }
          descriptorBufferInfos[constantCounter].buffer = *constantDeviceData->device();
          computeWriteDescriptorSets[i].pBufferInfo = &descriptorBufferInfos[constantCounter++];
          constantDeviceDataOffset += mAlignBy(arg.size, constantDataAlignment) * constantDataAlignment;
        }
        else
        {
          vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, constantDeviceDataOffset, arg.size, arg.cptr);
          constantDeviceDataOffset += arg.size;
        }
        break;
      case ARG_DATA_DEVICE_PTR:
      case ARG_DATA_CONST_DEVICE_PTR:
      {
        computeWriteDescriptorSets[i].pBufferInfo = ((const ComputeMemory*)arg.cptr)->getDescriptorBufferInfo();
        if (usePushConstants) constantDeviceDataOffset += 8;
      }
        break;
    }
  }

  if (updateConstantDeviceData)
  {
    constantDeviceData->syncDevice();
  }

  vkUpdateDescriptorSets(compute->getDevice(), (uint32_t)computeWriteDescriptorSets.size(), computeWriteDescriptorSets.data(), 0, nullptr);

  args.clear();
}

void ComputeKernel::registerResource(const ComputeMemory* resource)
{}


ComputeProgram::ComputeProgram()
  : ref(NULL), compute(NULL), kernelArgs({}), descriptorPool(NULL)
{}

ComputeProgram::ComputeProgram(ComputeProgramIdentifier ref, ComputeInterface* compute, const vector<FunctionArgs>& kernelArgs)
  : ref(ref), compute(compute), kernelArgs(kernelArgs), descriptorPool(NULL)
{}

ComputeProgram::ComputeProgram(const ComputeProgram& ref)
  : ref(ref.ref), compute(ref.compute), kernelArgs(ref.kernelArgs), descriptorPool(ref.descriptorPool)
{}

ComputeKernel ComputeProgram::createKernel(const char* kernelName)
{
#pragma mark -
#pragma mark CreateKernelDescriptorSetLayoutBinding

  const FunctionArgs* currentKernelArgs = NULL;

  for (const auto& kernelArgs : this->kernelArgs)
  {
    if (kernelArgs.funcName == kernelName)
    {
      currentKernelArgs = &kernelArgs;
      break;
    }
  }

  if (!currentKernelArgs) logComputeError("Kernel %s not in the current program!", kernelName);

  ComputeKernel computeKernel;

  computeKernel.compute       = compute;
  computeKernel.name          = kernelName;
  computeKernel.shaderModule  = (ComputeProgramIdentifier)*this;

  VkDescriptorSetLayout& descriptorSetLayout                = computeKernel.descriptorSetLayout;
  VkDescriptorSet& descriptorSet                            = computeKernel.descriptorSet;
  vector<VkWriteDescriptorSet>& computeWriteDescriptorSets  = computeKernel.computeWriteDescriptorSets;

  vector<VkDescriptorPoolSize> descriptorPoolSizes
  {
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_SAMPLER,                 0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,           0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,    0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,    0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,  0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,  0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,        0},
    VkDescriptorPoolSize {VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK,    0},
  };

  vector<VkDescriptorSetLayoutBinding> setLayoutBindings;

  for (int index = 0; index < currentKernelArgs->args.size(); index++)
  {
    const auto& kernelArg = currentKernelArgs->args[index];

    VkDescriptorSetLayoutBinding setLayoutBinding {};
    setLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    setLayoutBinding.binding    = index;

    if (kernelArg.find("constantKernelInput") != kernelArg.npos)
    {
      if (computeKernel.usePushConstants) continue;

      setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      descriptorPoolSizes[setLayoutBinding.descriptorType].descriptorCount++;
    }
    else if (kernelArg.find("atomicKernelInput") != kernelArg.npos)
    {
      setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      descriptorPoolSizes[setLayoutBinding.descriptorType].descriptorCount++;
    }
    else if (kernelArg.find("sharedMemKernelInput") != kernelArg.npos)
    {
      if (computeKernel.usePushConstants) continue;

      setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      descriptorPoolSizes[setLayoutBinding.descriptorType].descriptorCount++;
    }
    else if (kernelArg.find("Device ") != kernelArg.npos)
    {
      setLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      descriptorPoolSizes[setLayoutBinding.descriptorType].descriptorCount++;
    }
    else if (kernelArg.find("KERNEL_GLOBAL_ARGUMENTS") != kernelArg.npos ||
             kernelArg.find("KERNEL_THREAD_ARGUMENTS") != kernelArg.npos ||
             kernelArg.find("KERNEL_THREADGROUP_ARGUMENTS") != kernelArg.npos)
    {
      continue;
    }
    else
    {
      logComputeError("Kernel argument %s not handled!", kernelArg.c_str());
    }
    setLayoutBinding.descriptorCount = 1;

    setLayoutBindings.push_back(setLayoutBinding);
  }

  for (int i = descriptorPoolSizes.size() - 1; i >= 0; i--)
  {
    if (descriptorPoolSizes[i].descriptorCount != 0) continue;

    descriptorPoolSizes.erase(descriptorPoolSizes.begin()+i);
  }

  VkDescriptorPoolCreateInfo descriptorPoolInfo {};
  descriptorPoolInfo.sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolInfo.poolSizeCount  = descriptorPoolSizes.size();
  descriptorPoolInfo.pPoolSizes     = descriptorPoolSizes.data();
  descriptorPoolInfo.flags          = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  descriptorPoolInfo.maxSets        = 1;

  checkError(vkCreateDescriptorPool(compute->getDevice(), &descriptorPoolInfo, nullptr, &this->descriptorPool));

#pragma mark -
#pragma mark CreateKernelWriteDescriptorSet

  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo {};
  descriptorSetLayoutCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.pBindings     = setLayoutBindings.data();
  descriptorSetLayoutCreateInfo.bindingCount  = setLayoutBindings.size();

  checkError(vkCreateDescriptorSetLayout(compute->getDevice(),  &descriptorSetLayoutCreateInfo, nullptr,  &descriptorSetLayout));

  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo {};
  descriptorSetAllocateInfo.sType               = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.descriptorPool      = this->descriptorPool;
  descriptorSetAllocateInfo.pSetLayouts         = &descriptorSetLayout;
  descriptorSetAllocateInfo.descriptorSetCount  = 1;

  checkError(vkAllocateDescriptorSets(compute->getDevice(), &descriptorSetAllocateInfo, &descriptorSet));

  computeWriteDescriptorSets.resize(setLayoutBindings.size());

  for (int index = 0; index < setLayoutBindings.size(); index++)
  {
    VkWriteDescriptorSet writeDescriptorSet {};
    writeDescriptorSet.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet           = descriptorSet;
    writeDescriptorSet.descriptorType   = setLayoutBindings[index].descriptorType;
    writeDescriptorSet.dstBinding       = index;
    writeDescriptorSet.descriptorCount  = 1;

    computeWriteDescriptorSets[index] = writeDescriptorSet;
  }

  return computeKernel;
}

bool ComputeProgram::isEmpty()const
{
  return ref == NULL;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData)
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger)
{
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
  vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
  {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  }
  else
  {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
  {
    func(instance, debugMessenger, pAllocator);
  }
}

ComputeInterface::ComputeInterface()
  :heap(this, true)
{
  deviceId = NULL;
  context = NULL;
  queue = NULL;
  commandBufferRecording = false;
}

ComputeInterface::~ComputeInterface()
{
  vkDeviceWaitIdle(deviceId);
  vkDestroyDevice(deviceId, nullptr);
  DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
  vkDestroyInstance(instance, nullptr);

  if (deviceId)
  {

  }
  if (context)
  {

  }
  if (queue)
  {

  }
}

uint ComputeInterface::getMemoryType(uint typeBits, const VkMemoryPropertyFlags properties)const
{
  for (uint i = 0; i < memoryProperties.memoryTypeCount; i++)
  {
    if ((typeBits & 1) == 1 && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
    typeBits >>= 1;
  }

  throw runtime_error("Could not find a matching memory type");
}

void ComputeInterface::create(int deviceIndex)
{
  const bool validation = true;
//  const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
  const char* validationLayerName = "MoltenVK";
  stringArr enabledInstanceExtensions;

#pragma mark -
#pragma mark CreateApplicationInfo

  VkApplicationInfo appInfo {};
  appInfo.sType             = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName  = "Particle Physics";
  appInfo.applicationVersion= 1;
  appInfo.pEngineName       = "Particle Physics";
  appInfo.engineVersion     = VK_MAKE_VERSION(1, 1, 0);
  appInfo.apiVersion        = VK_API_VERSION_1_1;

#pragma mark -
#pragma mark HandleVulkanExtensions

  vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };
  stringArr supportedInstanceExtensions;

  uint32_t extensionCount = 0;
  checkError(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr));
  if (extensionCount > 0)
  {
    vector<VkExtensionProperties> extensions(extensionCount);
    checkError(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, &extensions.front()));
    for (const VkExtensionProperties& extension : extensions)
    {
      supportedInstanceExtensions.push_back(extension.extensionName);
    }
  }

  // Enabled requested instance extensions
  if (enabledInstanceExtensions.size() > 0)
  {
    for (const string& enabledExtension : enabledInstanceExtensions)
    {
      // Output message if requested extension is not available
      if (find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
      {
        logComputeError("Enabled instance extension \"%s\" is not present at instance level", enabledExtension.c_str());
      }
      instanceExtensions.push_back(enabledExtension.c_str());
    }
  }
  else
  {
    for (const string& supportedExtension : supportedInstanceExtensions)
    {
      instanceExtensions.push_back(supportedExtension.c_str());
    }
  }

  VkDebugReportCallbackCreateInfoEXT debugReport {};
  debugReport.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT;

  VkInstanceCreateInfo instanceCreateInfo {};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = &debugReport;
  instanceCreateInfo.pApplicationInfo = &appInfo;
  if (instanceExtensions.size() > 0)
  {
    if (validation)
    {
      instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
  }

#pragma mark -
#pragma mark HandleVulkanValidation

  // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
  // Note that on Android this layer requires at least NDK r20
  if (validation)
  {
    // Check if this layer is available at instance level
    uint32_t instanceLayerCount;
    checkError(vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr));

    vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
    checkError(vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data()));

    bool validationLayerPresent = false;
    for (const VkLayerProperties& layer : instanceLayerProperties)
    {
      if (strcmp(layer.layerName, validationLayerName) == 0)
      {
        validationLayerPresent = true;
        break;
      }
    }

    if (validationLayerPresent)
    {
      instanceCreateInfo.ppEnabledLayerNames = &validationLayerName;
      instanceCreateInfo.enabledLayerCount   = 1;
    }
    else
    {
      logComputeError("Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled");
    }
  }

#pragma mark -
#pragma mark CreateVulkanInstance

  instance = NULL;
  checkError(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

  VkDebugUtilsMessengerCreateInfoEXT debugMsgCreateInfo{};
  debugMsgCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  debugMsgCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  debugMsgCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  debugMsgCreateInfo.pfnUserCallback = debugCallback;
  debugMsgCreateInfo.pUserData = nullptr;

  checkError(CreateDebugUtilsMessengerEXT(instance, &debugMsgCreateInfo, nullptr, &debugMessenger));

  deviceCount = 0;
  checkError(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
  if (deviceCount == 0)
  {
    logComputeError("No device with Vulkan support found");
  }

#pragma mark -
#pragma mark SelectPhysicalDevice

  devices.clear();
  devices.resize(deviceCount);
  checkError(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()));

  int selectedDeviceIndex = (deviceIndex == -1) ? 0 : deviceIndex;

  for (uint j = 0; j < deviceCount && (deviceIndex == -1); j++)
  {
    VkPhysicalDeviceProperties deviceProperties {};
    vkGetPhysicalDeviceProperties(devices[j], &deviceProperties);
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
      selectedDeviceIndex = j;
      break;
    }
  }

#pragma mark -
#pragma mark GetPhysicalDeviceProperties

  const VkPhysicalDevice selectedDevice = devices[selectedDeviceIndex];

  VkPhysicalDeviceProperties deviceProperties {};
  vkGetPhysicalDeviceProperties(selectedDevice, &deviceProperties);

  const char *deviceName = deviceProperties.deviceName;
  const char *deviceVendor = deviceProperties.deviceName;
  const uint32_t *maxWGSizes = deviceProperties.limits.maxComputeWorkGroupSize;

  const size_t maxWorkgroupSize = max(max(maxWGSizes[0], maxWGSizes[1]), maxWGSizes[2]);

  logComputeMessage("Device Name:        %s", deviceName);
  logComputeMessage("Max Workgroup Size: %d", maxWorkgroupSize);

  simdGroupSize = 16;
  maxThreadsPerWorkgroup = maxWorkgroupSize;
  string selectedPhysicalDeviceName = deviceName;

  if (selectedPhysicalDeviceName.find("AMD") != selectedPhysicalDeviceName.npos)
  {
    simdGroupSize = 64;
  }
  else if (selectedPhysicalDeviceName.find("NVIDIA") != selectedPhysicalDeviceName.npos)
  {
    simdGroupSize = 32;
  }
  else if (selectedPhysicalDeviceName.find("Apple") != selectedPhysicalDeviceName.npos)
  {
    simdGroupSize = 32;
  }

  logComputeMessage("Selected device:   %s\nAssumed SIMD size: %ld", selectedPhysicalDeviceName.c_str(), simdGroupSize);

  vkGetPhysicalDeviceFeatures(selectedDevice, &enabledFeatures);
  vkGetPhysicalDeviceProperties(selectedDevice, &properties);
  vkGetPhysicalDeviceMemoryProperties(selectedDevice, &memoryProperties);

#pragma mark -
#pragma mark CreateDevice

  uint queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(selectedDevice, &queueFamilyCount, nullptr);

  vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(selectedDevice, &queueFamilyCount, queueFamilies.data());

  vector<VkDeviceQueueCreateInfo> queueCreateInfos {};
  const float defaultQueuePriority(0.0f);
  const int queueFamilyIndex = 0;

  VkDeviceQueueCreateInfo queueInfo {};
  queueInfo.sType             = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.queueFamilyIndex  = queueFamilyIndex;
  queueInfo.queueCount        = 1;
  queueInfo.pQueuePriorities  = &defaultQueuePriority;
  queueCreateInfos.push_back(queueInfo);

  VkDeviceCreateInfo deviceCreateInfo {};
  deviceCreateInfo.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
  deviceCreateInfo.pQueueCreateInfos    = queueCreateInfos.data();
  deviceCreateInfo.pEnabledFeatures     = &enabledFeatures;
  instanceCreateInfo.ppEnabledLayerNames= &validationLayerName;
  instanceCreateInfo.enabledLayerCount  = 1;
  checkError(vkCreateDevice(selectedDevice, &deviceCreateInfo, nullptr, &deviceId));

#pragma mark -
#pragma mark CreateCommandBufferPool

  vkGetDeviceQueue(this->getDevice(), queueFamilyIndex, 0, &queue);

  VkCommandPoolCreateInfo commandPoolInfo {};
  commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  commandPoolInfo.queueFamilyIndex = queueFamilyIndex;
  checkError(vkCreateCommandPool(this->getDevice(), &commandPoolInfo, nullptr, &commandPool));

  VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
  pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  checkError(vkCreatePipelineCache(this->getDevice(), &pipelineCacheCreateInfo, nullptr, &pipelineCache));

//  VkSemaphoreCreateInfo semaphoreCreateInfo {};
//  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
//  VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));

  VkFenceCreateInfo fenceCreateInfo {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  checkError(vkCreateFence(this->getDevice(), &fenceCreateInfo, nullptr, &computeFence));
}

ComputeCommandBuffer ComputeInterface::getComputeCommandBuffer()
{
  if (!commandBufferRecording)
  {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.commandBufferCount = 1;

    checkError(vkAllocateCommandBuffers(this->getDevice(), &commandBufferAllocateInfo, &currentCommandBuffer));

    VkCommandBufferBeginInfo commandBufferBeginInfo {};
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.flags = 0;
    commandBufferBeginInfo.pInheritanceInfo = nullptr;

    checkError(vkBeginCommandBuffer(currentCommandBuffer, &commandBufferBeginInfo));

    commandBufferRecording = true;
  }

  return currentCommandBuffer;
}

ComputeProgram ComputeInterface::createProgram(const char* sourceCode, size_t sourceSize,
                                               const stringArr* oldType, const stringArr* newType,
                                               const stringArr* includeFiles)
{
  stringArr localOldType = oldType ? *oldType : stringArr();
  stringArr localNewType = newType ? *newType : stringArr();

  string defines;
  for (uint i = 0; i < localOldType.size(); i++)
  {
    defines += "#define " + localOldType[i] + " " + localNewType[i] + "\n";
  }

  string finalSource = defines + deepReadShaderSource(sourceCode, includeFiles);

#ifdef USE_ARGUMENT_BUFFERS
  finalSource = processAutoArgumentBuffers(finalSource, kernelNameArgumentBufferMap);
#endif

//  const stringArr parts = convertToHLSL(finalSource, true);
  const auto kernelArgs = getKernelArgs(finalSource);

//  string command;
//
//  command = "rm " + IOInterface::getPath("temp.cl") + " " + IOInterface::getPath("temp.spv");
//  system(command.c_str());
//
//  IOInterface::writeFile("temp.cl", finalSource.c_str(), finalSource.size());
//
////  command = "clang -c -target spir64 -O0 -emit-llvm -o " + IOInterface::getPath("temp.bc") + " " + IOInterface::getPath("temp.cl");
//  command = "clang -c -target spir64 -emit-llvm -g0 -O0 -o " + IOInterface::getPath("temp.bc") + " " + IOInterface::getPath("temp.cl");
//  system(command.c_str());
//
//  command = IOInterface::getPath("llvm-spirv") + " " + IOInterface::getPath("temp.bc") + " -o " + IOInterface::getPath("temp.spv");
//  system(command.c_str());

//  IOInterface::writeFile("temp.spv", finalSource.c_str(), finalSource.size());
//
//  vector<char> fileData = IOInterface::readByteFile("temp.spv");

  string header;
  header.resize(4, 0);
  *((uint*)&header.at(0)) = 0x19960412; // spirv MSL header
  finalSource = header + finalSource;

  VkShaderModuleCreateInfo createInfo {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = finalSource.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(finalSource.data());

  ComputeProgramIdentifier shaderModule;
  checkError(vkCreateShaderModule(this->getDevice(), &createInfo, nullptr, &shaderModule));

  ComputeProgram program = ComputeProgram(shaderModule, this, kernelArgs);

  return program;
}

void ComputeInterface::copyBuffer(const ComputeMemory* source, ComputeMemory* destination, size_t sourceOffset, size_t destinationOffset, size_t sizeInBytes)
{
  VkBufferCopy region;
  region.srcOffset = (sourceOffset + source->getOffset());
  region.dstOffset = (destinationOffset + destination->getOffset());
  region.size      = sizeInBytes;
  vkCmdCopyBuffer(getComputeCommandBuffer(), *source, *destination, 1, &region);
}

void ComputeInterface::copyTextureToBuffer(const ComputeTexture* source, ComputeMemory* destination, size_t destinationOffset, size_t sourceSlice, size_t sourceLevel)
{
  logComputeError("Function copyTextureToBuffer not implemented on Vulkan");
}

void ComputeInterface::copyBufferToTexture(const ComputeMemory* source, ComputeTexture* destination, size_t sourceOffset, size_t destinationSlice, size_t destinationLevel)
{
  logComputeError("Function copyBufferToTexture not implemented on Vulkan");
}

void ComputeInterface::copyTexture(const ComputeTexture* source, ComputeTexture* destination)
{
  logComputeError("Function copyTexture not implemented on Vulkan");
}

void ComputeInterface::setBuffer(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, const void* hostValue, size_t hostValueSize)
{
  if (hostValueSize != 4)
  {
    logComputeError("Function setBuffer is only allowed for 4 bytes!");
  }

  void* data;
  checkError(vkMapMemory(this->getDevice(), (ComputeMemoryIdentifier)*source, sourceOffset, sizeInBytes, 0, &data));
  memset(source, *((uint*)hostValue), hostValueSize);
  vkUnmapMemory(this->getDevice(), (ComputeMemoryIdentifier)*source);
}

void ComputeInterface::copyToHost(const ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, void* hostPtr, bool waitForFinish)
{
  VkBufferMemoryBarrier bufferMemoryBarrier {};
  bufferMemoryBarrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufferMemoryBarrier.srcAccessMask       = VK_ACCESS_FLAG_BITS_MAX_ENUM;
  bufferMemoryBarrier.dstAccessMask       = VK_ACCESS_FLAG_BITS_MAX_ENUM;
  bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferMemoryBarrier.buffer              = (VkBuffer)*source;
  bufferMemoryBarrier.size                = VK_WHOLE_SIZE;

  auto commandBuffer = getComputeCommandBuffer();
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       0, 0, nullptr, 1, &bufferMemoryBarrier, 0, nullptr);

  sync();

  void* data;
  checkError(vkMapMemory(this->getDevice(), (ComputeMemoryIdentifier)*source, sourceOffset, sizeInBytes, 0, &data));
  memcpy(hostPtr, ((uint8_t*)data), sizeInBytes);
  vkUnmapMemory(this->getDevice(), (ComputeMemoryIdentifier)*source);

  if (waitForFinish)
  {
    sync();
  }
}

void ComputeInterface::copyFromHost(ComputeMemory* destination, size_t destinationOffset, size_t sizeInBytes, const void* hostPtr, bool waitForFinish)
{
  void* data;
  checkError(vkMapMemory(this->getDevice(), (ComputeMemoryIdentifier)*destination, destinationOffset, sizeInBytes, 0, &data));
  memcpy(data, hostPtr, sizeInBytes);
  vkUnmapMemory(this->getDevice(), (ComputeMemoryIdentifier)*destination);

  VkBufferMemoryBarrier bufferMemoryBarrier {};
  bufferMemoryBarrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufferMemoryBarrier.srcAccessMask       = VK_ACCESS_FLAG_BITS_MAX_ENUM;
  bufferMemoryBarrier.dstAccessMask       = VK_ACCESS_FLAG_BITS_MAX_ENUM;
  bufferMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufferMemoryBarrier.buffer              = (VkBuffer)*destination;
  bufferMemoryBarrier.size                = VK_WHOLE_SIZE;

  auto commandBuffer = getComputeCommandBuffer();
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       0, 0, nullptr, 1, &bufferMemoryBarrier, 0, nullptr);

  if (waitForFinish)
  {
    sync();
  }
}

void ComputeInterface::execute(ComputeKernel& kernel, const size_t workgroupSize[3], const size_t workgroupCount[3])
{
  auto commandBuffer = getComputeCommandBuffer();

  vkSetWorkgroupSizeMVK((VkShaderModule)kernel, workgroupSize[0], workgroupSize[1], workgroupSize[2]);
  kernel.setArgs(commandBuffer);
  vkCmdDispatch(commandBuffer, workgroupCount[0], workgroupCount[1], workgroupCount[2]);
}

void ComputeInterface::execute(ComputeKernel& kernel, const size_t workgroupSize[3], const ComputeMemory* workgroupCount, size_t bufferOffset)
{
  auto commandBuffer = getComputeCommandBuffer();

  vkSetWorkgroupSizeMVK((VkShaderModule)kernel, workgroupSize[0], workgroupSize[1], workgroupSize[2]);
  kernel.setArgs(commandBuffer);
  vkCmdDispatchIndirect(commandBuffer, (VkBuffer)*workgroupCount, bufferOffset);
}

void ComputeInterface::execute(ComputeKernel& kernel, const size_t threadCount)
{
  size_t workgroupSize[3];
  size_t workgroupCount[3];
  configureSize(workgroupSize, workgroupCount, (uint)threadCount);
  execute(kernel, workgroupSize, workgroupCount);
}

void ComputeInterface::sync(bool waitOnFinish)
{
  if (!commandBufferRecording) return;

  vkEndCommandBuffer(currentCommandBuffer);

  const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

  VkSubmitInfo commandBufferSubmitInfo {};
  commandBufferSubmitInfo.sType               = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  commandBufferSubmitInfo.commandBufferCount  = 1;
  commandBufferSubmitInfo.pWaitDstStageMask   = &waitStageMask;
  commandBufferSubmitInfo.pCommandBuffers     = &currentCommandBuffer;
  checkError(vkQueueSubmit(queue, 1, &commandBufferSubmitInfo, waitOnFinish ? computeFence : VK_NULL_HANDLE));

  if (waitOnFinish)
  {
    checkError(vkWaitForFences(this->getDevice(), 1, &computeFence, VK_TRUE, UINT64_MAX));
    checkError(vkResetFences(this->getDevice(), 1, &computeFence));
    checkError(vkQueueWaitIdle(queue));
  }

//  vkResetCommandBuffer(currentCommandBuffer, 0);
  vkFreeCommandBuffers(this->getDevice(), commandPool, 1, &currentCommandBuffer);

  commandBufferRecording = false;
}

uint ComputeInterface::maxCores()const
{
  return 0;
}

void ComputeInterface::startCapture()
{
}

void ComputeInterface::endCapture()
{
}

#endif
