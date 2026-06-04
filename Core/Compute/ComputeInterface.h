/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef COMPUTE_INTERFACE_H
#define COMPUTE_INTERFACE_H

#include "ComputeShaderCommon.h"

#ifdef __APPLE__
  #import <Foundation/Foundation.h>
  #if defined(USE_METAL_COMPUTE)
    #import <Metal/Metal.h>
  #elif defined(USE_VULKAN_COMPUTE)
    #include <vulkan/vulkan.h>
    #include <MoltenVK/vk_mvk_moltenvk.h>
  #else
    #define USE_OPENCL_COMPUTE
    #include <OpenCL/cl.h>
    #include <OpenCL/cl_ext.h>
  #endif
#else
  #if defined(USE_VULKAN_COMPUTE)
    #include <vulkan.h>
  #else
    #define USE_OPENCL_COMPUTE
    #ifdef USE_MINICL
      #include <MiniCL/cl.h>
    #else
      #include <CL/cl.h>
      #ifdef _WIN32
        #include <CL/cl.hpp>
        #include "CL/cl_gl.h"
      #endif
    #endif
  #endif
#endif

#if defined(USE_METAL_COMPUTE)

typedef id<MTLDevice>               ComputeContext;
typedef id<MTLDevice>               ComputeDeviceId;
typedef id<MTLComputePipelineState> ComputeKernelIdentifier;
typedef id<MTLLibrary>              ComputeProgramIdentifier;
typedef id<MTLCommandQueue>         ComputeQueue;
typedef id<MTLCommandBuffer>        ComputeCommandBuffer;
typedef id<MTLBuffer>               ComputeMemoryIdentifier;
typedef id<MTLBuffer>               ComputeBufferIdentifier;
typedef uint                        ComputeStatus;
typedef id<MTLTexture>              ComputeTextureIdentifier;

enum ComputeMemoryUsage : uint8_t
{
  KERNEL_RW,
  KERNEL_W,
  KERNEL_R
};

#elif defined(USE_VULKAN_COMPUTE)

typedef VkDevice        ComputeContext;
typedef VkDevice        ComputeDeviceId;
typedef VkPipeline      ComputeKernelIdentifier;
typedef VkShaderModule  ComputeProgramIdentifier;
typedef VkQueue         ComputeQueue;
typedef VkCommandBuffer ComputeCommandBuffer;
typedef VkDeviceMemory  ComputeMemoryIdentifier;
typedef VkBuffer        ComputeBufferIdentifier;
typedef VkResult        ComputeStatus;
typedef uint            ComputeTextureIdentifier;

enum ComputeMemoryUsage : uint8_t
{
  KERNEL_RW = 0,
  KERNEL_W  = 0,
  KERNEL_R  = 0,
};

#else

typedef cl_context        ComputeContext;
typedef cl_device_id      ComputeDeviceId;
typedef cl_kernel         ComputeKernelIdentifier;
typedef cl_program        ComputeProgramIdentifier;
typedef cl_command_queue  ComputeQueue;
typedef cl_mem            ComputeMemoryIdentifier;
typedef cl_mem            ComputeBufferIdentifier;
typedef cl_int            ComputeStatus;
typedef cl_mem            ComputeTextureIdentifier;

enum ComputeMemoryUsage : uint8_t
{
  KERNEL_RW = CL_MEM_READ_WRITE,
  KERNEL_W  = CL_MEM_WRITE_ONLY,
  KERNEL_R  = CL_MEM_READ_ONLY
};

#endif

enum ComputeMemoryStorage : uint8_t
{
  Default,
  Private,
  Shared,
  Managed
};

extern void logComputeMessage(const char* format, ...);
extern void logComputeError(const char* format, ...);

extern const char* getStatusMessage(ComputeStatus status);
#define computeCheckError(a, b) if((a)!=(b)) { printf("Compute Error : %s (code %d)\n", getStatusMessage(a), (int)(a)); assert((a) == (b)); }
#define checkError(a) computeCheckError(a, 0)

class ComputeInterface;

#if defined(USE_VULKAN_COMPUTE)
class ComputeMemory : private VkDescriptorBufferInfo
#else
class ComputeMemory
#endif
{
  ComputeMemoryIdentifier ref;
  ComputeMemoryStorage storage;
  friend class ComputeHeap;

#if defined(USE_VULKAN_COMPUTE)
  VkDeviceSize& size = range;
  ComputeBufferIdentifier& buf = buffer;

public:

  ComputeMemory(ComputeMemoryIdentifier ref, ComputeBufferIdentifier buf, size_t offset, size_t size);

  operator const ComputeBufferIdentifier()const {return buf;}

  const VkDescriptorBufferInfo* getDescriptorBufferInfo()const {return this;}

#else

  size_t size;
  size_t offset;
  ComputeBufferIdentifier buf;

#endif

public:

  ComputeMemory();

  ComputeMemory(const ComputeMemory& ref);

  ComputeMemory(ComputeMemoryIdentifier ref, size_t offset = 0, size_t size = 0, ComputeMemoryStorage storage = Default);

  ComputeMemory(ComputeMemory* memory, size_t offset = 0, size_t size = 0);

  ComputeMemory& operator = (const ComputeMemory& ref);

  ComputeMemoryUsage getUsage()const;

  size_t getOffset()const {return offset;}

  size_t getSize()const {return size;}

  operator const ComputeMemoryIdentifier()const {return ref;}
};


class ComputeTexture
{
  ComputeTextureIdentifier ref;
  uint size[2];
  uint bytesPerPixel;

public:

  ComputeTexture();

  ComputeTexture(ComputeTextureIdentifier ref, uint size[2], uint bytesPerPixel);

  uint getBytesPerPixel()const {return bytesPerPixel;}

  const uint* getSize()const {return size;}

  operator const ComputeTextureIdentifier()const {return ref;}
};


class ComputeHeap
{
  bool              bypass;
  ComputeMemory*    heap;
  ComputeInterface* compute;
  vector<ComputeMemory*>  childs;

public:

  ComputeHeap(ComputeInterface* compute, bool bypass = false);

  ~ComputeHeap();

  void create(size_t sizeInBytes);

  ComputeMemory* alloc(size_t sizeInBytes, void* data = NULL, ComputeMemoryStorage storage = Default, ComputeMemoryUsage flag = KERNEL_RW);

  void free(ComputeMemory* memory);

  ComputeMemory* get()const {return heap;}
};


class ComputeKernel
{
  ComputeInterface* compute;
  ComputeKernelIdentifier ref;

  enum ArgDataType
  {
    ARG_DATA_HOST_PTR,
    ARG_DATA_CONST_HOST_PTR,
    ARG_DATA_DEVICE_PTR,
    ARG_DATA_CONST_DEVICE_PTR,
    ARG_DATA_SHARED_PTR,
  };

  // argument data structure to defer setting argument
  struct ArgData
  {
    ArgDataType type = ARG_DATA_HOST_PTR;
    uint size        = 0;
    uint index       = 0;
    const void* cptr = NULL;
  };

  vector<ArgData> args;
  bool setArgumentBuffer;
  vector<pair<uint, uint>> argumentBufferRange;
  unordered_map<uint, ComputeMemoryIdentifier> argumentBuffers;

  // Number of arguments the kernel actually has after compilation. setArg
  // calls with an index >= numArgs become a no-op. NVIDIA's OpenCL strips
  // unused kernel arguments, so the host's expected indices may exceed what
  // the surviving kernel exposes. UINT32_MAX means "no limit / not queried"
  // (default for Metal/Vulkan/Apple-OpenCL paths that don't strip args).
  uint numArgs = (uint)-1;

  uint mapArgumentIndex(uint index)const;

  friend class ComputeProgram;

#if defined(USE_VULKAN_COMPUTE)
  ComputeProgramIdentifier  shaderModule;
  VkDescriptorSetLayout     descriptorSetLayout;
  VkDescriptorSet           descriptorSet;
  VkPipelineLayout          pipelineLayout;

  vector<VkWriteDescriptorSet>    computeWriteDescriptorSets;
  vector<VkDescriptorBufferInfo>  descriptorBufferInfos;
  vector<VkPushConstantRange>     pushConstantRanges;

  void*   constantData;
  string  name;
  bool    usePushConstants;

  void createPipeline();

public:

  ComputeKernel(const ComputeKernel& ref);

  ~ComputeKernel();

  void setArgs(VkCommandBuffer commandBuffer);

  operator const ComputeProgramIdentifier()const {return shaderModule;}

#endif

public:

  ComputeKernel(ComputeKernelIdentifier ref = NULL, ComputeInterface* compute = NULL);

  void addArgumentBufferRange(uint startIndex, uint inclusiveEndIndex);

  void setArg(void* valuePtr, const size_t valueSize, uint index);

  void setArg(const void* valuePtr, const size_t valueSize, uint index);

  template<class ArgType> void setArg(ArgType* valuePtr, uint index)
  {
    setArg(valuePtr, sizeof(ArgType), index);
  }

  void setArg(ComputeMemory* buffer, uint index);

  void setArg(const ComputeMemory* buffer, uint index);

  void setArgs(ComputeMemory* buffers[], const uint count, uint* indices = NULL);

  void setSharedMemArg(const size_t valueSize, uint index);

  operator const ComputeKernelIdentifier()const {return ref;}

  void setArgs();

  void registerResource(const ComputeMemory* resource);
};


class ComputeProgram
{
  ComputeInterface* compute;
  ComputeProgramIdentifier ref;
  vector<FunctionArgs> kernelArgs;

#if defined(USE_VULKAN_COMPUTE)
  VkDescriptorPool descriptorPool;
#endif

public:

  ComputeProgram();

  ComputeProgram(const ComputeProgram& ref);

  ComputeProgram(ComputeProgramIdentifier ref, ComputeInterface* compute, const vector<FunctionArgs>& kernelArgs = {});

  ComputeKernel createKernel(const char* kernelName);

  bool isEmpty()const;

  operator const ComputeProgramIdentifier()const {return ref;}
};


/*!
@class Interface providing abstraction to API calls.
*/
class ComputeInterface
{
  ComputeDeviceId   deviceId;
  ComputeContext    context;
  ComputeQueue      queue;
  size_t            simdGroupSize;
  size_t            maxThreadsPerWorkgroup;

  double  executionTime = 0.f;
  uint    deviceCount;

#if defined(USE_VULKAN_COMPUTE)
  vector<VkPhysicalDevice> devices;

  /** @brief Properties of the physical device including limits that the application can check against */
  VkPhysicalDeviceProperties properties;

  /** @brief Features of the physical device that an application can use to check if a feature is supported */
  VkPhysicalDeviceFeatures features;

  /** @brief Features that have been enabled for use on the physical device */
  VkPhysicalDeviceFeatures enabledFeatures;

  /** @brief Memory types and heaps of the physical device */
  VkPhysicalDeviceMemoryProperties memoryProperties;

  /** @brief Queue family properties of the physical device */
  vector<VkQueueFamilyProperties> queueFamilyProperties;

  /** @brief List of extensions supported by the device */
  vector<string> supportedExtensions;

  VkCommandPool commandPool;

  VkInstance instance;
  VkDebugUtilsMessengerEXT debugMessenger;

  VkSemaphore computeSemaphore;
  VkFence computeFence;

public:
  VkPipelineCache pipelineCache;
  uint getMemoryType(uint typeBits, const VkMemoryPropertyFlags properties)const;

#else
  vector<ComputeDeviceId> devices;

#if defined(USE_METAL_COMPUTE)

  id<MTLCommandQueue> commandQueue = nil;
  id<MTLBlitCommandEncoder> currentBlitEncoder = nil;
  id<MTLComputeCommandEncoder> currentComputeEncoder = nil;

  bool commandBufferRecording;
  ComputeCommandBuffer currentCommandBuffer{};
  ComputeCommandBuffer getComputeCommandBuffer();

  void endEncoders();

public:
  id<MTLBlitCommandEncoder>     getBlitEncoder();
  id<MTLComputeCommandEncoder>  getComputeEncoder();

#endif

#endif

private:

  friend class ComputeHeap;

  ComputeProgram createProgram(const char* sourceCode, size_t sourceSize, const vector<string>* oldType = NULL, const vector<string>* newType = NULL, const vector<string>* includeFiles = NULL);

public:

  ComputeHeap heap;

  ComputeInterface();

  ~ComputeInterface();

  void create(int deviceIndex = -1);

  ComputeProgram createTemplateProgram(const char* fileName, const vector<string>* oldType = NULL,
    const vector<string>* newType = NULL, const vector<string>* includeFiles = NULL);

  ComputeProgram createTemplateProgram(const string& sourceCode, const vector<string>* oldType = NULL,
    const vector<string>* newType = NULL, const vector<string>* includeFiles = NULL);

  void copyBuffer(const ComputeMemory* source, ComputeMemory* destination, size_t sourceOffset, size_t destinationOffset, size_t sizeInBytes);

  void copyTexture(const ComputeTexture* source, ComputeTexture* destination);

  void copyTextureToBuffer(const ComputeTexture* source, ComputeMemory* destination, size_t destinationOffset = 0, size_t sourceSlice = 0, size_t sourceLevel = 0);

  void copyBufferToTexture(const ComputeMemory* source, ComputeTexture* destination, size_t sourceOffset = 0, size_t destinationSlice = 0, size_t destinationLevel = 0);

  void setBuffer(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, const void* hostValue, size_t hostValueSize);

  void copyToHost(const ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, void* hostPtr, bool waitForFinish);

  void copyFromHost(ComputeMemory* destination, size_t destinationOffset, size_t sizeInBytes, const void* hostPtr, bool waitForFinish);


  void configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount);

  void configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount, const uint maxThreadsPerThreadgroup);

  void configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount[3]);

  void configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount[3], const uint maxThreadsPerThreadgroup);

  void execute(ComputeKernel& kernel, const size_t workgroupSize[3], const size_t workgroupCount[3]);

  void execute(ComputeKernel& kernel, const size_t workgroupSize[3], const ComputeMemory* workgroupCount, size_t bufferOffset);

  void execute(ComputeKernel& kernel, const size_t threadCount);

  void sync(bool waitOnFinish = true);

  uint simdSize()const {return (uint)simdGroupSize;}

  uint maxThreadsPerGroup()const {return (uint)maxThreadsPerWorkgroup;}

  uint maxCores()const;

  void startCapture();

  void endCapture();

  double lastExecutionTime()const {return executionTime;}

  ComputeDeviceId getDevice() {return deviceId;}

#ifdef ENABLE_RENDERING

  ComputeMemory createMemoryFromGLBuffer(GLuint glObject);

  ComputeMemory createMemoryFromGLTexture(GLuint glObject);

#endif
};

#endif
