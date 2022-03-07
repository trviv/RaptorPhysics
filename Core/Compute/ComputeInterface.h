#ifndef COMPUTE_INTERFACE_H
#define COMPUTE_INTERFACE_H

#include <Header/Math.h>
#include <Utils/IOInterface.h>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#ifdef USE_METAL_COMPUTE
#import <Metal/Metal.h>
#else
#define USE_OPENCL_COMPUTE
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#endif
#else
#define USE_OPENCL_COMPUTE
#ifdef USE_MINICL
#include <MiniCL/cl.h>
#else
#include <CL/cl.h>
#ifdef _WIN32
#include <CL/cl.hpp>
#include "CL/cl_gl.h"
#endif //_WIN32
#endif
#endif //__APPLE__

#ifdef USE_METAL_COMPUTE

typedef uint                        ComputePlatformId;
typedef uint                        ComputeDeviceType;
typedef id<MTLDevice>               ComputeContext;
typedef id<MTLDevice>               ComputeDeviceId;
typedef id<MTLComputePipelineState> ComputeKernelIdentifier;
typedef id<MTLLibrary>              ComputeProgramIdentifier;
typedef id<MTLCommandQueue>         ComputeQueue;
typedef id<MTLBuffer>               ComputeMemoryIdentifier;
typedef uint                        ComputeStatus;
typedef id<MTLTexture>              ComputeTextureIdentifier;

enum ComputeMemoryFlag
{
  KERNEL_RW,
  KERNEL_W,
  KERNEL_R
};

#else

typedef cl_platform_id    ComputePlatformId;
typedef cl_device_type    ComputeDeviceType;
typedef cl_context        ComputeContext;
typedef cl_device_id      ComputeDeviceId;
typedef cl_kernel         ComputeKernelIdentifier;
typedef cl_program        ComputeProgramIdentifier;
typedef cl_command_queue  ComputeQueue;
typedef cl_mem            ComputeMemoryIdentifier;
typedef cl_int            ComputeStatus;
typedef cl_mem            ComputeTextureIdentifier;

enum ComputeMemoryFlag
{
  KERNEL_RW = CL_MEM_READ_WRITE,
  KERNEL_W = CL_MEM_WRITE_ONLY,
  KERNEL_R = CL_MEM_READ_ONLY
};

#endif

extern const char* getStatusMessage(ComputeStatus status);

extern void logComputeMessage(const char* format, ...);
extern void logComputeError(const char* format, ...);

#define computeCheckError(a, b) if((a)!=(b)) { printf("Compute Error : %s\n", getStatusMessage(a)); assert((a) == (b)); }

class ComputeInterface;

class ComputeMemory
{
  ComputeMemoryIdentifier ref;
  size_t offset;
  size_t size;

  friend class ComputeHeap;
public:

  ComputeMemory();

  ComputeMemory(ComputeMemoryIdentifier ref, size_t offset = 0, size_t size = 0);

  ComputeMemoryFlag getFlag()const;

  size_t getOffset()const;

  size_t getSize()const;

  operator const ComputeMemoryIdentifier()const
  {
    return ref;
  }
};


class ComputeTexture
{
  ComputeTextureIdentifier ref;
  uint size[2];
  uint bytesPerPixel;

public:

  ComputeTexture();

  ComputeTexture(ComputeTextureIdentifier ref, uint size[2], uint bytesPerPixel);

  uint getBytesPerPixel()const;

  const uint* getSize()const;

  operator const ComputeTextureIdentifier()const
  {
    return ref;
  }
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

  ComputeMemory* alloc(size_t sizeInBytes, void* data = NULL, ComputeMemoryFlag flag = KERNEL_RW);

  void free(ComputeMemory* memory);

  ComputeMemory* get()const
  {
    return heap;
  }
};


class ComputeKernel
{
  ComputeKernelIdentifier ref;

  // argument data structure to defer setting argument
  struct ArgData
  {
    uint type   = 0;
    uint size   = 0;
    uint index  = 0;
    void* ptr   = NULL;
    const void* cptr  = NULL;
  };

  vector<ArgData> args;
  bool setArgumentBuffer;
  vector<pair<uint, uint>> argumentBufferRange;
  unordered_map<uint, ComputeMemoryIdentifier> argumentBuffers;

  uint mapArgumentIndex(uint index)const;

public:

  ComputeKernel();

  ComputeKernel(ComputeKernelIdentifier ref);

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

  operator const ComputeKernelIdentifier()const
  {
    return ref;
  }

  void setArgs();

  void registerResource(const ComputeMemory* resource);
};


class ComputeProgram
{
  ComputeProgramIdentifier ref;

public:

  ComputeProgram();

  ComputeProgram(ComputeProgramIdentifier ref);

  ComputeKernel createKernel(const char* kernelName);

  bool isEmpty()const;

  operator const ComputeProgramIdentifier()const
  {
    return ref;
  }
};


/*!
@class Interface providing abstraction to API calls.
*/
class ComputeInterface
{
  ComputePlatformId platform;
  ComputeDeviceId   deviceId;
  ComputeContext    context;
  ComputeQueue      queue;
  size_t            simdGroupSize;
  size_t            maxThreadsPerWorkgroup;

  friend class ComputeHeap;

  ComputeProgram createProgram(const char* sourceCode, size_t sourceSize);

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

  uint simdSize()const;

  uint maxThreadsPerGroup()const;

  uint maxCores()const;

  void startCapture();

  void endCapture();

  double lastExecutionTime()const;

  ComputeDeviceId getDevice();

#ifdef ENABLE_RENDERING

  ComputeMemory createMemoryFromGLBuffer(GLuint glObject);

  ComputeMemory createMemoryFromGLTexture(GLuint glObject);

#endif
};

#endif
