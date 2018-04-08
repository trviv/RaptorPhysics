#ifndef COMPUTE_INTERFACE_H
#define COMPUTE_INTERFACE_H

#include "Root.h"
#include <CL/cl.hpp>

#ifdef __APPLE__
#ifdef USE_MINICL
#include <MiniCL/cl.h>
#else
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h> //clLogMessagesToStderrAPPLE
#endif
#else
#ifdef USE_MINICL
#include <MiniCL/cl.h>
#else
#include <CL/cl.h>
#ifdef _WIN32
#include "CL/cl_gl.h"
#endif //_WIN32
#endif
#endif //__APPLE__

#include <map>
#include <assert.h>
#include <stdio.h>
#include <vector>
using namespace std;

typedef cl_platform_id    ComputePlatformId;
typedef cl_device_type    ComputeDeviceType;
typedef cl_context        ComputeContext;
typedef cl_device_id      ComputeDeviceId;
typedef cl_kernel         ComputeKernelIdentifier;
typedef cl_program        ComputeProgramIdentifier;
typedef cl_command_queue  ComputeQueue;
typedef cl_mem            ComputeMemoryIdentifier;
typedef cl_int            ComputeStatus;

extern string readFile(const char* fileName);
extern const char* getStatusMessage(ComputeStatus status);

extern void logComputeMessage(const char* format, ...);
extern void logComputeError(const char* format, ...);

#define computeCheckError(a, b) if((a)!=(b)) { printf("Compute Error : %s\n", getStatusMessage(a)); assert((a) == (b)); }

class ComputeInterface;

enum ComputeMemoryFlag
{
  KERNEL_RW = CL_MEM_READ_WRITE,
  KERNEL_W = CL_MEM_WRITE_ONLY,
  KERNEL_R = CL_MEM_READ_ONLY
};


class ComputeMemory
{
  ComputeMemoryIdentifier ref;
  size_t offset;
  size_t size;

public:

  ComputeMemory();

  ComputeMemory(ComputeMemoryIdentifier ref, size_t offset = 0, size_t size = 0);

  ComputeMemoryFlag getFlag()const;

  size_t getOffset()const;

  size_t getSize()const;

  operator ComputeMemoryIdentifier()
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

public:

  ComputeKernel();

  ComputeKernel(ComputeKernelIdentifier ref);

  void setArg(void* valuePtr, const size_t valueSize, uint index);

  template<class ArgType> void setArg(ArgType* valuePtr, uint index)
  {
    setArg(valuePtr, sizeof(ArgType), index);
  }

  void setArg(ComputeMemory* buffer, uint index);

  void setArgs(ComputeMemory* buffers[], const uint count, uint* indices = NULL);

  operator ComputeKernelIdentifier()
  {
    return ref;
  }
};


class ComputeProgram
{
  ComputeProgramIdentifier ref;

public:

  ComputeProgram();

  ComputeProgram(ComputeProgramIdentifier ref);

  ComputeKernel createKernel(const char* kernelName);

  bool isEmpty()const;

  operator ComputeProgramIdentifier()
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

  friend class ComputeHeap;

public:

  ComputeHeap heap;

  ComputeInterface();

  ~ComputeInterface();

  void create(uint platformIndex);


  ComputeProgram createProgram(const char* sourceCode, size_t sourceSize);

  ComputeProgram createTemplateProgram(const char* fileName, const vector<string>* oldType = NULL,
    const vector<string>* newType = NULL, const vector<string>* includeFiles = NULL);


  void copyBuffer(ComputeMemory* source, ComputeMemory* destin, size_t sourceOffset, size_t destinOffset, size_t sizeInBytes);

  void setBuffer(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, const void* hostValue, size_t hostValueSize);

  void copyToHost(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, void* hostPtr, bool waitForFinish);

  void copyFromHost(ComputeMemory* destin, size_t destinOffset, size_t sizeInBytes, const void* hostPtr, bool waitForFinish);


  void configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount);

  void execute(ComputeKernel kernel, const size_t workgroupSize[3], const size_t workgroupCount[3]);

  void sync();

  uint maxThreadsPerGroup()const;

#ifdef ENABLE_RENDERING

  ComputeMemory createMemoryFromGLBuffer(GLuint glObject);

  ComputeMemory createMemoryFromGLTexture(GLuint glObject);

#endif
};

#endif