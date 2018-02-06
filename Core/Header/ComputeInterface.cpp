#include "ComputeInterface.h"
#include <fstream>
#include <stdio.h>  /* defines FILENAME_MAX */
// #define WINDOWS  /* uncomment this line to use it for windows.*/ 
#ifdef _WIN32
//#include <direct.h>
#include <Pathcch.h>
#else
#include <limits.h>
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <iostream>
#include <string.h>
#include <algorithm>

#define CREATE_SUB_BUFFER

const char* getStatusMessage(ComputeStatus status)
{
  switch (status)
  {
    // run-time and JIT compiler errors
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default: return "Unknown OpenCL error";
  }
}

std::string getCurrentDir(void)
{
  char currentPath[1024];
#ifdef _WIN32
  int len = GetModuleFileName(NULL, currentPath, 1024);
#else
  ssize_t len = ::readlink("/proc/self/exe", currentPath, 1023);
#endif
  if (len != -1)
  {
    currentPath[len] = '\0';
  }
  std::string ret = std::string(currentPath);
  return ret.substr(0, ret.find_last_of("\\/"));
}


ComputeMemory::ComputeMemory()
{
  ref = NULL;
  offset = 0;
  size = 0;
}

ComputeMemory::ComputeMemory(ComputeMemoryIdentifier ref, size_t offset, size_t size)
{
  this->ref = ref;
  this->offset = offset;
  this->size = size;
}

ComputeMemoryFlag ComputeMemory::getFlag()const
{
  cl_mem_flags ret;
  size_t size;
  ComputeStatus status = clGetMemObjectInfo(ref, CL_MEM_FLAGS, sizeof(cl_mem_flags), &ret, &size);
  computeCheckError(status, 0);
  switch (ret)
  {
  case CL_MEM_READ_WRITE:
    return KERNEL_RW;
  case CL_MEM_READ_ONLY:
    return KERNEL_R;
  case CL_MEM_WRITE_ONLY:
    return KERNEL_W;
  default:
    assert(0);
  }
  return KERNEL_RW;
}

size_t ComputeMemory::getOffset()const
{
  return offset;
}

size_t ComputeMemory::getSize()const
{
  //size_t ret;
  //ComputeStatus status = clGetMemObjectInfo(ref, CL_MEM_SIZE, sizeof(size_t), &ret, NULL);
  //computeCheckError(status, 0);
  return size;
}


ComputeHeap::ComputeHeap(ComputeInterface* compute, bool bypass)
  : heap(NULL), compute(compute), bypass(bypass)
{}

ComputeHeap::~ComputeHeap()
{
  for (ComputeMemory* mem : childs)
  {
    free(mem);
  }
  childs.clear();
  if (heap)
  {
    free(heap);
    heap = NULL;
  }
}

void ComputeHeap::create(size_t sizeInBytes)
{
  if (!bypass)
  {
    bypass = true;
    heap = alloc(sizeInBytes);
    bypass = false;
    childs.pop_back();
  }
}

ComputeMemory* ComputeHeap::alloc(size_t sizeInBytes, void* data, ComputeMemoryFlag flag)
{
  ComputeMemory* ret;
  ComputeStatus status;
  if (bypass)
  {
    ret = new ComputeMemory(
      clCreateBuffer(compute->context, flag, sizeInBytes, data, &status),
      0, sizeInBytes);
    computeCheckError(status, 0);
  }
  else
  {
    size_t offset = 0;
    if (childs.size())
    {
      const ComputeMemory* last = childs.back();
      offset = last->getOffset() + last->getSize();
    }
    cl_buffer_region region;
    region.origin = offset;
    region.size = sizeInBytes;

#ifdef CREATE_SUB_BUFFER
    ret = new ComputeMemory(
      clCreateSubBuffer(*heap, heap->getFlag(), CL_BUFFER_CREATE_TYPE_REGION, &region, &status),
      offset, sizeInBytes);
#else
    ret = new ComputeMemory(*heap, offset, sizeInBytes);
    status = 0;
#endif
    computeCheckError(status, 0);
  }
  childs.push_back(ret);

  return ret;
}

void ComputeHeap::free(ComputeMemory* memory)
{
#ifdef CREATE_SUB_BUFFER
  ComputeStatus status = clReleaseMemObject(*memory);
  computeCheckError(status, 0);
#endif

  vector<ComputeMemory*>::iterator pos = find(childs.begin(), childs.end(), memory);
  if (pos != childs.end())
  {
    childs.erase(pos);
  }
  else if (memory != heap) // not during destructor
  {
    printf("Error: Trying to free outside memory!");
    assert(0);
  }
}


ComputeKernel::ComputeKernel()
{
  ref = NULL;
}

ComputeKernel::ComputeKernel(ComputeKernelIdentifier ref)
{
  this->ref = ref;
}

void ComputeKernel::setArg(void* valuePtr, size_t valueSize, uint argIndex)
{
  ComputeStatus status = clSetKernelArg(ref, argIndex, valueSize, valuePtr);
  computeCheckError(status, 0);
}

void ComputeKernel::setArgs(ComputeMemory* buffers[], const uint count, uint* indices)
{
  for (uint i = 0; i < count; i++)
  {
    ComputeMemoryIdentifier ident = *buffers[i];
    setArg<ComputeMemoryIdentifier>(&ident, indices ? indices[i] : i);
  }
}


ComputeProgram::ComputeProgram()
{
  ref = NULL;
}

ComputeProgram::ComputeProgram(ComputeProgramIdentifier ref)
{
  this->ref = ref;
};

ComputeKernel ComputeProgram::createKernel(const char* kernelName)
{
  ComputeStatus status;
  ComputeKernel kernel(clCreateKernel(ref, kernelName, &status));
  computeCheckError(status, 0);
  return kernel;
}

bool ComputeProgram::isEmpty()const
{
  return ref == NULL;
}

static ComputePlatformId platforms[8];
static uint platformCount = 0;

static ComputeDeviceId devices[8];
static uint deviceCount = 0;

ComputeInterface::ComputeInterface()
  :heap(this, true)
{
  platform = NULL;
  deviceId = NULL;
  context = NULL;
  queue = NULL;
}

ComputeInterface::~ComputeInterface()
{
  if (platform)
  {

  }
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

void ComputeInterface::create(uint platformIndex)
{
  /* Get Platform and Device Info */
  ComputeStatus status = clGetPlatformIDs(8, platforms, &platformCount);
  computeCheckError(status, 0);
  for (uint i = 0; i < platformCount; i++)
  {
    const uint MAX_STRING_LENGTH = 128;
    cl_int status;
    char vendor[MAX_STRING_LENGTH];
    char name[MAX_STRING_LENGTH];
    char version[MAX_STRING_LENGTH];
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, MAX_STRING_LENGTH, vendor, NULL);
    computeCheckError(status, 0);
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, MAX_STRING_LENGTH, name, NULL);
    computeCheckError(status, 0);
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, MAX_STRING_LENGTH, version, NULL);
    computeCheckError(status, 0);

    printf("Platform info:\n");
    printf("  CL_PLATFORM_VENDOR:   %s\n", vendor);
    printf("  CL_PLATFORM_NAME:     %s\n", name);
    printf("  CL_PLATFORM_VERSION:  %s\n", version);
  }
  platform = platforms[platformIndex];

  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 8, devices, &deviceCount);
  computeCheckError(status, 0);
  deviceId = devices[0];

  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &status);
  computeCheckError(status, 0);

  queue = clCreateCommandQueue(context, deviceId, 0, &status);
  computeCheckError(status, 0);
}

std::string readFile(const char* fileName)
{
  std::string directory = getCurrentDir();
  std::string data;
  std::ifstream file;
  file.open(directory + "/" + fileName, std::ios::binary);

  file.seekg(0, std::ios::end);
  data.reserve((size_t)file.tellg());
  file.seekg(0, std::ios::beg);
  data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  file.close();

  return data;
}

ComputeProgram ComputeInterface::createProgram(const char* sourceCode, size_t sourceSize)
{
  ComputeStatus status;
  ComputeProgram program(clCreateProgramWithSource(context, 1, (const char **)&sourceCode, (const size_t *)&sourceSize, &status));
  computeCheckError(status, 0);

  status = clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL);
  // Determine the size of the log
  size_t logSize;
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

  // Allocate memory for the log
  char* log = new char[logSize];

  // Get the log
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

  // Print the log
  printf("Compilation Log:\n\n%s\n", log);
  delete log;

  computeCheckError(status, 0);


  return program;
}

ComputeProgram ComputeInterface::createTemplateProgram(const char* fileName,
  const vector<string>* oldType,
  const vector<string>* newType,
  const vector<string>* includeFiles)
{
  std::string data = "\n";
  if (oldType)
  {
    for (uint i = 0; i < oldType->size(); i++)
    {
      data += "typedef " + (*oldType)[i] + " " + (*newType)[i] + ";\n";
    }
  }
  if (includeFiles)
  {
    for (uint i = 0; i < includeFiles->size(); i++)
    {
      data += readFile((*includeFiles)[i].c_str()) + "\n";
    }
  }
  data += readFile(fileName);
  data += "\n";

  return createProgram(data.c_str(), data.size());
}

void ComputeInterface::copyBuffer(ComputeMemory* source, ComputeMemory* destin, size_t sourceOffset, size_t destinOffset, size_t sizeInBytes)
{
#ifdef CREATE_SUB_BUFFER
  ComputeStatus status = clEnqueueCopyBuffer(queue, *source, *destin, sourceOffset, destinOffset, sizeInBytes, 0, NULL, NULL);
#else
  ComputeStatus status = clEnqueueCopyBuffer(queue, *source, *destin, sourceOffset + source->getOffset(), destinOffset + destin->getOffset(), sizeInBytes, 0, NULL, NULL);
#endif
  computeCheckError(status, 0);
}

void ComputeInterface::setBuffer(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, const void* hostValue, size_t hostValueSize)
{
  ComputeStatus status = clEnqueueFillBuffer(queue, *source, hostValue, hostValueSize, sourceOffset, sizeInBytes, 0, NULL, NULL);
  computeCheckError(status, 0);
}

void ComputeInterface::copyToHost(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, void* hostPtr, bool waitForFinish)
{
  ComputeStatus status = clEnqueueReadBuffer(queue, *source, waitForFinish, sourceOffset, sizeInBytes, hostPtr, 0, NULL, NULL);
  computeCheckError(status, 0);
}

void ComputeInterface::copyFromHost(ComputeMemory* destin, size_t destinOffset, size_t sizeInBytes, const void* hostPtr, bool waitForFinish)
{
  ComputeStatus status = clEnqueueWriteBuffer(queue, *destin, waitForFinish, destinOffset, sizeInBytes, hostPtr, 0, NULL, NULL);
  computeCheckError(status, 0);
}

void ComputeInterface::configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount)
{
  uint maxThreads = 1024;
  uint height = maxThreads / 32;
  workgroupSize[0] = 32;
  workgroupSize[1] = (uint)ceil(threadCount / float(height));
  workgroupSize[2] = 1;

  workgroupCount[0] = 1;
  if (workgroupSize[1] > height)
  {
    workgroupCount[0] = (uint)ceil(workgroupSize[1] / float(height));
    workgroupSize[1] = height;
  }
  workgroupCount[1] = 1;
  workgroupCount[2] = 1;
}

void ComputeInterface::execute(ComputeKernel kernel, const size_t workgroupSize[3], const size_t workgroupCount[3])
{
  const size_t workgroup[3] = {
    workgroupSize[0] * workgroupCount[0],
    workgroupSize[1] * workgroupCount[1],
    workgroupSize[2] * workgroupCount[2] };
  ComputeStatus status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, workgroup, workgroupSize, 0, NULL, NULL);
  computeCheckError(status, 0);
}

void ComputeInterface::sync()
{
  ComputeStatus status = clFinish(queue);
  computeCheckError(status, 0);
}

#ifdef ENABLE_RENDERING
ComputeMemory ComputeInterface::createMemoryFromGLBuffer(GLuint glObject)
{
  ComputeStatus status;
  ComputeMemory ret(clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, glObject, &status));
  computeCheckError(status, 0);
  return ret;
}
#endif