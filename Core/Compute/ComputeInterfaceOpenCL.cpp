/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "ComputeInterface.h"
#include <unordered_set>

#ifdef USE_OPENCL_COMPUTE
#define CREATE_SUB_BUFFER
//#define ENABLE_CL_PROFILING
//#define DEBUG_KERNEL_LAUNCHES // sync after every dispatch + log faulting kernel
//#define DEBUG_KERNEL_TRACE    // log every kernel name as it's dispatched

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

ComputeMemory* ComputeHeap::alloc(size_t sizeInBytes, void* data, ComputeMemoryStorage storage, ComputeMemoryUsage flag)
{
  // Sub-buffer offsets must satisfy CL_DEVICE_MEM_BASE_ADDR_ALIGN. Apple and
  // NVIDIA tolerate 16; POCL on AVX-512 / wider SIMD devices need more, and
  // mismatch yields CL_MISALIGNED_SUB_BUFFER_OFFSET. Query once and round up.
  static size_t deviceAlign = 0;
  if (deviceAlign == 0)
  {
    cl_uint alignBits = 0;
    clGetDeviceInfo(compute->deviceId, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                    sizeof(alignBits), &alignBits, NULL);
    deviceAlign = alignBits / 8;
    if (deviceAlign < 16) deviceAlign = 16;
  }
  sizeInBytes = mAlignBy(sizeInBytes, deviceAlign) * deviceAlign;
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
      clCreateSubBuffer(*heap, (cl_mem_flags)heap->getUsage(), CL_BUFFER_CREATE_TYPE_REGION, &region, &status),
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
    logComputeError("Trying to free outside memory!");
  }
}


ComputeKernel::ComputeKernel(ComputeKernelIdentifier ref, ComputeInterface* compute)
{
  this->ref = ref;
  this->compute = compute;
  setArgumentBuffer = false;
}

void ComputeKernel::addArgumentBufferRange(uint, uint)
{
  // Argument buffers are a Metal-specific optimization; OpenCL has no equivalent.
}

void ComputeKernel::setArg(void* valuePtr, size_t valueSize, uint index)
{
  if (index >= numArgs) return;  // arg was stripped by the optimizer
  ComputeStatus status = clSetKernelArg(ref, index, valueSize, valuePtr);
  computeCheckError(status, 0);
}

void ComputeKernel::setArg(const void* valuePtr, size_t valueSize, uint index)
{
  if (index >= numArgs) return;
  ComputeStatus status = clSetKernelArg(ref, index, valueSize, valuePtr);
  computeCheckError(status, 0);
}

void ComputeKernel::setArg(const ComputeMemory* buffer, uint index)
{
  ComputeMemoryIdentifier ident = *buffer;
  setArg<ComputeMemoryIdentifier>(&ident, index);
}

void ComputeKernel::setArg(ComputeMemory* buffer, uint index)
{
  ComputeMemoryIdentifier ident = *buffer;
  setArg<ComputeMemoryIdentifier>(&ident, index);
}

void ComputeKernel::setSharedMemArg(const size_t valueSize, uint index)
{
  if (index >= numArgs) return;
  ComputeStatus status = clSetKernelArg(ref, index, valueSize, NULL);
  computeCheckError(status, 0);
}

void ComputeKernel::setArgs()
{
  // Argument buffers are Metal-specific; OpenCL sets args eagerly via clSetKernelArg.
}

void ComputeKernel::registerResource(const ComputeMemory*)
{
  logComputeError("Register resource is not implemented!");
}


ComputeProgram::ComputeProgram()
{
  ref = NULL;
  compute = NULL;
  kernelArgs.clear();
}

ComputeProgram::ComputeProgram(const ComputeProgram& ref)
{
  this->ref = ref.ref;
  this->compute = ref.compute;
  this->kernelArgs = ref.kernelArgs;
}

ComputeProgram::ComputeProgram(ComputeProgramIdentifier ref, ComputeInterface* compute, const vector<FunctionArgs>& kernelArgs)
{
  this->ref = ref;
  this->compute = compute;
  this->kernelArgs = kernelArgs;
}

ComputeKernel ComputeProgram::createKernel(const char* kernelName)
{
  ComputeStatus status;
  cl_kernel rawKernel = clCreateKernel(ref, kernelName, &status);
  computeCheckError(status, 0);

  // Query the surviving argument count once. NVIDIA's OpenCL compiler strips
  // unused arguments, so this can be smaller than what the source declared.
  // setArg() then becomes a no-op for indices the optimized kernel no longer
  // exposes, instead of failing with CL_INVALID_ARG_INDEX at runtime.
  cl_uint actualNumArgs = 0;
  ComputeStatus argStatus = clGetKernelInfo(rawKernel, CL_KERNEL_NUM_ARGS,
                                            sizeof(actualNumArgs), &actualNumArgs, NULL);
  computeCheckError(argStatus, 0);

  ComputeKernel kernel(rawKernel, compute);
  kernel.numArgs = (uint)actualNumArgs;
  return kernel;
}

bool ComputeProgram::isEmpty()const
{
  return ref == NULL;
}


ComputeInterface::ComputeInterface()
  :heap(this, true)
{
  deviceId = NULL;
  context = NULL;
  queue = NULL;
}

ComputeInterface::~ComputeInterface()
{
}

void ComputeInterface::create(int deviceIndex)
{
  const uint MAX_STRING_LENGTH = 128;

  string selectedDevice;
  simdGroupSize = 16;

  ComputeDeviceId defaultDeviceId = NULL;
  ComputeDeviceId selectedDeviceId = NULL;
  int absoluteDeviceIndex = 0;
  uint platformCount = 0;
  // Picker state for the auto-select path: track the best non-Intel device
  // by max compute units across all platforms.
  size_t bestComputeUnits = 0;
  bool   bestIsIntel = true;

  devices.resize(8);

  cl_platform_id platforms[8];

  ComputeStatus status = clGetPlatformIDs(8, platforms, &platformCount);
  computeCheckError(status, 0);

  for (uint i = 0; i < platformCount; i++)
  {
    char vendor[MAX_STRING_LENGTH];
    char name[MAX_STRING_LENGTH];
    char version[MAX_STRING_LENGTH];
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, MAX_STRING_LENGTH - 1, vendor, NULL);
    computeCheckError(status, 0);
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, MAX_STRING_LENGTH - 1, name, NULL);
    computeCheckError(status, 0);
    status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, MAX_STRING_LENGTH - 1, version, NULL);
    computeCheckError(status, 0);

    logComputeMessage("Platform info:");
    logComputeMessage("  Platform vendor:  %s", vendor);
    logComputeMessage("  Platform name:    %s", name);
    logComputeMessage("  Platform version: %s", version);

    cl_platform_id platform = platforms[i];

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 8, devices.data(), &deviceCount);
    computeCheckError(status, 0);
    devices.resize(deviceCount);

    logComputeMessage("  Device info:");

    for (uint j = 0; j < deviceCount; j++)
    {
      ComputeDeviceId currentDeviceId = devices[j];
      // skip device if null, happens on OpenCL
      if (currentDeviceId == NULL) continue;

      size_t  maxWorkgroupSize;
      size_t  maxComputeUnits;
      size_t  maxWorkitemSizes[3];
      char    deviceName[MAX_STRING_LENGTH];
      char    deviceVendor[MAX_STRING_LENGTH];
      status = clGetDeviceInfo(currentDeviceId, CL_DEVICE_NAME, MAX_STRING_LENGTH - 1, deviceName, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(currentDeviceId, CL_DEVICE_VENDOR, MAX_STRING_LENGTH - 1, deviceVendor, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(currentDeviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &maxComputeUnits, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(currentDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkgroupSize, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(currentDeviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, maxWorkitemSizes, NULL);
      computeCheckError(status, 0);
      maxWorkgroupSize = 256;

      logComputeMessage("    Device Name:        %s", deviceName);
      logComputeMessage("    Compute Units:      %ld", maxComputeUnits);
      logComputeMessage("    Max Workgroup Size: %ld", maxWorkgroupSize);
      logComputeMessage("    Max Workitems:      %ld %ld %ld\n", maxWorkitemSizes[0], maxWorkitemSizes[1], maxWorkitemSizes[2]);

      // Auto-select the device with the most compute units (skipping Intel
      // when alternatives exist). On systems with both an iGPU and a dGPU
      // exposed by the same OpenCL platform (e.g. AMD APU + discrete via
      // RustICL), the iGPU may enumerate first and would otherwise win the
      // pick — leading to GPU watchdog timeouts under load.
      const string nameStr(deviceVendor);
      const bool isIntel = (nameStr.find("Intel") != nameStr.npos);

      bool takeThis = false;
      if (deviceIndex == -1)
      {
        // Auto-pick path: prefer non-Intel; among non-Intel, prefer most CUs.
        if (bestIsIntel && !isIntel) takeThis = true;
        else if (bestIsIntel == isIntel && maxComputeUnits > bestComputeUnits) takeThis = true;
      }
      else
      {
        // Manual override: caller passed an explicit deviceIndex.
        takeThis = (deviceIndex == absoluteDeviceIndex);
      }

      if (takeThis)
      {
        selectedDeviceId = currentDeviceId;
        this->maxThreadsPerWorkgroup = maxWorkgroupSize;
        selectedDevice = deviceName;
        bestComputeUnits = maxComputeUnits;
        bestIsIntel = isIntel;
        if (nameStr.find("AMD") != nameStr.npos)
        {
          simdGroupSize = 64;
        }
        if (nameStr.find("NVIDIA") != nameStr.npos)
        {
          simdGroupSize = 32;
        }
        if (nameStr.find("Apple") != nameStr.npos)
        {
          simdGroupSize = 32;
        }
      }

      if (absoluteDeviceIndex == 0)
      {
        defaultDeviceId = currentDeviceId;
      }
      absoluteDeviceIndex++;
    }
  }

  // if nothing found take the first one
  if (selectedDeviceId == NULL)
  {
    selectedDeviceId = defaultDeviceId;
    selectedDevice = "Default";
  }

  deviceId = selectedDeviceId;

  logComputeMessage("Selected device:   %s\nAssumed SIMD size: %ld", selectedDevice.c_str(), simdGroupSize);

  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &status);
  computeCheckError(status, 0);

  cl_command_queue_properties prop = 0;
#ifdef ENABLE_CL_PROFILING
  prop = CL_QUEUE_PROFILING_ENABLE;
#endif

  queue = clCreateCommandQueue(context, deviceId, prop, &status);
  computeCheckError(status, 0);
}

ComputeProgram ComputeInterface::createProgram(const char* sourceCode, size_t sourceSize,
                                               const stringArr* oldType, const stringArr* newType, const stringArr* includeFiles)
{
  ComputeStatus status;

  stringArr localOldType = oldType ? *oldType : stringArr();
  stringArr localNewType = newType ? *newType : stringArr();

  string defines;
  for (uint i = 0; i < localOldType.size(); i++)
  {
    defines += "#define " + localOldType[i] + " " + localNewType[i] + "\n";
  }

  string finalSource = defines + deepReadShaderSource(sourceCode, includeFiles);
  auto ptr = finalSource.c_str();
  sourceSize = finalSource.size();

  ComputeProgram program(clCreateProgramWithSource(context, 1, (const char **)&ptr, (const size_t *)&sourceSize, &status), this);
  computeCheckError(status, 0);

  const char* options = "-cl-std=CL1.2 -cl-mad-enable -cl-kernel-arg-info";
  status = clBuildProgram(program, 1, &deviceId, options, NULL, NULL);
  // Determine the size of the log
  size_t logSize;
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

  // Allocate memory for the log
  char* log = new char[logSize ? logSize : 1];
  log[0] = 0;

  // Get the log
  if (logSize)
  {
    clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
  }

  string logs = log;
  logs.erase(remove(logs.begin(), logs.end(), ' '), logs.end());
  logs.erase(remove(logs.begin(), logs.end(), '\n'), logs.end());

  if (logs.size() != 0)
  {
    logComputeMessage("Compilation Log:\n%s", log);
  }

  delete[] log;
  computeCheckError(status, 0);

  return program;
}


#ifdef ENABLE_CL_PROFILING
static void* registerKernelLaunched(ComputeKernel kernel, const size_t workgroup[3])
{
  char name[64];
  ComputeStatus status;
  status = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 63, name, NULL);
  computeCheckError(status, 0);

  int dispatchSize = (int)(workgroup[0] * workgroup[1] * workgroup[2]);

  std::string ret(name);
  ret += ": " + std::to_string(dispatchSize);

  return new std::string(ret + ": " + std::to_string(dispatchSize));
}

static void* registerBufferLaunched(const char name[64], const size_t bufferSize)
{
  int dispatchSize = (int)bufferSize;

  std::string ret(name);
  ret += ": " + std::to_string(dispatchSize);

  return new std::string(ret + ": " + std::to_string(dispatchSize));
}

static void eventCallback(cl_event event, cl_int event_command_exec_status, void *user_data)
{
  cl_ulong start, end;
  ComputeStatus status;

  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
  computeCheckError(status, 0);
  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
  computeCheckError(status, 0);

  logComputeMessage("%s time: %f", ((std::string*)user_data)->c_str(), (end - start) * 1.0e-6f);

  delete (std::string*)user_data;
}
#endif

void ComputeInterface::copyBuffer(const ComputeMemory* source, ComputeMemory* destination, size_t sourceOffset, size_t destinationOffset, size_t sizeInBytes)
{
#ifdef CREATE_SUB_BUFFER
  ComputeStatus status = clEnqueueCopyBuffer(queue, *source, *destination, sourceOffset, destinationOffset, sizeInBytes, 0, NULL, NULL);
#else
  ComputeStatus status = clEnqueueCopyBuffer(queue, *source, *destination, sourceOffset + source->getOffset(), destinationOffset + destination->getOffset(), sizeInBytes, 0, NULL, NULL);
#endif
  computeCheckError(status, 0);
}

void ComputeInterface::copyTextureToBuffer(const ComputeTexture*, ComputeMemory*, size_t, size_t, size_t)
{
  logComputeError("Function copyTextureToBuffer not implemented for OpenCL");
}

void ComputeInterface::copyBufferToTexture(const ComputeMemory*, ComputeTexture*, size_t, size_t, size_t)
{
  logComputeError("Function copyBufferToTexture not implemented for OpenCL");
}

void ComputeInterface::copyTexture(const ComputeTexture*, ComputeTexture*)
{
  logComputeError("Function copyTexture not implemented for OpenCL");
}

void ComputeInterface::setBuffer(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, const void* hostValue, size_t hostValueSize)
{
  ComputeStatus status = clEnqueueFillBuffer(queue, *source, hostValue, hostValueSize, sourceOffset, sizeInBytes, 0, NULL, NULL);
  computeCheckError(status, 0);
}

void ComputeInterface::copyToHost(const ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, void* hostPtr, bool waitForFinish)
{
  ComputeStatus status = clEnqueueReadBuffer(queue, *source, waitForFinish, sourceOffset, sizeInBytes, hostPtr, 0, NULL, NULL);
  computeCheckError(status, 0);
}

void ComputeInterface::copyFromHost(ComputeMemory* destination, size_t destinationOffset, size_t sizeInBytes, const void* hostPtr, bool waitForFinish)
{
  ComputeStatus status = clEnqueueWriteBuffer(queue, *destination, waitForFinish, destinationOffset, sizeInBytes, hostPtr, 0, NULL, NULL);
  computeCheckError(status, 0);
}

void ComputeInterface::execute(ComputeKernel& kernel, const size_t workgroupSize[3], const size_t workgroupCount[3])
{
  const size_t workgroup[3] = {
    workgroupSize[0] * workgroupCount[0],
    workgroupSize[1] * workgroupCount[1],
    workgroupSize[2] * workgroupCount[2] };

#ifdef DEBUG_KERNEL_TRACE
  {
    char kernelName[128] = {};
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kernelName), kernelName, NULL);
    fprintf(stderr, "[exec] %s wg=[%zu,%zu,%zu] cnt=[%zu,%zu,%zu]\n",
            kernelName, workgroupSize[0], workgroupSize[1], workgroupSize[2],
            workgroupCount[0], workgroupCount[1], workgroupCount[2]);
    fflush(stderr);
  }
#endif

  ComputeStatus status;

#ifdef ENABLE_CL_PROFILING
  cl_event localEvent;
  status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, workgroup, workgroupSize, 0, NULL, &localEvent);
  computeCheckError(status, 0);

  status = clSetEventCallback(localEvent, CL_COMPLETE, eventCallback, registerKernelLaunched(kernel, workgroup));
  computeCheckError(status, 0);

  clReleaseEvent(localEvent);
#else
  status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, workgroup, workgroupSize, 0, NULL, NULL);
  computeCheckError(status, 0);
#endif

#ifdef DEBUG_KERNEL_LAUNCHES
  ComputeStatus syncStatus = clFinish(queue);
  if (syncStatus != 0)
  {
    char kernelName[128] = {};
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kernelName), kernelName, NULL);
    logComputeMessage("Kernel '%s' faulted on sync: %s (code %d). Workgroup=[%zu,%zu,%zu] Count=[%zu,%zu,%zu]",
                      kernelName, getStatusMessage(syncStatus), (int)syncStatus,
                      workgroupSize[0], workgroupSize[1], workgroupSize[2],
                      workgroupCount[0], workgroupCount[1], workgroupCount[2]);
    computeCheckError(syncStatus, 0);
  }
#endif
}

void ComputeInterface::execute(ComputeKernel& kernel, const size_t workgroupSize[3], const ComputeMemory* workgroupCount, size_t bufferOffset)
{
  uint count = 0;
  copyToHost(workgroupCount, bufferOffset, 4, &count, true);

  size_t workgroupCountLocal[3] = {count, 1, 1};

  const size_t workgroup[3] = {
    workgroupSize[0] * workgroupCountLocal[0],
    workgroupSize[1] * workgroupCountLocal[1],
    workgroupSize[2] * workgroupCountLocal[2] };

  ComputeStatus status;

#ifdef ENABLE_CL_PROFILING
  cl_event localEvent;
  status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, workgroup, workgroupSize, 0, NULL, &localEvent);
  computeCheckError(status, 0);

  status = clSetEventCallback(localEvent, CL_COMPLETE, eventCallback, registerKernelLaunched(kernel, workgroup));
  computeCheckError(status, 0);

  clReleaseEvent(localEvent);
#else
  status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, workgroup, workgroupSize, 0, NULL, NULL);
  computeCheckError(status, 0);
#endif

#ifdef DEBUG_KERNEL_LAUNCHES
  ComputeStatus syncStatus = clFinish(queue);
  if (syncStatus != 0)
  {
    char kernelName[128] = {};
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kernelName), kernelName, NULL);
    logComputeMessage("Kernel '%s' (indirect) faulted on sync: %s (code %d). count=%u",
                      kernelName, getStatusMessage(syncStatus), (int)syncStatus, count);
    computeCheckError(syncStatus, 0);
  }
#endif
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
  ComputeStatus status;
  if (waitOnFinish)
  {
    status = clFinish(queue);
  }
  else
  {
    status = clFlush(queue);
  }
  computeCheckError(status, 0);
}

uint ComputeInterface::maxCores()const
{
  uint ret = 0;
  size_t retSize = 0;
  ComputeStatus status = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &ret, &retSize);
  computeCheckError(status, 0);
  return 64;
}

void ComputeInterface::startCapture()
{
  logComputeError("Start capture only defined for Metal");
}

void ComputeInterface::endCapture()
{
  logComputeError("End capture only defined for Metal");
}

#endif // USE_OPENCL_COMPUTE
