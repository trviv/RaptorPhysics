/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "ComputeInterface.h"
#include <unordered_set>

#if defined(USE_METAL_COMPUTE) || defined(USE_OPENCL_COMPUTE)
#define CREATE_SUB_BUFFER
//#define ENABLE_CL_PROFILING

#ifdef USE_METAL_COMPUTE
#ifndef DISABLE_PROFILING
#define ALWAYS_END_ENCODERS
#endif

//#define USE_ARGUMENT_BUFFERS

#if __has_feature(objc_arc)
#define retainComputeObj(obj)
#define releaseComputeObj(obj)  obj = nil;
#else
#define retainComputeObj(obj)   [obj retain];
#define releaseComputeObj(obj)  [obj release];  obj = nil;
#endif

id<MTLCaptureScope> captureScope = nil;
MTLCaptureManager *captureManager = nil;
MTLCaptureDescriptor* captureDescriptor = nil;
#define TEMP_BUFFER_OCCUPIED_FLAG 0x8000
static vector<pair<ushort, id<MTLBuffer>>> tempBuffers;
static unordered_map<id<MTLComputePipelineState>, id<MTLFunction>> kernelNameMap;
static unordered_map<string, uintPairList> kernelNameArgumentBufferMap;
pthread_mutex_t memMutex;

// 128 bytes aligned
uint alignAllocSize(uint minimumSize)
{
  return ((minimumSize & 0x7F) > 0) ? ((minimumSize & (~0x7F)) + 0x80) : minimumSize;
}

static id<MTLBuffer> getTempBuffer(uint minimumSize, ComputeDeviceId device)
{ @autoreleasepool {
  pthread_mutex_lock(&memMutex);
  // find a suitable candidate if available
  uint smallerSizeDifference = -1;
  uint biggerSizeDifference = -1;
  int smallerBufferIndex = -1;
  int biggerBufferIndex = -1;

  const int maxAllowedDifference = 1024;

  minimumSize = alignAllocSize(minimumSize);

  for (int i=0; i<tempBuffers.size(); i++)
  {
    // if not occupied
    if (tempBuffers[i].first & TEMP_BUFFER_OCCUPIED_FLAG)
      continue;

    const int sizeDiff  = abs((int)tempBuffers[i].second.length - (int)minimumSize);

    if (sizeDiff > maxAllowedDifference)
    {
      tempBuffers[i].first++;
      if (tempBuffers[i].first == 0x3ff)
      {
        releaseComputeObj(tempBuffers.begin()->second);
        tempBuffers.erase(tempBuffers.begin() + i);
        i--;
      }
      continue;
    }

    if (tempBuffers[i].second.length >= minimumSize && sizeDiff < biggerSizeDifference)
    {
      biggerBufferIndex = i;
      biggerSizeDifference = (int)tempBuffers[i].second.length - minimumSize;
      if (biggerSizeDifference == 0)
      {
        //break;
      }
    }
    if (tempBuffers[i].second.length < minimumSize && sizeDiff < smallerSizeDifference)
    {
      smallerBufferIndex = i;
      smallerSizeDifference = minimumSize - (int)tempBuffers[i].second.length;
    }
  }

  id<MTLBuffer> retBuffer = nullptr;
  // if both found
  if (smallerSizeDifference != -1 && biggerSizeDifference != -1)
  {
    // expand smaller if its closer in size
    if (smallerSizeDifference < biggerSizeDifference)
    {
      tempBuffers[smallerBufferIndex] = pair<ushort, id<MTLBuffer>>(TEMP_BUFFER_OCCUPIED_FLAG, [device newBufferWithLength:minimumSize options:MTLResourceStorageModeShared]);
      retBuffer = tempBuffers[smallerBufferIndex].second;
    }
    // if bigger is closer in size return it
    else
    {
      tempBuffers[biggerBufferIndex].first = TEMP_BUFFER_OCCUPIED_FLAG;
      retBuffer = tempBuffers[biggerBufferIndex].second;
    }
  }
  else if (biggerSizeDifference != -1)
  {
    tempBuffers[biggerBufferIndex].first = TEMP_BUFFER_OCCUPIED_FLAG;
    retBuffer = tempBuffers[biggerBufferIndex].second;
  }
  else if (smallerSizeDifference != -1)
  {
    tempBuffers[smallerBufferIndex] = pair<ushort, id<MTLBuffer>>(TEMP_BUFFER_OCCUPIED_FLAG, [device newBufferWithLength:minimumSize options:MTLResourceStorageModeShared]);
    retBuffer = tempBuffers[smallerBufferIndex].second;
  }
  else
  {
    tempBuffers.push_back(pair<ushort, id<MTLBuffer>>(TEMP_BUFFER_OCCUPIED_FLAG, [device newBufferWithLength:minimumSize options:MTLResourceStorageModeShared]));
    retBuffer = tempBuffers.back().second;
  }

  pthread_mutex_unlock(&memMutex);
  return retBuffer;
}}

static void freeTempBuffer(id<MTLBuffer> buffer)
{ @autoreleasepool {
  pthread_mutex_lock(&memMutex);
  for (int i=0; i<tempBuffers.size(); i++)
  {
    // if not occupied
    if (tempBuffers[i].second == buffer)
    {
      tempBuffers[i].first = 0;
      pthread_mutex_unlock(&memMutex);
      return;
    }
  }
  pthread_mutex_unlock(&memMutex);
}}

id<MTLBlitCommandEncoder> ComputeInterface::getBlitEncoder()
{ @autoreleasepool {
  if (currentComputeEncoder != nil)
  {
    [currentComputeEncoder endEncoding];
    releaseComputeObj(currentComputeEncoder);
  }
  if (currentBlitEncoder != nil)
  {
    return currentBlitEncoder;
  }
  if (currentCommandBuffer == nil)
  {
    currentCommandBuffer = [commandQueue commandBufferWithUnretainedReferences];
    retainComputeObj(currentCommandBuffer);
  }
  currentBlitEncoder = [currentCommandBuffer blitCommandEncoder];
  retainComputeObj(currentBlitEncoder);
  return currentBlitEncoder;
}}

id<MTLComputeCommandEncoder> ComputeInterface::getComputeEncoder()
{ @autoreleasepool {
  if (currentBlitEncoder != nil)
  {
    [currentBlitEncoder endEncoding];
    releaseComputeObj(currentBlitEncoder);
  }
  if (currentComputeEncoder != nil)
  {
    return currentComputeEncoder;
  }
  if (currentCommandBuffer == nil)
  {
    currentCommandBuffer = [commandQueue commandBufferWithUnretainedReferences];
    retainComputeObj(currentCommandBuffer);
  }
  currentComputeEncoder = [currentCommandBuffer computeCommandEncoder];
  retainComputeObj(currentComputeEncoder);
  return currentComputeEncoder;
}}

void ComputeInterface::endEncoders()
{ @autoreleasepool {
  if (currentComputeEncoder != nil)
  {
    [currentComputeEncoder endEncoding];
    releaseComputeObj(currentComputeEncoder);
  }
  if (currentBlitEncoder != nil)
  {
    [currentBlitEncoder endEncoding];
    releaseComputeObj(currentBlitEncoder);
  }
}}

#endif

const char* getStatusMessage(ComputeStatus status)
{
#ifdef USE_OPENCL_COMPUTE
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
#else

  return "No status";

#endif
}

MTLResourceOptions getResourceOption(ComputeMemoryStorage storage) {
  switch(storage) {
    case Default:
#if TARGET_OS_IPHONE
      return MTLResourceStorageModeShared;
#else
      return MTLResourceStorageModePrivate;
#endif
    case Private:
      return MTLResourceStorageModePrivate;
    case Shared:
      return MTLResourceStorageModeShared;
    case Managed:
#if TARGET_OS_IPHONE
      return MTLResourceStorageModeShared;
#else
      return MTLResourceStorageModeManaged;
#endif
  }
  return MTLResourceStorageModePrivate;
}

ComputeMemory* ComputeHeap::alloc(size_t sizeInBytes, void* data, ComputeMemoryStorage storage, ComputeMemoryUsage flag)
{
  sizeInBytes = mAlignBy(sizeInBytes, 16) * 16;
  ComputeMemory* ret;
  ComputeStatus status;
  if (bypass)
  {
#ifdef USE_OPENCL_COMPUTE
    ret = new ComputeMemory(
      clCreateBuffer(compute->context, flag, sizeInBytes, data, &status),
      0, sizeInBytes);
    computeCheckError(status, 0);
#else
    ret = new ComputeMemory([compute->context newBufferWithLength:sizeInBytes options:getResourceOption(storage)], 0, sizeInBytes, storage);
#endif
  }
  else
  {
    size_t offset = 0;
    if (childs.size())
    {
      const ComputeMemory* last = childs.back();
      offset = last->getOffset() + last->getSize();
    }
#ifdef USE_OPENCL_COMPUTE
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
#else
    ret = new ComputeMemory(heap, offset, sizeInBytes);
#endif
  }
  childs.push_back(ret);

  return ret;
}

void ComputeHeap::free(ComputeMemory* memory)
{
#ifdef USE_OPENCL_COMPUTE

#ifdef CREATE_SUB_BUFFER
  ComputeStatus status = clReleaseMemObject(*memory);
  computeCheckError(status, 0);
#endif

#else

#ifdef CREATE_SUB_BUFFER
  memory->ref = nullptr;
#endif

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

void ComputeKernel::addArgumentBufferRange(uint startIndex, uint inclusiveEndIndex)
{
#ifdef USE_METAL_COMPUTE
  argumentBufferRange.push_back(uintPair(startIndex, inclusiveEndIndex));
  setArgumentBuffer = true;
#else
#endif
}

void ComputeKernel::setArg(void* valuePtr, size_t valueSize, uint index)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clSetKernelArg(ref, index, valueSize, valuePtr);
  computeCheckError(status, 0);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [compute->getComputeEncoder() setBytes:valuePtr length:valueSize atIndex:index];
  }
  else
  {
    ArgData data;
    data.type = ARG_DATA_HOST_PTR;
    data.cptr = valuePtr;
    data.size = (uint)valueSize;
    data.index = index;
    args.push_back(data);
  }
#endif
}

void ComputeKernel::setArg(const void* valuePtr, size_t valueSize, uint index)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clSetKernelArg(ref, index, valueSize, valuePtr);
  computeCheckError(status, 0);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [compute->getComputeEncoder() setBytes:valuePtr length:valueSize atIndex:index];
  }
  else
  {
    ArgData data;
    data.type = ARG_DATA_CONST_HOST_PTR;
    data.cptr = valuePtr;
    data.size = (uint)valueSize;
    data.index = index;
    args.push_back(data);
  }
#endif
}

void ComputeKernel::setArg(const ComputeMemory* buffer, uint index)
{
  ComputeMemoryIdentifier ident = *buffer;
#ifdef USE_OPENCL_COMPUTE
  setArg<ComputeMemoryIdentifier>(&ident, index);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [compute->getComputeEncoder() setBuffer:ident offset:buffer->getOffset() atIndex:index];
    [compute->getComputeEncoder() useResource:ident usage:MTLResourceUsageRead];
  }
  else
  {
    ArgData data;
    data.type = ARG_DATA_CONST_DEVICE_PTR;
    data.cptr = buffer;
    data.index = index;
    args.push_back(data);
  }
#endif
}

void ComputeKernel::setArg(ComputeMemory* buffer, uint index)
{
  ComputeMemoryIdentifier ident = *buffer;
#ifdef USE_OPENCL_COMPUTE
  setArg<ComputeMemoryIdentifier>(&ident, index);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [compute->getComputeEncoder() setBuffer:ident offset:buffer->getOffset() atIndex:index];
  }
  else
  {
    ArgData data;
    data.type = ARG_DATA_DEVICE_PTR;
    data.cptr = buffer;
    data.index = index;
    args.push_back(data);
  }
#endif
}

void ComputeKernel::setSharedMemArg(const size_t valueSize, uint index)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clSetKernelArg(ref, index, valueSize, NULL);
  computeCheckError(status, 0);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [compute->getComputeEncoder() setThreadgroupMemoryLength:valueSize atIndex:index];
  }
  else
  {
    ArgData data;
    data.type = ARG_DATA_SHARED_PTR;
    data.size = (uint)valueSize;
    data.index = index;
    args.push_back(data);
  }
#endif
}

#ifndef USE_OPENCL_COMPUTE
void ComputeKernel::setArgs()
{
  if (setArgumentBuffer)
  @autoreleasepool {
  for (uint r=0; r<argumentBufferRange.size(); r++)
  {
    const auto& range = argumentBufferRange[r];
    id <MTLArgumentEncoder> argumentEncoder = [kernelNameMap[ref] newArgumentEncoderWithBufferIndex:range.first];
    id <MTLBuffer>          argumentBuffer  = [ref.device newBufferWithLength:argumentEncoder.encodedLength options:0];

    [argumentEncoder setArgumentBuffer:argumentBuffer offset:0];
    argumentBuffers[mapArgumentIndex(range.first)] = argumentBuffer;

    for (uint i=0; i<args.size(); i++)
    {
      auto &arg = args[i];
      if (arg.index >= range.first && arg.index <= range.second)
      {
        uint index = (arg.index - range.first);
        switch (arg.type)
        {
          case ARG_DATA_HOST_PTR:
          case ARG_DATA_CONST_HOST_PTR:
          case ARG_DATA_SHARED_PTR:
            memcpy([argumentEncoder constantDataAtIndex:index], arg.cptr, arg.size);
            break;
          case ARG_DATA_DEVICE_PTR:
          case ARG_DATA_CONST_DEVICE_PTR:
            [argumentEncoder setBuffer:*((const ComputeMemory*)arg.cptr) offset:((const ComputeMemory*)arg.cptr)->getOffset() atIndex:index];
            break;
          default:
            logComputeError("Unknown Argument type!");
            break;
        }
      }
    }
  }
  setArgumentBuffer = false;
  }
  for (auto& range : argumentBufferRange)
  @autoreleasepool {
    [compute->getComputeEncoder() setBuffer:argumentBuffers[mapArgumentIndex(range.first)] offset:0 atIndex:range.first];
  }
  @autoreleasepool {
  for (auto& i : args)
  {
    bool skip = false;
    for (const auto& range : argumentBufferRange)
    {
      if (i.index >= range.first && i.index <= range.second)
      {
        skip = true;
      }
    }

    if (skip) continue;

    switch (i.type)
    {
      case ARG_DATA_HOST_PTR:
      case ARG_DATA_CONST_HOST_PTR:
      case ARG_DATA_SHARED_PTR:
        [compute->getComputeEncoder() setBytes:i.cptr length:i.size atIndex:i.index];
        break;
      case ARG_DATA_DEVICE_PTR:
      case ARG_DATA_CONST_DEVICE_PTR:
        [compute->getComputeEncoder() setBuffer:*((const ComputeMemory*)i.cptr) offset:((const ComputeMemory*)i.cptr)->getOffset() atIndex:i.index];
        break;
      default:
        logComputeError("Unknown Argument type!");
        break;
    }
  }
  args.clear();
}}
#endif

void ComputeKernel::registerResource(const ComputeMemory* resource)
{
#ifdef USE_OPENCL_COMPUTE
  logComputeError("Register resource is not implemented!");
#else
  [compute->getComputeEncoder() useResource:*resource usage:MTLResourceUsageRead];
#endif
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
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status;
  ComputeKernel kernel(clCreateKernel(ref, kernelName, &status));
  computeCheckError(status, 0);
  return kernel;
#else
  @autoreleasepool {
  // map the function to kernel so it can be retrived later
  id<MTLFunction> function = [ref newFunctionWithName:[NSString stringWithCString:kernelName encoding:NSASCIIStringEncoding]];
  NSError* error;

  MTLComputePipelineDescriptor* pipelineDesc = [MTLComputePipelineDescriptor new];
  pipelineDesc.computeFunction = function;
  pipelineDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
  for (uint i=0 ; kernelNameArgumentBufferMap.count(kernelName) && i<kernelNameArgumentBufferMap[kernelName].size(); i++)
  {
    pipelineDesc.buffers[kernelNameArgumentBufferMap[kernelName][i].first].mutability = MTLMutabilityImmutable;
  }

  ComputeKernel ret = ComputeKernel([ref.device newComputePipelineStateWithDescriptor:pipelineDesc options:0 reflection:0 error:&error], compute);
  kernelNameMap[ret] = function;

  for (uint i=0 ; kernelNameArgumentBufferMap.count(kernelName) && i<kernelNameArgumentBufferMap[kernelName].size(); i++)
  {
    const auto& r = kernelNameArgumentBufferMap[kernelName][i];
    ret.addArgumentBufferRange(r.first, r.second);
  }
  return ret;
  }
#endif
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

void ComputeInterface::create(int deviceIndex)
{
  const uint MAX_STRING_LENGTH = 128;

  string selectedDevice;
  simdGroupSize = 16;

  ComputeDeviceId defaultDeviceId;
  ComputeDeviceId selectedDeviceId;
  int absoluteDeviceIndex = 0;
  uint platformCount = 0;

  devices.resize(8);

#ifdef USE_OPENCL_COMPUTE
  cl_platform_id platforms[8];

  // print platform info
  ComputeStatus status = clGetPlatformIDs(8, platforms, &platformCount);
  computeCheckError(status, 0);
#else
  platformCount = 1;
#endif

  for (uint i = 0; i < platformCount; i++)
  {
#ifdef USE_OPENCL_COMPUTE
    int status;
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
#else
#if TARGET_OS_OSX
    NSArray<id<MTLDevice>> *localDevices = MTLCopyAllDevices();
#else
    NSArray<id<MTLDevice>> *localDevices = [[NSArray<id<MTLDevice>> alloc] initWithObjects:MTLCreateSystemDefaultDevice(), nil];
#endif
    deviceCount = (uint)localDevices.count;
    for (int i=0; i<deviceCount; i++)
    {
      devices[i] = [localDevices objectAtIndex:i];
    }
    releaseComputeObj(localDevices);
#endif
    devices.resize(deviceCount);

    logComputeMessage("  Device info:");

    // print device info and select one if not supplied
    for (uint j = 0; j < deviceCount; j++)
    {
      ComputeDeviceId deviceId = devices[j];
#ifdef USE_OPENCL_COMPUTE
      // skip device if null, happens on OpenCL
      if (deviceId == NULL) continue;

      int     status;
      size_t  maxWorkgroupSize;
      size_t  maxComputeUnits;
      size_t  maxWorkitemSizes[3];
      char    deviceName[MAX_STRING_LENGTH];
      char    deviceVendor[MAX_STRING_LENGTH];
      status = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, MAX_STRING_LENGTH - 1, deviceName, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(deviceId, CL_DEVICE_VENDOR, MAX_STRING_LENGTH - 1, deviceVendor, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &maxComputeUnits, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkgroupSize, NULL);
      computeCheckError(status, 0);
      status = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, maxWorkitemSizes, NULL);
      computeCheckError(status, 0);
      maxWorkgroupSize = 256;
#else
#if TARGET_OS_IPHONE
      // TODO: investigate 1024 thread group size not working with irregular reduce
      size_t  maxWorkgroupSize = 256;//deviceId.maxThreadsPerThreadgroup.width;
#else
      // TODO: remove this size once other issues are resolved.
      size_t  maxWorkgroupSize = 256;//deviceId.maxThreadsPerThreadgroup.width;
#endif
      size_t  maxComputeUnits = 64;
      size_t  maxWorkitemSizes[3] = {0, 0, 0};
      const char *deviceName = deviceId.name.UTF8String;
      const char *deviceVendor = deviceId.name.UTF8String;
#endif
      logComputeMessage("    Device Name:        %s", deviceName);
      logComputeMessage("    Compute Units:      %ld", maxComputeUnits);
      logComputeMessage("    Max Workgroup Size: %ld", maxWorkgroupSize);
      logComputeMessage("    Max Workitems:      %ld %ld %ld\n", maxWorkitemSizes[0], maxWorkitemSizes[1], maxWorkitemSizes[2]);

      if (deviceIndex == -1)
      {
        string name(deviceVendor);
        size_t endIndex = name.find("Intel");

        // select any other device over intel
        if (endIndex == name.npos)
        {
          deviceIndex = absoluteDeviceIndex;
        }
      }

      if (deviceIndex == absoluteDeviceIndex)
      {
        selectedDeviceId = deviceId;
        this->maxThreadsPerWorkgroup = maxWorkgroupSize;
        selectedDevice = deviceName;
        string name(deviceVendor);
        if (name.find("AMD") != name.npos)
        {
          simdGroupSize = 64;
        }
        if (name.find("NVIDIA") != name.npos)
        {
          simdGroupSize = 32;
        }
        if (name.find("Apple") != name.npos)
        {
          simdGroupSize = 32;
        }
      }

      if (absoluteDeviceIndex == 0)
      {
        defaultDeviceId = deviceId;
      }
      absoluteDeviceIndex++;
    }
  }

  // if nothing found take the first one
  if (deviceIndex == -1)
  {
    selectedDeviceId = defaultDeviceId;
    selectedDevice = "Default";
  }

  deviceId = selectedDeviceId;

  logComputeMessage("Selected device:   %s\nAssumed SIMD size: %ld", selectedDevice.c_str(), simdGroupSize);

#ifdef USE_OPENCL_COMPUTE
  /* Create OpenCL context */
  context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &status);
  computeCheckError(status, 0);

  cl_command_queue_properties prop = NULL;
#ifdef ENABLE_CL_PROFILING
  prop = CL_QUEUE_PROFILING_ENABLE;
#endif

  queue = clCreateCommandQueue(context, deviceId, prop, &status);
  computeCheckError(status, 0);
#else
  @autoreleasepool {
  // initialize metal objects
  queue = [deviceId newCommandQueue];
  retainComputeObj(queue);
  tempBuffers.clear();
  currentCommandBuffer = [queue commandBufferWithUnretainedReferences];
  retainComputeObj(currentCommandBuffer);
  commandQueue = queue;
  captureManager = [MTLCaptureManager sharedCaptureManager];
  captureScope = [captureManager newCaptureScopeWithCommandQueue:queue];
  captureDescriptor = [[MTLCaptureDescriptor alloc] init];
  captureDescriptor.captureObject = deviceId;
  context = deviceId;
  }
#endif
}

ComputeProgram ComputeInterface::createProgram(const char* sourceCode, size_t sourceSize,
                                               const stringArr* oldType, const stringArr* newType, const stringArr* includeFiles)
{
  ComputeStatus status;
#ifdef USE_OPENCL_COMPUTE

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

  const char* options = "-cl-std=CL1.2 -cl-mad-enable";
  status = clBuildProgram(program, 1, &deviceId, options, NULL, NULL);
  // Determine the size of the log
  size_t logSize;
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

  // Allocate memory for the log
  char* log = new char[logSize];

  // Get the log
  clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
#else
  ComputeProgram program;
  char* log = new char[1];
  log[0] = NULL;
  @autoreleasepool {
  NSError *error = nil;

  stringArr localOldType = oldType ? *oldType : stringArr();
  stringArr localNewType = newType ? *newType : stringArr();

  NSString *source = @"";
  if ([deviceId supportsFamily:MTLGPUFamilyApple6])
  {
    localOldType.push_back("USE_SIMD_COMPUTE");
    localNewType.push_back("");
  }

  string defines;
  for (uint i = 0; i < localOldType.size(); i++)
  {
    defines += "#define " + localOldType[i] + " " + localNewType[i] + "\n";
  }

  string finalSource = defines + deepReadShaderSource(sourceCode, includeFiles);

#ifdef USE_ARGUMENT_BUFFERS
  finalSource = processAutoArgumentBuffers(finalSource, kernelNameArgumentBufferMap);
#endif

  source = [source stringByAppendingString:[NSString stringWithUTF8String:finalSource.c_str()]];
  ComputeProgramIdentifier programId = [deviceId newLibraryWithSource:source options:0 error:&error];
  program = ComputeProgram(programId, this);
  // Allocate memory for the log
  if (error != nil)
  {
    delete[] log;
    log = new char[error.description.length + 1];
    strcpy(log, error.description.UTF8String);
  }
  status = (programId == nil);
  }
#endif

  string logs = log;
  logs.erase(remove(logs.begin(), logs.end(), ' '), logs.end());
  logs.erase(remove(logs.begin(), logs.end(), '\n'), logs.end());

  if (logs.size() != 0)
  {
    // Print the log
    logComputeMessage("Compilation Log:\n%s", log);
  }

  delete[] log;
  computeCheckError(status, 0);

  return program;
}


#ifdef ENABLE_CL_PROFILING
void* registerKernelLaunched(ComputeKernel kernel, const size_t workgroup[3])
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

void* registerBufferLaunched(const char name[64], const size_t bufferSize)
{
  int dispatchSize = (int)bufferSize;

  std::string ret(name);
  ret += ": " + std::to_string(dispatchSize);

  return new std::string(ret + ": " + std::to_string(dispatchSize));
}

void eventCallback(cl_event event, cl_int event_command_exec_status, void *user_data)
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
#ifdef USE_OPENCL_COMPUTE
  cl_event localEvent;
#ifdef CREATE_SUB_BUFFER
  ComputeStatus status = clEnqueueCopyBuffer(queue, *source, *destination, sourceOffset, destinationOffset, sizeInBytes, 0, NULL, &localEvent);
#else
  ComputeStatus status = clEnqueueCopyBuffer(queue, *source, *destination, sourceOffset + source->getOffset(), destinationOffset + destination->getOffset(), sizeInBytes, 0, NULL, &localEvent);
#endif
  computeCheckError(status, 0);

#ifdef ENABLE_CL_PROFILING
  status = clSetEventCallback(localEvent, CL_COMPLETE, eventCallback, registerBufferLaunched("copyBuffer", sizeInBytes));
  computeCheckError(status, 0);

  clReleaseEvent(localEvent);
#endif
#else
  @autoreleasepool {
  [getBlitEncoder() copyFromBuffer:*source sourceOffset:(sourceOffset + source->getOffset()) toBuffer:*destination destinationOffset:(destinationOffset + destination->getOffset()) size:sizeInBytes];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif
  }
#endif
}

void ComputeInterface::copyTextureToBuffer(const ComputeTexture* source, ComputeMemory* destination, size_t destinationOffset, size_t sourceSlice, size_t sourceLevel)
{
#ifdef USE_OPENCL_COMPUTE
  logComputeError("Function copyTextureToBuffer not implemented for OpenCL");
#else
  @autoreleasepool {
  [getBlitEncoder() copyFromTexture:*source
                        sourceSlice:sourceSlice
                        sourceLevel:sourceLevel
                       sourceOrigin:MTLOriginMake(0, 0, 0)
                         sourceSize:MTLSizeMake(source->getSize()[0], source->getSize()[1], 1)
                           toBuffer:*destination
                  destinationOffset:(destinationOffset + destination->getOffset())
             destinationBytesPerRow:source->getSize()[0] * source->getBytesPerPixel()
           destinationBytesPerImage:source->getSize()[0] * source->getSize()[1] * source->getBytesPerPixel()];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif
  }
#endif
}

void ComputeInterface::copyBufferToTexture(const ComputeMemory* source, ComputeTexture* destination, size_t sourceOffset, size_t destinationSlice, size_t destinationLevel)
{
#ifdef USE_OPENCL_COMPUTE
  logComputeError("Function copyTextureToBuffer not implemented for OpenCL");
#else
  @autoreleasepool {
  [getBlitEncoder() copyFromBuffer:*source
                      sourceOffset:sourceOffset
                 sourceBytesPerRow:destination->getSize()[0] * destination->getBytesPerPixel()
               sourceBytesPerImage:destination->getSize()[0] * destination->getSize()[1] * destination->getBytesPerPixel()
                        sourceSize:MTLSizeMake(destination->getSize()[0], destination->getSize()[1], 1)
                         toTexture:*destination
                  destinationSlice:destinationSlice
                  destinationLevel:destinationLevel
                 destinationOrigin:MTLOriginMake(0, 0, 0)];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif
  }
#endif
}

void ComputeInterface::copyTexture(const ComputeTexture* source, ComputeTexture* destination)
{
#ifdef USE_OPENCL_COMPUTE
  logComputeError("Function copyTexture not implemented for OpenCL");
#else
  @autoreleasepool {
  [getBlitEncoder() copyFromTexture:*source toTexture:*destination];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif
  }
#endif
}

void ComputeInterface::setBuffer(ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, const void* hostValue, size_t hostValueSize)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clEnqueueFillBuffer(queue, *source, hostValue, hostValueSize, sourceOffset, sizeInBytes, 0, NULL, NULL);
  computeCheckError(status, 0);
#endif
}

void ComputeInterface::copyToHost(const ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, void* hostPtr, bool waitForFinish)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clEnqueueReadBuffer(queue, *source, waitForFinish, sourceOffset, sizeInBytes, hostPtr, 0, NULL, NULL);
  computeCheckError(status, 0);
#else
  @autoreleasepool {
  id<MTLBuffer> tempBuffer = getTempBuffer((uint)sizeInBytes, getDevice());
  [getBlitEncoder() copyFromBuffer:*source sourceOffset:(sourceOffset + source->getOffset()) toBuffer:tempBuffer destinationOffset:0 size:sizeInBytes];
  [currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
    memcpy(hostPtr, tempBuffer.contents, sizeInBytes);
    freeTempBuffer(tempBuffer);
  }];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif

  if (waitForFinish)
  {
    sync();
//    freeTempBuffer(tempBuffer);
  }}
#endif
}

void ComputeInterface::copyFromHost(ComputeMemory* destination, size_t destinationOffset, size_t sizeInBytes, const void* hostPtr, bool waitForFinish)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clEnqueueWriteBuffer(queue, *destination, waitForFinish, destinationOffset, sizeInBytes, hostPtr, 0, NULL, NULL);
  computeCheckError(status, 0);
#else
#if !TARGET_OS_IPHONE
  @autoreleasepool {
  id<MTLBuffer> tempBuffer = getTempBuffer((uint)sizeInBytes, getDevice());
  memcpy(tempBuffer.contents, hostPtr, sizeInBytes);
  [getBlitEncoder() copyFromBuffer:tempBuffer sourceOffset:0 toBuffer:*destination destinationOffset:(destinationOffset + destination->getOffset()) size:sizeInBytes];
  [currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
    freeTempBuffer(tempBuffer);
  }];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif

  if (waitForFinish)
  {
    sync();
//    freeTempBuffer(tempBuffer);
  }}
#else
  memcpy((char*)(id<MTLBuffer>(*destination)).contents + (destinationOffset + destination->getOffset()), hostPtr, sizeInBytes);
#endif
#endif
}

void ComputeInterface::execute(ComputeKernel& kernel, const size_t workgroupSize[3], const size_t workgroupCount[3])
{
#ifdef USE_OPENCL_COMPUTE
  const size_t workgroup[3] = {
    workgroupSize[0] * workgroupCount[0],
    workgroupSize[1] * workgroupCount[1],
    workgroupSize[2] * workgroupCount[2] };

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
#else
  @autoreleasepool {
  // get kernel name
  NSString* kernelName = [kernelNameMap[kernel] name];

  // populate encoder
  id<MTLComputeCommandEncoder> encoder = getComputeEncoder();
  kernel.setArgs();
  // set dispatch info
  [encoder setLabel:[NSString stringWithFormat:@"%@: %d", kernelName,
                     (int)(workgroupCount[0]*workgroupCount[1]*workgroupCount[2]*workgroupSize[0]*workgroupSize[1]*workgroupSize[2])]];
  [encoder setComputePipelineState:kernel];
  [encoder dispatchThreadgroups:MTLSizeMake(workgroupCount[0], workgroupCount[1], workgroupCount[2])
          threadsPerThreadgroup:MTLSizeMake(workgroupSize[0], workgroupSize[1], workgroupSize[2])];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif
  }
#endif
}

void ComputeInterface::execute(ComputeKernel& kernel, const size_t workgroupSize[3], const ComputeMemory* workgroupCount, size_t bufferOffset)
{
#ifdef USE_OPENCL_COMPUTE
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
#else
  @autoreleasepool {
  // get kernel name
  NSString* kernelName = [kernelNameMap[kernel] name];

  // populate encoder
  id<MTLComputeCommandEncoder> encoder = getComputeEncoder();
  kernel.setArgs();
  // set dispatch info
  [encoder setLabel:[NSString stringWithFormat:@"%@: %d", kernelName, (int)(workgroupSize[0]*workgroupSize[1]*workgroupSize[2])]];
  [encoder setComputePipelineState:kernel];
  [encoder dispatchThreadgroupsWithIndirectBuffer:(*workgroupCount)
                             indirectBufferOffset:workgroupCount->getOffset()+bufferOffset
                            threadsPerThreadgroup:MTLSizeMake(workgroupSize[0], workgroupSize[1], workgroupSize[2])];
#ifdef ALWAYS_END_ENCODERS
  endEncoders();
#endif
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
#ifdef USE_OPENCL_COMPUTE
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
#else
  @autoreleasepool {
  endEncoders();
  if (currentCommandBuffer)
  {
    [currentCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
      executionTime = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.f;
    }];
    [currentCommandBuffer commit];
    if (waitOnFinish)
    {
      [currentCommandBuffer waitUntilCompleted];
    }
    releaseComputeObj(currentCommandBuffer);
  }}
#endif
}

uint ComputeInterface::maxCores()const
{
#ifdef USE_OPENCL_COMPUTE
  uint ret = 0;
  size_t retSize = 0;
  ComputeStatus status = clGetDeviceInfo(deviceId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &ret, &retSize);
  computeCheckError(status, 0);
  return 64;
#else
  return 64;
#endif
}

void ComputeInterface::startCapture()
{
#ifdef USE_METAL_COMPUTE
  @autoreleasepool {
  endEncoders();
  [currentCommandBuffer commit];

  NSError *error;
  if (![captureManager startCaptureWithDescriptor:captureDescriptor error:&error])
  {
    NSLog(@"Failed to start capture, error %@", error);
  }

  [captureScope beginScope];
  currentCommandBuffer = [queue commandBufferWithUnretainedReferences];
  retainComputeObj(currentCommandBuffer);
  getComputeEncoder();
  }
#else
  logComputeError("Start captured only defined for Metal");
#endif
}

void ComputeInterface::endCapture()
{
#ifdef USE_METAL_COMPUTE
  @autoreleasepool {
  endEncoders();
  [currentCommandBuffer commit];
  [captureScope endScope];
  [captureManager stopCapture];
  currentCommandBuffer = [queue commandBufferWithUnretainedReferences];
  retainComputeObj(currentCommandBuffer);
  getComputeEncoder();
  }
#else
  logComputeError("End captured only defined for Metal");
#endif
}

#endif
