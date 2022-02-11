#include "ComputeInterface.h"

#define CREATE_SUB_BUFFER
//#define ENABLE_CL_PROFILING

static ComputePlatformId platforms[8];
static uint platformCount = 0;

static ComputeDeviceId devices[8];
static uint deviceCount = 0;

typedef pair<uint, uint>  uintPair;
typedef vector<uintPair>  uintPairList;
typedef vector<string>    stringList;

inline string trim(const string& str, const char* delim = " ") //" \t\r\n"
{
  size_t first = str.find_first_not_of(delim);
  if (string::npos == first)
  {
    return str;
  }
  return str.substr(first, (str.find_last_not_of(delim) - first + 1));
}

inline bool isComment(const string& input, const size_t startIndex)
{
  // returns true when / is succeeded by / or *
  return input[startIndex] == '/' && (startIndex+1) < input.size() && (input[startIndex+1] == '/' || input[startIndex+1] == '*');
}

inline size_t skipComment(const string& input, size_t startIndex)
{
  startIndex++;
  if (input[startIndex] == '/')
  {
    // go till next line in case of //
    while (startIndex < input.size() && input[startIndex] != '\n')
    {
      startIndex++;
    }
  }
  if (input[startIndex] == '*')
  {
    // find first */ in case of /*
    while ((startIndex < input.size() && input[startIndex] != '*') || ((startIndex+1) < input.size() && input[startIndex+1] != '/'))
    {
      startIndex++;
    }
  }
  return startIndex;
}

inline size_t findEndOfFunction(const string& input, const size_t startIndex, const char item)
{
  vector<char> bracketStack;
  const string openBrackets   = "({[";
  const string closeBrackets  = ")}]";

  for (size_t curr=startIndex; curr<input.size(); curr++)
  {
    const char c = input[curr];

    if (item == c)
    {
      if (bracketStack.size() == 0)
      {
        return curr;
      }
    }

    if (isComment(input, curr))
    {
      curr = skipComment(input, curr);
      continue;
    }

    const size_t openIndex = openBrackets.find(c);
    if (openIndex != openBrackets.npos)
    {
      bracketStack.push_back(closeBrackets[openIndex]);
    }
    else if (closeBrackets.find(c) != openBrackets.npos)
    {
      if (!bracketStack.empty() && bracketStack.back() == c)
      {
        bracketStack.pop_back();
      }
    }
  }

  return input.npos;
}

inline stringList tokenize(const string& input, const string delimiter = " \t\r\n")
{
  stringList tokens;
  string temp;
  size_t curr = 0;
  size_t prev = 0;
  vector<char> bracketStack;
  const string openBrackets   = "({[";
  const string closeBrackets  = ")}]";

  for (uint curr=0; curr<input.size(); curr++)
  {
    const char c = input[curr];

    if (delimiter.find(c) != delimiter.npos)
    {
      if (bracketStack.size())
      {
        continue;
      }

      temp = trim(input.substr(prev, curr - prev));
      if (!temp.empty())
      {
        tokens.push_back(temp);
      }
      prev = curr+1;
    }

    size_t openIndex = openBrackets.find(c);
    if (openIndex != openBrackets.npos)
    {
      bracketStack.push_back(closeBrackets[openIndex]);
    }
    else if (closeBrackets.find(c) != openBrackets.npos)
    {
      if (!bracketStack.empty() && bracketStack.back() == c)
      {
        bracketStack.pop_back();
      }
    }
  }

  temp = trim(input.substr(prev, curr - prev));
  if (!temp.empty())
  {
    tokens.push_back(temp);
  }

  return tokens;
}

#ifdef USE_METAL_COMPUTE
#ifndef DISABLE_PROFILING
#define ALWAYS_END_ENCODERS
#endif

#define USE_ARGUMENT_BUFFERS

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
static id<MTLCommandQueue> commandQueue = nil;
static id<MTLCommandBuffer> currentCommandBuffer          = nil;
static id<MTLBlitCommandEncoder> currentBlitEncoder       = nil;
static id<MTLComputeCommandEncoder> currentComputeEncoder = nil;
unordered_map<id<MTLComputePipelineState>, id<MTLFunction>> kernelNameMap;
unordered_map<string, uintPairList> kernelNameArgumentBufferMap;
static double  lastExecutionTime = 0.f;

// 128 bytes aligned
uint alignAllocSize(uint minimumSize)
{
  return ((minimumSize & 0x7F) > 0) ? ((minimumSize & (~0x7F)) + 0x80) : minimumSize;
}

static id<MTLBuffer> getTempBuffer(uint minimumSize)
{ @autoreleasepool {
  // find a suitable candidate if available
  uint smallerSizeDifference = -1;
  uint biggerSizeDifference = -1;
  int smallerBufferIndex = -1;
  int biggerBufferIndex = -1;

  const int maxAllowedDifference = 1024;

  id<MTLDevice> device = commandQueue.device;

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

  // if both found
  if (smallerSizeDifference != -1 && biggerSizeDifference != -1)
  {
    // expand smaller if its closer in size
    if (smallerSizeDifference < biggerSizeDifference)
    {
      tempBuffers[smallerBufferIndex] = pair<ushort, id<MTLBuffer>>(TEMP_BUFFER_OCCUPIED_FLAG, [device newBufferWithLength:minimumSize options:MTLResourceStorageModeShared]);
      return tempBuffers[smallerBufferIndex].second;
    }
    // if bigger is closer in size return it
    else
    {
      tempBuffers[biggerBufferIndex].first = TEMP_BUFFER_OCCUPIED_FLAG;
      return tempBuffers[biggerBufferIndex].second;
    }
  }
  else if (biggerSizeDifference != -1)
  {
    tempBuffers[biggerBufferIndex].first = TEMP_BUFFER_OCCUPIED_FLAG;
    return tempBuffers[biggerBufferIndex].second;
  }
  else if (smallerSizeDifference != -1)
  {
    tempBuffers[smallerBufferIndex] = pair<ushort, id<MTLBuffer>>(TEMP_BUFFER_OCCUPIED_FLAG, [device newBufferWithLength:minimumSize options:MTLResourceStorageModeShared]);
    return tempBuffers[smallerBufferIndex].second;
  }
  else
  {
    tempBuffers.push_back(pair<ushort, id<MTLBuffer>>(TEMP_BUFFER_OCCUPIED_FLAG, [device newBufferWithLength:minimumSize options:MTLResourceStorageModeShared]));
    return tempBuffers.back().second;
  }
  return nil;
}}

static void freeTempBuffer(id<MTLBuffer> buffer)
{ @autoreleasepool {
  for (int i=0; i<tempBuffers.size(); i++)
  {
    // if not occupied
    if (tempBuffers[i].second == buffer)
    {
      tempBuffers[i].first = 0;
      return;
    }
  }
}}

static id<MTLBlitCommandEncoder> getBlitEncoder()
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

static id<MTLComputeCommandEncoder> getComputeEncoder()
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

static void endEncoders()
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

inline string generateNewKernelHeader(const string& oldKernelHeader,
                                      const vector<pair<string, stringList>>& structData,
                                      const uintPairList& structPos,
                                      bool useDefines = true)
{
  string newKernelHeader;

  // declare structure before kernel declaration
  for (const auto &s : structData)
  {
    newKernelHeader.append("typedef struct "+s.first+"{\n");
    for (uint i=0; i<s.second.size(); i++)
    {
      newKernelHeader.append("  "+s.second[i]+" [[ id ("+to_string(i)+") ]];\n");
    }
    newKernelHeader.append("} "+s.first+";\n\n");
  }

  // copy kernel arguments before arg buffer
  newKernelHeader.append(oldKernelHeader, 0, structPos[0].first);

  // place argument buffer struct and members in kernel declaration
  for (uint i=0; i<structPos.size(); i++)
  {
    newKernelHeader.append("Const "+structData[i].first+" *"+structData[i].first+"_args,\n");
  }
  newKernelHeader.pop_back();
  newKernelHeader.pop_back();

  // copy kernel arguments after arg buffer
  newKernelHeader.append(oldKernelHeader, structPos.back().second, oldKernelHeader.size() - structPos.back().second);

  // create local variables using argument buffer members
  for (const auto &s : structData)
  {
    newKernelHeader.append("\n");
    for (const auto& m : s.second)
    {
      string memberName = m.substr(m.rfind(' ')+1, m.size() - m.find(' '));
      if (useDefines)
      {
        newKernelHeader.append("#define "+memberName+" "+s.first+"_args->"+memberName+" \n");
      }
      else
      {
        newKernelHeader.append(m+" = "+s.first+"_args->"+memberName+"; \n");
      }
    }
  }

  return newKernelHeader;
}

inline string generateKernelEnding(const vector<pair<string, stringList>>& structData,
                                   const uintPairList& structPos)
{
  string newKernelEnding;

  // create local variables using argument buffer members
  for (const auto &s : structData)
  {
    newKernelEnding.append("\n");
    for (const auto& m : s.second)
    {
      newKernelEnding.append("#undef "+m.substr(m.rfind(' ')+1, m.size() - m.find(' '))+"\n");
    }
  }

  return newKernelEnding;
}

inline string processArgumentBuffers(string source, bool useDefines = true)
{
  const string argBufferToken = "argumentStructMember";

  size_t kernelDecl = 0;
  do
  {
    // find the occurrence
    size_t pos = source.find(argBufferToken, kernelDecl);
    if (pos == source.npos)
    {
      break;
    }

    // find kernel declaration
    kernelDecl = source.rfind("Kernel ", pos);
    // find kernel definition
    size_t kernelDef  = source.find("{", pos);

    string kernelHeader = source.substr(kernelDecl, kernelDef - kernelDecl + 1);
    string newKernelHeader;
    vector<pair<string, stringList>> structData;
    uintPairList structPos;

    size_t argOffset = 0;
    do
    {
      // find argument buffer token
      size_t tokenPos = kernelHeader.find(argBufferToken, argOffset);
      if (tokenPos == kernelHeader.npos)
      {
        break;
      }

      // argument buffer member declaration begin and end
      size_t argBegin = kernelHeader.find('(', tokenPos + argBufferToken.size());
      size_t argEnd   = kernelHeader.find(')', argBegin+1);

      // tokenize declaration
      string argDecl  = kernelHeader.substr(argBegin + 1, argEnd - argBegin - 1);
      stringList tokens = tokenize(argDecl, ",");

      // add new structure if not already declared
      if (structData.empty() || structData.back().first != tokens[0])
      {
        structData.push_back(pair<string, stringList>(tokens[0], stringList()));
        structPos.push_back(uintPair(tokenPos, 0));
      }
      argOffset = argEnd+1;
      structPos.back().second = (uint)argOffset;
      structData.back().second.push_back(tokens[1]+" "+tokens[2]);
    }
    while (true);

    if (useDefines)
    {
      size_t end = findEndOfFunction(source, kernelDef + 1, '}');
      if (end != source.npos)
      {
        string ending = generateKernelEnding(structData, structPos);
        source.insert(end+1, ending);
      }
    }
    source.replace(kernelDecl, kernelDef - kernelDecl + 1, generateNewKernelHeader(kernelHeader, structData, structPos, useDefines));
  }
  while (true);

  return source;
}

inline string processAutoArgumentBuffers(string source)
{
  const string autoArgBufferToken = "#autoArgumentBuffer";

  // find the first occurrence
  size_t autoArgTokenPos = source.find(autoArgBufferToken);

  while (autoArgTokenPos != source.npos)
  {
    source.replace(autoArgTokenPos, autoArgBufferToken.size(), "");

    // find kernel declaration for the tag
    // go to the kernel declaration
    size_t kernelDecl   = source.find("Kernel ", autoArgTokenPos);
    // go till the beginning of kernel definitions
    size_t kernelDef    = source.find("{", autoArgTokenPos);
    string kernelHeader = source.substr(kernelDecl, kernelDef - kernelDecl + 1);

    // arguments declaration begin and end
    size_t argsBegin    = kernelHeader.find('(');
    size_t argsEnd      = kernelHeader.rfind(')');
    string kernelName   = tokenize(kernelHeader.substr(0, argsBegin)).back();

    stringList args = tokenize(kernelHeader.substr(argsBegin + 1, argsEnd - argsBegin - 1), ",");

    // find continuous constant arguments
    uint startIndex = -1;
    uintPairList argRanges;
    for (uint i=0; i<args.size(); i++)
    {
      const auto &arg = args[i];
      if (arg.find("const ") != arg.npos || arg.find("Const ") != arg.npos || arg.find("constantKernelInput") != arg.npos)
      {
        if (startIndex == -1)
        {
          startIndex = i;
        }
      }
      else
      {
        if (startIndex != -1)
        {
          argRanges.push_back(uintPair(startIndex, i-1));
          startIndex = -1;
        }
      }
    }

    if (startIndex != -1)
    {
      argRanges.push_back(uintPair(startIndex, (uint)args.size()-1));
    }

    // eleminate arguments with < 4 consecutive items
    for (uint i=0; i<argRanges.size(); i++)
    {
      if ((argRanges[i].second - argRanges[i].first) < 3)
      {
        argRanges.erase(argRanges.begin()+i);
        i--;
      }
    }

    kernelNameArgumentBufferMap[kernelName] = argRanges;

    vector<pair<string, stringList>> structData;
    uintPairList structPos;

    for (const auto& range : argRanges)
    {
      for (uint i=range.first; i<=range.second; i++)
      {
        auto &arg = args[i];
        size_t pos = -1;
        stringList parts;
        if (arg.find("constantKernelInput") != arg.npos)
        {
          pos = arg.find("constantKernelInput");
          arg.replace(pos, sizeof("constantKernelInput")-1, "");
          const size_t argsBegin = arg.find('(');
          parts = tokenize(arg.substr(argsBegin + 1, arg.size() - argsBegin - 1), ",");
        }
        else
        {
          pos = arg.find("const ");
          if (pos == arg.npos)
          {
            pos = arg.find("Const ");
          }
          const stringList tokens = tokenize(arg.substr(pos, arg.size() - pos));
          parts.push_back("");
          for (uint t=0; t<tokens.size()-1; t++)
          {
            parts.back().append(tokens[t]+" ");
          }
          parts.push_back(tokens.back());
          parts[1] += ")";
        }
        args[i] = arg.substr(0, pos)+"argumentStructMember("+kernelName+"_"+to_string(range.first)+"_"+to_string(range.second)+", const "+parts[0]+", "+parts[1];
      }
    }

    string newHeader;
    for (const auto& s : args)
    {
      newHeader += s+",";
    }
    newHeader.pop_back();

    size_t kernelEnd = findEndOfFunction(source, kernelDef + 1, '}');
    string kernelSource = source.substr(kernelDecl, kernelEnd - kernelDecl + 1);
    source.replace(kernelDecl, kernelEnd - kernelDecl + 1, processArgumentBuffers(kernelSource.replace(argsBegin + 1, argsEnd - argsBegin - 1, newHeader), true));

//    printf("%s", source.c_str());
    autoArgTokenPos = source.find(autoArgBufferToken, kernelDecl);
  }

  return source;
}

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

void logComputeMessage(const char* format, ...)
{
#ifndef DISABLE_LOGGING
  va_list args;
  va_start(args, format);
  printf("\nInfo: ");
  vprintf(format, args);
#endif
}

void logComputeError(const char* format, ...)
{
#ifndef DISABLE_LOGGING
  va_list args;
  va_start(args, format);
  printf("\nError: ");
  vprintf(format, args);
#endif
  assert(0);
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
#ifdef USE_OPENCL_COMPUTE
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
#else
  return KERNEL_RW;
#endif
}

size_t ComputeMemory::getOffset()const
{
  return offset;
}

size_t ComputeMemory::getSize()const
{
  return size;
}

ComputeTexture::ComputeTexture()
{
  ref = NULL;
  bytesPerPixel = 0;
  size[0] = 0;
  size[1] = 0;
}

ComputeTexture::ComputeTexture(ComputeTextureIdentifier ref, uint size[2], uint bytesPerPixel)
{
  this->ref = ref;
  this->bytesPerPixel = bytesPerPixel;
  this->size[0] = size[0];
  this->size[1] = size[1];
}

uint ComputeTexture::getBytesPerPixel()const
{
  return bytesPerPixel;
}

const uint* ComputeTexture::getSize()const
{
  return size;
}

ComputeHeap::ComputeHeap(ComputeInterface* compute, bool bypass)
  : bypass(bypass), heap(NULL), compute(compute)
{
  childs.clear();
}

ComputeHeap::~ComputeHeap()
{
  while(childs.size())
  {
    free(*childs.begin());
  }
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
  else
  {
    heap = alloc(sizeInBytes);
    childs.pop_back();
  }
}

ComputeMemory* ComputeHeap::alloc(size_t sizeInBytes, void* data, ComputeMemoryFlag flag)
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
#if TARGET_OS_IPHONE
    ret = new ComputeMemory([compute->context newBufferWithLength:sizeInBytes options:MTLResourceStorageModeShared], 0, sizeInBytes);
#else
    ret = new ComputeMemory([compute->context newBufferWithLength:sizeInBytes options:MTLResourceStorageModePrivate], 0, sizeInBytes);
#endif
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
    ret = new ComputeMemory(*heap, offset, sizeInBytes);
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


ComputeKernel::ComputeKernel()
{
  ref = NULL;
  setArgumentBuffer = false;
}

ComputeKernel::ComputeKernel(ComputeKernelIdentifier ref)
{
  this->ref = ref;
  setArgumentBuffer = false;
}

uint ComputeKernel::mapArgumentIndex(uint index)const
{
  uint ret = index;

  for (const auto& range : argumentBufferRange)
  {
    if (index >= range.second)
    {
      ret -= range.second - range.first;
      continue;
    }
    break;
  }

  return ret;
}

void ComputeKernel::addArgumentBufferRange(uint startIndex, uint inclusiveEndIndex)
{
#ifdef USE_METAL_COMPUTE
  argumentBufferRange.push_back(uintPair(startIndex, inclusiveEndIndex));
  setArgumentBuffer = true;
#else
#endif
}

void ComputeKernel::setArg(void* valuePtr, size_t valueSize, uint argIndex)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clSetKernelArg(ref, argIndex, valueSize, valuePtr);
  computeCheckError(status, 0);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [getComputeEncoder() setBytes:valuePtr length:valueSize atIndex:argIndex];
  }
  else
  {
    ArgData data;
    data.type = 1;
    data.ptr = valuePtr;
    data.size = (uint)valueSize;
    data.index = argIndex;
    args.push_back(data);
  }
#endif
}

void ComputeKernel::setArg(const void* valuePtr, size_t valueSize, uint argIndex)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clSetKernelArg(ref, argIndex, valueSize, valuePtr);
  computeCheckError(status, 0);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [getComputeEncoder() setBytes:valuePtr length:valueSize atIndex:argIndex];
  }
  else
  {
    ArgData data;
    data.type = 4;
    data.cptr = valuePtr;
    data.size = (uint)valueSize;
    data.index = argIndex;
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
    [getComputeEncoder() setBuffer:ident offset:buffer->getOffset() atIndex:index];
  }
  else
  {
    ArgData data;
    data.type = 2;
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
    [getComputeEncoder() setBuffer:ident offset:buffer->getOffset() atIndex:index];
  }
  else
  {
    ArgData data;
    data.type = 3;
    data.ptr = buffer;
    data.index = index;
    args.push_back(data);
  }
#endif
}

void ComputeKernel::setArgs(ComputeMemory* buffers[], const uint count, uint* indices)
{
  for (uint i = 0; i < count; i++)
  {
    ComputeMemoryIdentifier ident = *buffers[i];
#ifdef USE_OPENCL_COMPUTE
    setArg<ComputeMemoryIdentifier>(&ident, indices ? indices[i] : i);
#else
    setArg(buffers[i], (indices ? indices[i] : i));
#endif
  }
}

void ComputeKernel::setSharedMemArg(const size_t valueSize, uint index)
{
#ifdef USE_OPENCL_COMPUTE
  ComputeStatus status = clSetKernelArg(ref, index, valueSize, NULL);
  computeCheckError(status, 0);
#else
  if (!setArgumentBuffer) @autoreleasepool
  {
    [getComputeEncoder() setThreadgroupMemoryLength:valueSize atIndex:index];
  }
  else
  {
    ArgData data;
    data.type = 4;
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
          case 1:
            memcpy([argumentEncoder constantDataAtIndex:index], arg.ptr, arg.size);
            break;
          case 2:
            [argumentEncoder setBuffer:*((const ComputeMemory*)arg.cptr) offset:((const ComputeMemory*)arg.cptr)->getOffset() atIndex:index];
            break;
          case 3:
            [argumentEncoder setBuffer:*((ComputeMemory*)arg.ptr) offset:((ComputeMemory*)arg.ptr)->getOffset() atIndex:index];
            break;
          case 4:
            memcpy([argumentEncoder constantDataAtIndex:index], arg.cptr, arg.size);
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
    [getComputeEncoder() setBuffer:argumentBuffers[mapArgumentIndex(range.first)] offset:0 atIndex:range.first];
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
      case 1:
        [getComputeEncoder() setBytes:i.ptr length:i.size atIndex:i.index];
        break;
      case 2:
        [getComputeEncoder() setBuffer:*((const ComputeMemory*)i.cptr) offset:((const ComputeMemory*)i.cptr)->getOffset() atIndex:i.index];
        break;
      case 3:
        [getComputeEncoder() setBuffer:*((ComputeMemory*)i.ptr) offset:((ComputeMemory*)i.ptr)->getOffset() atIndex:i.index];
        break;
      case 4:
        [getComputeEncoder() setThreadgroupMemoryLength:i.size atIndex:i.index];
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
  [getComputeEncoder() useResource:*resource usage:MTLResourceUsageRead];
#endif
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

  ComputeKernel ret = ComputeKernel([ref.device newComputePipelineStateWithDescriptor:pipelineDesc options:0 reflection:0 error:&error]);
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

void ComputeInterface::create(int deviceIndex)
{
  const uint MAX_STRING_LENGTH = 128;

  string selectedDevice;
  simdGroupSize = 16;

  ComputeDeviceId defaultDeviceId;
  ComputeDeviceId selectedDeviceId;
  int absoluteDeviceIndex = 0;

#ifdef USE_OPENCL_COMPUTE
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

    platform = platforms[i];

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 8, devices, &deviceCount);
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

    logComputeMessage("  Device info:");

    // print device info and select one if not supplied
    for (uint j = 0; j < deviceCount; j++)
    {
      ComputeDeviceId deviceId = devices[j];
#ifdef USE_OPENCL_COMPUTE
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

ComputeProgram ComputeInterface::createProgram(const char* sourceCode, size_t sourceSize)
{
  ComputeStatus status;
#ifdef USE_OPENCL_COMPUTE
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
#else
  ComputeProgram program;
  char* log = new char[1];
  log[0] = NULL;
  @autoreleasepool {
  NSError *error = nil;
  NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n";
  if ([deviceId supportsFamily:MTLGPUFamilyApple6])
  {
    source = [source stringByAppendingString:@"#define USE_SIMD_COMPUTE"];
  }
#ifdef USE_ARGUMENT_BUFFERS
  source = [source stringByAppendingString:[NSString stringWithUTF8String:processAutoArgumentBuffers(sourceCode).c_str()]];
#else
  source = [source stringByAppendingString:[NSString stringWithUTF8String:sourceCode]];
#endif
  ComputeProgramIdentifier programId = [deviceId newLibraryWithSource:source options:0 error:&error];
  program = ComputeProgram(programId);
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

ComputeProgram ComputeInterface::createTemplateProgram(const char* fileName, const vector<string>* oldType,
  const vector<string>* newType, const vector<string>* includeFiles)
{
  std::string data = "\n";
  if (oldType)
  {
    for (uint i = 0; i < oldType->size(); i++)
    {
      data += "#define " + (*oldType)[i] + " " + (*newType)[i] + "\n";
    }
  }
  logComputeMessage("Template types%s", data.c_str());
  if (includeFiles)
  {
    for (uint i = 0; i < includeFiles->size(); i++)
    {
      data += IOInterface::readFile((*includeFiles)[i].c_str()) + "\n";
    }
  }
  data += IOInterface::readFile(fileName);
  data += "\n";

  logComputeMessage("Compiling File: %s", fileName);
  return createProgram(data.c_str(), data.size());
}

ComputeProgram ComputeInterface::createTemplateProgram(const string& sourceCode, const vector<string>* oldType,
  const vector<string>* newType, const vector<string>* includeFiles)
{
  std::string data = "\n";
  if (oldType)
  {
    for (uint i = 0; i < oldType->size(); i++)
    {
      data += "#define " + (*oldType)[i] + " " + (*newType)[i] + "\n";
    }
  }
  logComputeMessage("Template types%s", data.c_str());
  if (includeFiles)
  {
    for (uint i = 0; i < includeFiles->size(); i++)
    {
      data += IOInterface::readFile((*includeFiles)[i].c_str()) + "\n";
    }
  }
  data += sourceCode;
  data += "\n";

  return createProgram(data.c_str(), data.size());
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

void ComputeInterface::setBuffer(const ComputeMemory* source, size_t sourceOffset, size_t sizeInBytes, const void* hostValue, size_t hostValueSize)
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
  id<MTLBuffer> tempBuffer = getTempBuffer((uint)sizeInBytes);
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
    freeTempBuffer(tempBuffer);
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
  id<MTLBuffer> tempBuffer = getTempBuffer((uint)sizeInBytes);
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
    freeTempBuffer(tempBuffer);
  }}
#else
  memcpy((char*)(id<MTLBuffer>(*destination)).contents + (destinationOffset + destination->getOffset()), hostPtr, sizeInBytes);
#endif
#endif
}

void ComputeInterface::configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount)
{
  configureSize(workgroupSize, workgroupCount, threadCount, maxThreadsPerGroup());
}

void ComputeInterface::configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount, const uint maxThreadsPerThreadgroup)
{
  const uint maxThreads = maxThreadsPerThreadgroup;
  const uint width = maxThreadsPerThreadgroup > simdSize() ? simdSize() : maxThreadsPerThreadgroup;
  const uint height = maxThreads / width;

  workgroupSize[0] = width;
  workgroupSize[1] = mAlignBy(threadCount, width);
  workgroupSize[2] = 1;

  workgroupCount[0] = 1;
  if (workgroupSize[1] > height)
  {
    workgroupCount[0] = mAlignBy(workgroupSize[1], height);
    workgroupSize[1] = height;
  }
  workgroupSize[0] *= workgroupSize[1];
  workgroupSize[1] = 1;
  workgroupCount[1] = 1;
  workgroupCount[2] = 1;
}

void ComputeInterface::configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount[3])
{
  configureSize(workgroupSize, workgroupCount, threadCount, maxThreadsPerGroup());
}

void ComputeInterface::configureSize(size_t workgroupSize[3], size_t workgroupCount[3], const uint threadCount[3], const uint maxThreadsPerThreadgroup)
{
  const uint width = maxThreadsPerThreadgroup > simdSize() ? simdSize() : maxThreadsPerThreadgroup;
  const uint height = maxThreadsPerThreadgroup / width;

  workgroupSize[0] = width;
  workgroupSize[1] = height;
  workgroupSize[2] = 1;

  workgroupCount[0] = mAlignBy(threadCount[0], workgroupSize[0]);
  workgroupCount[1] = mAlignBy(threadCount[1], workgroupSize[1]);
  workgroupCount[2] = 1;
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
  copyToHost(indirectBuffer, bufferOffset, 4, &count, true);

  size_t workgroupCount[3] = {count, 1, 1};

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
      ::lastExecutionTime = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.f;
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

uint ComputeInterface::maxThreadsPerGroup()const
{
  return (uint)maxThreadsPerWorkgroup;
}

uint ComputeInterface::simdSize()const
{
  return (uint)simdGroupSize;
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

ComputeDeviceId ComputeInterface::getDevice()
{
  return deviceId;
}

double ComputeInterface::lastExecutionTime()const
{
  return ::lastExecutionTime;
}

#ifdef ENABLE_RENDERING
ComputeMemory ComputeInterface::createMemoryFromGLBuffer(GLuint glObject)
{
  ComputeStatus status;
  ComputeMemory ret(clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, glObject, &status));
  computeCheckError(status, 0);
  return ret;
}

ComputeMemory ComputeInterface::createMemoryFromGLTexture(GLuint glObject)
{
  ComputeStatus status;
  ComputeMemory ret(clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, glObject, &status));
  computeCheckError(status, 0);
  return ret;
}
#endif
