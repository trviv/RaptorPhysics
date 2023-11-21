#include "ComputeInterface.h"
#include <unordered_set>

static unordered_map<string, ComputeProgram> cachedPrograms;

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
  printf("\n");
#endif
  assert(0);
}


ComputeMemory::ComputeMemory()
{
  ref = NULL;
  buf = NULL;
  offset = 0;
  size = 0;
}

ComputeMemory::ComputeMemory(const ComputeMemory& ref)
{
  this->ref = ref.ref;
  this->buf = ref.buf;
  this->offset = ref.offset;
  this->size = ref.size;
}

ComputeMemory::ComputeMemory(ComputeMemoryIdentifier ref, size_t offset, size_t size)
{
  this->ref = ref;
  this->buf = NULL;
  this->offset = offset;
  this->size = size;
}

ComputeMemory::ComputeMemory(ComputeMemory* memory, size_t offset, size_t size)
{
  this->ref = memory->ref;
  this->buf = memory->buf;
  this->offset = offset;
  this->size = size;
}

ComputeMemory& ComputeMemory::operator = (const ComputeMemory& ref)
{
  this->ref = ref.ref;
  this->buf = ref.buf;
  this->offset = ref.offset;
  this->size = ref.size;

  return *this;
}

ComputeMemoryUsage ComputeMemory::getUsage()const
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


ComputeHeap::ComputeHeap(ComputeInterface* compute, bool bypass)
  : bypass(bypass), heap(NULL), compute(compute), childs({})
{}

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

void ComputeKernel::setArgs(ComputeMemory* buffers[], const uint count, uint* indices)
{
  for (uint i = 0; i < count; i++)
  {
    ComputeMemoryIdentifier ident = *buffers[i];
    setArg(buffers[i], (indices ? indices[i] : i));
  }
}


ComputeProgram ComputeInterface::createTemplateProgram(const string& sourceCode, const stringArr* oldType,
  const stringArr* newType, const stringArr* includeFiles)
{
  string programSignature;

  if ((oldType == NULL) ^ (newType == NULL))
  {
    logComputeError("Old and New type inconsistent for the program!");
  }
  if (oldType && oldType->size() != newType->size())
  {
    logComputeError("Different lengths for Old and New types for the program!");
  }

  programSignature += to_string(hash<string>{}(sourceCode))+"_";
  programSignature += join(oldType, "_");
  programSignature += join(newType, "_");
  programSignature += join(unique(includeFiles), "_");

  if (cachedPrograms.count(programSignature))
  {
    return cachedPrograms[programSignature];
  }

  ComputeProgram program = createProgram(sourceCode.c_str(), sourceCode.size(), oldType, newType, includeFiles);
  cachedPrograms[programSignature] = program;

  return program;
}

ComputeProgram ComputeInterface::createTemplateProgram(const char* fileName, const stringArr* oldType,
  const stringArr* newType, const stringArr* includeFiles)
{
  logComputeMessage("Compiling File: %s", fileName);
//  string command = "";
//
//  command +=
//  "val="+string(fileName)+"\n"
//  "SCRIPT_FILE_NAME=$val\n"
//  "IFS='.' read -ra fname <<< `$val`\n"
//  "INCLUDE_INPUT_PATH_0=/Users/vivek/Projects/ParticlePhysics/build/build_macos/Tests/Debug/Tests.app/Contents/Resources/\n"
//  "SCRIPT_INPUT_FILE_0=$INCLUDE_INPUT_PATH_0/$SCRIPT_FILE_NAME\n"
//  "SCRIPT_OUTPUT_FILE_0=$INCLUDE_INPUT_PATH_0/${fname[0]}.air\n"
//  "SCRIPT_OUTPUT_FILE_1=$INCLUDE_INPUT_PATH_0/${fname[0]}.metallib\n"
//  "xcrun -sdk iphoneos metal -x metal -dynamiclib -fvisibility=hidden $SCRIPT_INPUT_FILE_0 -o $SCRIPT_OUTPUT_FILE_0 -I $INCLUDE_INPUT_PATH_0 -D COMPUTE_SHADER_SCOPE "
////  "xcrun -sdk iphoneos metal -x metal $SCRIPT_INPUT_FILE_0 -o $SCRIPT_OUTPUT_FILE_0 -I $INCLUDE_INPUT_PATH_0 -D COMPUTE_SHADER_SCOPE\n"
////  "xcrun -sdk iphoneos metallib $SCRIPT_OUTPUT_FILE_0 -o $SCRIPT_OUTPUT_FILE_1\n"
//  ;
//
//  for (int i=0; oldType && i<oldType->size(); i++)
//  {
//    command += " -D '"+oldType->at(i)+"="+newType->at(i)+"'";
//  }
//
//  system(command.c_str());
  return createTemplateProgram(IOInterface::readFile(fileName), oldType, newType, includeFiles);
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
