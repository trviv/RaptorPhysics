#include "Core.h"
#include "../ParticlePhysics/Common/ParticleStruct.h"
#include <algorithm>

class PerfShaderEntity : public ShaderEntity
{
public:
  PerfShaderEntity(ComputeInterface* compute, const char* source, const vector<string>* oldType, const vector<string>* newType, const vector<string>* includeFiles = NULL)
  {
    ShaderEntity::registerShaderFromSource(compute, source, oldType, newType, includeFiles);
  }

  ComputeProgram& getProgram()
  {
    return programs.back();
  }
};

static ComputeInterface* compute;

static int warmIterations     = 2;
static int baselineIterations = 20;
static int shaderIterations   = 20;

const int roughElements = 1024 * 1024 * 4;

void printStats(float mean, uint elements, uint rwCount, uint sizeOfElements)
{
  logComputeMessage("Average time    : %f ms", mean);
  logComputeMessage("Elts/sec        : %f M", elements * 1000.f / (mean * 1000 * 1000));
  logComputeMessage("Bandwidth util  : %f GB/s", (rwCount * sizeOfElements * elements) * (1000.f / mean) / float(1024 * 1024 * 1024));
}

template<class DataType> vector<string> profileSharedRW(string dataType, int dataTypeSize, ComputeInterface* compute)
{
  vector<string> ret;

  logComputeMessage("Profiling Shared Read/Write:");

  DeviceArray<DataType> indata(compute, NULL);
  DeviceArray<DataType> outdata(compute, NULL);

  const int elements = roughElements;

  string sourceCode =
    "#include \"ComputeHeader.h\"\n"
    "#include \"ComputeShared.h\"\n"
    "Kernel void sharedKernel( "
    "  Device dataType*               outputArray, "
    "  const Device dataType*         inputArray, "
    "  constantKernelInput(int,       iterations), "
    "  constantKernelInput(short,     mask), "
    "  constantKernelInput(short,     shift), "
    "  constantKernelInput(short,     offset), "
    "  sharedMemKernelInput(dataType, localArray1D, 6) "
    "  KERNEL_GLOBAL_ARGUMENTS "
    "  KERNEL_THREAD_ARGUMENTS "
    "  KERNEL_THREADGROUP_ARGUMENTS) "
    "{ "
    "    const uint index = threadIndex(); "
    "    const uint localIndex = threadLocalIndex(); "
    "    const ushort subGroupLocalIndex = localIndex & (ComputeSimdWidth - 1); "
    "    const ushort subGroupIndex = localIndex >> ComputeSimdWidthExp; "
    "    localArray1D[localIndex] = inputArray[index]; "
    "    localMemBarrier(); "
    "    const uint finalLocalIndex = ((localIndex & mask) << shift) + subGroupIndex * offset; "
    "    for (int i=0; i<iterations; i++) "
    "    { "
    "\n#ifndef BASELINE_RUN \n"
    "        localArray1D[finalLocalIndex] = localArray1D[finalLocalIndex] + i; "
    "\n#endif \n"
    "        localMemBarrier(); "
    "    } "
    "    outputArray[index] = localArray1D[localIndex]; "
    "} "
  ;

  indata.host()->reserve(elements * dataTypeSize);
  for (int i = 0; i < (elements * dataTypeSize); i++)
  {
    indata.host()->push_back(rand()&0x3);
  }

  indata.syncDevice();
  outdata.resize(indata.size(), true);

  size_t workgroupSize[3], workgroupCount[3];

  compute->configureSize(workgroupSize, workgroupCount, elements);

  vector<string> oldType = { "dataType", "ComputeSimdWidth", "ComputeSimdWidthExp", "BASELINE_RUN"};
  vector<string> newType = { dataType, to_string(compute->simdSize()), to_string(compute->simdSize()), "" };

  PerfShaderEntity baselineProgram(compute, sourceCode.c_str(), &oldType, &newType);
  ComputeKernel baselineKernel = baselineProgram.getProgram().createKernel("sharedKernel");

  ComputeMemory* buffers[] = {
    outdata.device(),
    indata.device()
  };

  size_t localMemorySize = workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * dataTypeSize * sizeof(DataType);

  {
    int iterations = 0;
    short mask = workgroupSize[0] * workgroupSize[1] * workgroupSize[2] - 1;
    short shift = 0;
    short offset  = 0;

    //--------------------------------------------------------------------------------
    // warm up run
    //--------------------------------------------------------------------------------
    for (uint i = 0; i < warmIterations; i++)
    {
      baselineKernel.setArgs(buffers,         2);
      baselineKernel.setArg<int>(&iterations, 2);
      baselineKernel.setArg<short>(&mask,     3);
      baselineKernel.setArg<short>(&shift,    4);
      baselineKernel.setArg<short>(&offset,   5);
      baselineKernel.setSharedMemArg(localMemorySize, 6);

      compute->execute(baselineKernel, workgroupSize, workgroupCount);
    }
    compute->sync(false);
  }

  oldType.pop_back();
  newType.pop_back();

  PerfShaderEntity mainProgram(compute, sourceCode.c_str(), &oldType, &newType);
  ComputeKernel mainKernel = mainProgram.getProgram().createKernel("sharedKernel");

  const int minIterationExp = 1;
  const int maxIterationExp = 4;

  const short minOffset = 0;
  const short maxOffset = 1;

  const short minShiftExp = 0;
  const short maxShiftExp = 2;

  const short minMaskExp = 2;
  const short maxMaskExp = 8;

//  const int minIterationExp = 1;
//  const int maxIterationExp = 2;
//
//  const short minOffset = 0;
//  const short maxOffset = 0;
//
//  const short minShiftExp = 0;
//  const short maxShiftExp = 0;
//
//  const short minMaskExp = 1;
//  const short maxMaskExp = 9;

  //--------------------------------------------------------------------------------
  // profiling run
  //--------------------------------------------------------------------------------
  for (short iterationExp = minIterationExp; iterationExp <= maxIterationExp; iterationExp++)
  {
//    int iterations = (16 << iterationExp);
    int iterations = 10 * iterationExp;

    {
      short mask = workgroupSize[0] * workgroupSize[1] * workgroupSize[2] - 1;
      short shift = 0;
      short offset  = 0;

      //--------------------------------------------------------------------------------
      // baseline run
      //--------------------------------------------------------------------------------
      for (uint i = 0; i < baselineIterations; i++)
      {
        baselineKernel.setArgs(buffers,         2);
        baselineKernel.setArg<int>(&iterations, 2);
        baselineKernel.setArg<short>(&mask,     3);
        baselineKernel.setArg<short>(&shift,    4);
        baselineKernel.setArg<short>(&offset,   5);
        baselineKernel.setSharedMemArg(localMemorySize, 6);

        compute->execute(baselineKernel, workgroupSize, workgroupCount);
      }

      compute->sync();
    }

    float baseline = compute->lastExecutionTime() / baselineIterations;

    for (short shiftExp = minShiftExp; shiftExp <= maxShiftExp; shiftExp++)
    {
      for (short offset = minOffset; offset <= maxOffset; offset++)
      {
        for (short maskExp = minMaskExp; maskExp <= maxMaskExp; maskExp++)
        {
          short shift = (1 << shiftExp) - 1;
          short mask = (1 << maskExp) - 1;

          for (short si = 0; si < shaderIterations; si++)
          {
            mainKernel.setArgs(buffers,         2);
            mainKernel.setArg<int>(&iterations, 2);
            mainKernel.setArg<short>(&mask,     3);
            mainKernel.setArg<short>(&shift,    4);
            mainKernel.setArg<short>(&offset,   5);
            mainKernel.setSharedMemArg(localMemorySize, 6);

            compute->execute(mainKernel, workgroupSize, workgroupCount);
          }
          compute->sync();

          float currentTime = compute->lastExecutionTime() / shaderIterations;
          char line[1024] = {NULL};

          sprintf(line, "%d, %d, %02hd, %02hd, %04hd, %f", dataTypeSize, iterations, mask, shift, offset, (currentTime - baseline) / iterations);
          printf("%s\n", line);
          ret.push_back(line);
        }
      }
    }
  }

  return ret;
}

template<class DataType> vector<string> profileVectorDevRead(string dataType, int dataTypeSize, ComputeInterface* compute)
{
  vector<string> ret;

  logComputeMessage("Profiling Vectorized Read:");

  DeviceArray<DataType> indata(compute, NULL);
  DeviceArray<DataType> outdata(compute, NULL);

  const int elements = roughElements;

  string sourceCode =
    "#include \"ComputeHeader.h\"\n"
    "#include \"ComputeShared.h\"\n"
    "Kernel void sharedKernel( "
    "  Device dataType*               outputArray, "
    "  const Device dataType*         inputArray, "
    "  constantKernelInput(int,       iterations), "
    "  constantKernelInput(short,     mask), "
    "  constantKernelInput(short,     shift), "
    "  constantKernelInput(short,     offset) "
    "  KERNEL_GLOBAL_ARGUMENTS "
    "  KERNEL_THREAD_ARGUMENTS "
    "  KERNEL_THREADGROUP_ARGUMENTS) "
    "{ "
    "    const uint index = threadIndex(); "
    "    const uint localIndex = threadLocalIndex(); "
    "    const ushort subGroupLocalIndex = localIndex & (ComputeSimdWidth - 1); "
    "    const ushort subGroupIndex = localIndex >> ComputeSimdWidthExp; "
    "    localArray1D[localIndex] = inputArray[index]; "
    "    localMemBarrier(); "
    "    const uint finalLocalIndex = ((localIndex & mask) << shift) + subGroupIndex * offset; "
    "    for (int i=0; i<iterations; i++) "
    "    { "
    "\n#ifndef BASELINE_RUN \n"
    "        localArray1D[finalLocalIndex] = localArray1D[finalLocalIndex] + i; "
    "\n#endif \n"
    "        localMemBarrier(); "
    "    } "
    "    outputArray[index] = localArray1D[localIndex]; "
    "} "
  ;

  indata.host()->reserve(elements * dataTypeSize);
  for (int i = 0; i < (elements * dataTypeSize); i++)
  {
    indata.host()->push_back(rand()&0x3);
  }

  indata.syncDevice();
  outdata.resize(indata.size(), true);

  size_t workgroupSize[3], workgroupCount[3];

  compute->configureSize(workgroupSize, workgroupCount, elements);

  vector<string> oldType = { "dataType", "ComputeSimdWidth", "ComputeSimdWidthExp", "BASELINE_RUN"};
  vector<string> newType = { dataType, to_string(compute->simdSize()), to_string(compute->simdSize()), "" };

  PerfShaderEntity baselineProgram(compute, sourceCode.c_str(), &oldType, &newType);
  ComputeKernel baselineKernel = baselineProgram.getProgram().createKernel("sharedKernel");

  ComputeMemory* buffers[] = {
    outdata.device(),
    indata.device()
  };

  size_t localMemorySize = workgroupSize[0] * workgroupSize[1] * workgroupSize[2] * dataTypeSize * sizeof(DataType);

  {
    int iterations = 0;
    short mask = workgroupSize[0] * workgroupSize[1] * workgroupSize[2] - 1;
    short shift = 0;
    short offset  = 0;

    //--------------------------------------------------------------------------------
    // warm up run
    //--------------------------------------------------------------------------------
    for (uint i = 0; i < warmIterations; i++)
    {
      baselineKernel.setArgs(buffers,         2);
      baselineKernel.setArg<int>(&iterations, 2);
      baselineKernel.setArg<short>(&mask,     3);
      baselineKernel.setArg<short>(&shift,    4);
      baselineKernel.setArg<short>(&offset,   5);
      baselineKernel.setSharedMemArg(localMemorySize, 6);

      compute->execute(baselineKernel, workgroupSize, workgroupCount);
    }
    compute->sync(false);
  }

  oldType.pop_back();
  newType.pop_back();

  PerfShaderEntity mainProgram(compute, sourceCode.c_str(), &oldType, &newType);
  ComputeKernel mainKernel = mainProgram.getProgram().createKernel("sharedKernel");

  const int minIterationExp = 1;
  const int maxIterationExp = 4;

  const short minOffset = 0;
  const short maxOffset = 1;

  const short minShiftExp = 0;
  const short maxShiftExp = 2;

  const short minMaskExp = 2;
  const short maxMaskExp = 8;

//  const int minIterationExp = 1;
//  const int maxIterationExp = 2;
//
//  const short minOffset = 0;
//  const short maxOffset = 0;
//
//  const short minShiftExp = 0;
//  const short maxShiftExp = 0;
//
//  const short minMaskExp = 1;
//  const short maxMaskExp = 9;

  //--------------------------------------------------------------------------------
  // profiling run
  //--------------------------------------------------------------------------------
  for (short iterationExp = minIterationExp; iterationExp <= maxIterationExp; iterationExp++)
  {
//    int iterations = (16 << iterationExp);
    int iterations = 10 * iterationExp;

    {
      short mask = workgroupSize[0] * workgroupSize[1] * workgroupSize[2] - 1;
      short shift = 0;
      short offset  = 0;

      //--------------------------------------------------------------------------------
      // baseline run
      //--------------------------------------------------------------------------------
      for (uint i = 0; i < baselineIterations; i++)
      {
        baselineKernel.setArgs(buffers,         2);
        baselineKernel.setArg<int>(&iterations, 2);
        baselineKernel.setArg<short>(&mask,     3);
        baselineKernel.setArg<short>(&shift,    4);
        baselineKernel.setArg<short>(&offset,   5);
        baselineKernel.setSharedMemArg(localMemorySize, 6);

        compute->execute(baselineKernel, workgroupSize, workgroupCount);
      }

      compute->sync();
    }

    float baseline = compute->lastExecutionTime() / baselineIterations;

    for (short shiftExp = minShiftExp; shiftExp <= maxShiftExp; shiftExp++)
    {
      for (short offset = minOffset; offset <= maxOffset; offset++)
      {
        for (short maskExp = minMaskExp; maskExp <= maxMaskExp; maskExp++)
        {
          short shift = (1 << shiftExp) - 1;
          short mask = (1 << maskExp) - 1;

          for (short si = 0; si < shaderIterations; si++)
          {
            mainKernel.setArgs(buffers,         2);
            mainKernel.setArg<int>(&iterations, 2);
            mainKernel.setArg<short>(&mask,     3);
            mainKernel.setArg<short>(&shift,    4);
            mainKernel.setArg<short>(&offset,   5);
            mainKernel.setSharedMemArg(localMemorySize, 6);

            compute->execute(mainKernel, workgroupSize, workgroupCount);
          }
          compute->sync();

          float currentTime = compute->lastExecutionTime() / shaderIterations;
          char line[1024] = {NULL};

          sprintf(line, "%d, %d, %02hd, %02hd, %04hd, %f", dataTypeSize, iterations, mask, shift, offset, (currentTime - baseline) / iterations);
          printf("%s\n", line);
          ret.push_back(line);
        }
      }
    }
  }

  return ret;
}

int main(int argc, char** argv)
{
  compute = new ComputeInterface();
  compute->create();

  for (int dataTypeSize = 1; dataTypeSize <= 3; dataTypeSize++)
  {
    vector<string> stat = profileSharedRW<float>(("float"+string(dataTypeSize > 1 ? to_string(dataTypeSize) : "")).c_str(), dataTypeSize, compute);
//    vector<string> stat2 = profileVectorDevRead<float>(("float"+string(dataTypeSize > 1 ? to_string(dataTypeSize) : "")).c_str(), dataTypeSize, compute);
  }

  delete compute;
  return 0;
}
