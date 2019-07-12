#include "ComputeUtils.h"
#include "ComputeUtilsShared.h"

vector<ComputeUtil> computeUtils;
vector<string>      computeConfig;

#define COMPUTE_UTIL_SUM_1D_KERNEL                    0
#define COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL            1
#define COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL          2
#define COMPUTE_UTIL_SHOW_MATRIX_KERNEL               3
#define COMPUTE_UTIL_SECTION_OFFSET                   4
#define COMPUTE_UTIL_COMPACT_SPARSE_ARRAY             5
#define COMPUTE_UTIL_CONSOLIDATE_FROM_PARTITIONS      6
#define COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL    7
#define COMPUTE_UTIL_RADIX_SORT1                      8
#define COMPUTE_UTIL_RADIX_SORT2                      9
#define COMPUTE_UTIL_BITONIC_SORT                     10

enum UtilTemporaryBuffer
{
  UtilTempPrefixGroupSum,
  UtilTempPrefixGroupStatus,
  UtilTempRadixGroupSum,
  UtilTempRadixPrefixGroupSum,
  UtilTempReduceSum,
  UtilTempReduceStatus,
  UtilTempBufferMax
};

#define RADIX_SORT_BIT_COUNT          4
#define RADIX_REDUCTION_PACKING_EXP   2
#define RADIX_PREFIX_SCAN_PACKING_EXP 2

string getKeyName(ComputeUtilKey key)
{
  switch (key)
  {
  case ComputeUtilStructType:
    return "StructType";
  case ComputeUtilStructSize:
    return "StructSize";
  case ComputeUtilStructMemberType:
    return "MemberStructType";
  case ComputeUtilStructMemberSize:
    return "StructMemberSize";
  case ComputeUtilStructMember:
    return "StructMember";
  case ComputeUtilStructTypeIntegral:
    return "StructTypeIntegral";

  case ComputeUtilIdentityStructType:
    return "IdentityStructType";
  case ComputeUtilIdentityStructMember:
    return "IdentityStructMember";
  case ComputeUtilIdentityFunction:
    return "IdentityFunction";

  case ComputeUtilCustomAddFunction:
    return "AddFunction";
  case ComputeUtilCustomCopyFunction:
    return "CopyFunction";
  case ComputeUtilCustomDivFunction:
    return "DivFunction";
  case ComputeUtilCustomClearFunction:
    return "ClearFunction";

  case ComputeUtilBatchSize:
    return "BatchSize";

  case ComputeUtilSkipParallelPrimitives:
    return "SkipParallelPrimitives";

  case ComputeUtilOnlyReduce:
    return "OnlyReduce";

  case ComputeUtilMaxWorkgroupSize:
    return "MaxWorkgroupSize";

  default:
    assert("Enumeration not defined!" && 0);
  }
  return "";
}

uint ComputeUtil::create(ComputeInterface* compute, map<ComputeUtilKey, string>& dataMap, const vector<string>* includeFiles)
{
  ComputeUtil util;
  vector<string> oldType;
  vector<string> newType;
  vector<string> kernelNames;

  util.includeFiles.push_back("ComputeHeader.shader");
  util.includeFiles.push_back("ComputeShared.h");

  if (includeFiles)
  {
    for (auto i : *includeFiles)
    {
      util.includeFiles.push_back(i);
    }
  }

  util.includeFiles.push_back("ComputeUtilsShared.h");
  util.includeFiles.push_back("PrefixScan.shader");
  util.includeFiles.push_back("Reduce.shader");
  util.includeFiles.push_back("ComplexReduce.shader");
  util.includeFiles.push_back("RadixSort.shader");

  string key;

  for (uint i = 0; i < ComputeUtilMaxKey; i++)
  {
    if (dataMap.find((ComputeUtilKey)i) != dataMap.end())
    {
      key += getKeyName((ComputeUtilKey)i) + ":" + dataMap[(ComputeUtilKey)i];
      oldType.push_back(getKeyName((ComputeUtilKey)i));
      newType.push_back(dataMap[(ComputeUtilKey)i]);
    }
  }

  for (const string& include : util.includeFiles)
  {
    key += ":" + include;
  }

  vector<string>::iterator pos = find(computeConfig.begin(), computeConfig.end(), key);
  if (pos != computeConfig.end())
  {
    logComputeMessage("Using existing utility instance.");
    return pos - computeConfig.begin();
  }

  logComputeMessage("Adding utility kernels");

  util.kernelIndices[COMPUTE_UTIL_SHOW_MATRIX_KERNEL] = kernelNames.size();
  kernelNames.push_back("showMatrix");

  // use size tuned for best performance
  util.batchSize = 8;
  // override if specified
  if (dataMap.find(ComputeUtilBatchSize) != dataMap.end())
  {
    util.batchSize = atoi(dataMap[ComputeUtilBatchSize].c_str());
  }
  oldType.push_back(getKeyName(ComputeUtilBatchSize));
  newType.push_back(to_string(util.batchSize));

  util.maxWorkgroupSize = compute->maxThreadsPerGroup();
  // override if specified
  if (dataMap.find(ComputeUtilMaxWorkgroupSize) != dataMap.end())
  {
    util.maxWorkgroupSize = atoi(dataMap[ComputeUtilMaxWorkgroupSize].c_str());
  }
  oldType.push_back(getKeyName(ComputeUtilMaxWorkgroupSize));
  newType.push_back(to_string(util.maxWorkgroupSize));

  util.structSize = 4;
  // default sizes
  if (dataMap.find(ComputeUtilStructSize) != dataMap.end())
  {
    util.structSize = atoi(dataMap[ComputeUtilStructSize].c_str());
  }

  if (dataMap.find(ComputeUtilStructMemberType) != dataMap.end())
  {
    util.structMemberSize = 4;
    // default sizes
    if (dataMap.find(ComputeUtilStructMemberSize) != dataMap.end())
    {
      util.structMemberSize = atoi(dataMap[ComputeUtilStructMemberSize].c_str());
    }
  }
  else
  {
    util.structMemberSize = util.structSize;
  }

  if (dataMap.find(ComputeUtilStructType) != dataMap.end())
  {
    if (dataMap.find(ComputeUtilSkipParallelPrimitives) == dataMap.end())
    {
      util.kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL] = kernelNames.size();
      kernelNames.push_back("reduce");

      util.kernelIndices[COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL] = kernelNames.size();
      kernelNames.push_back("reduce2DKernel");

      if (dataMap.find(ComputeUtilStructTypeIntegral) != dataMap.end())
      {
        util.kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL] = kernelNames.size();
        kernelNames.push_back("prefixGroupScanKernel");

        util.kernelIndices[COMPUTE_UTIL_BITONIC_SORT] = kernelNames.size();
        kernelNames.push_back("bitonicSort32BitKernel");

        util.kernelIndices[COMPUTE_UTIL_RADIX_SORT1] = kernelNames.size();
        kernelNames.push_back("radixSort32BitReduceKernel");

        util.kernelIndices[COMPUTE_UTIL_RADIX_SORT2] = kernelNames.size();
        kernelNames.push_back("radixSort32BitSortKernel");

        oldType.push_back("SortBits");
        newType.push_back(to_string(RADIX_SORT_BIT_COUNT));

        oldType.push_back("RadixReductionPackingExp");
        newType.push_back(to_string(RADIX_REDUCTION_PACKING_EXP));

        oldType.push_back("RadixPrefixScanPackingExp");
        newType.push_back(to_string(RADIX_PREFIX_SCAN_PACKING_EXP));

        oldType.push_back("BankConflictShift");
        newType.push_back(to_string(2));

        util.kernelIndices[COMPUTE_UTIL_COMPACT_SPARSE_ARRAY] = kernelNames.size();
        kernelNames.push_back("compactSparseArray");
      }
    }
    else
      if (dataMap.find(ComputeUtilOnlyReduce) != dataMap.end())
      {
        util.kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL] = kernelNames.size();
        kernelNames.push_back("reduce");
      }

    util.localArrays.clear();
    util.localArrays.reserve(UtilTempBufferMax);

    for (uint i = 0; i < UtilTempBufferMax; i++)
    {
      util.localArrays.push_back(NULL);
    }

    if (dataMap.find(ComputeUtilOnlyReduce) == dataMap.end())
    {
      util.kernelIndices[COMPUTE_UTIL_CONSOLIDATE_FROM_PARTITIONS] = kernelNames.size();
      kernelNames.push_back("consolidateFromPartitionsKernel");

      if (dataMap.find(ComputeUtilIdentityStructType) != dataMap.end())
      {
        util.kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL] = kernelNames.size();
        kernelNames.push_back("sumIrregular2DKernel");
      }
    }
  }

  util.registerShader(compute, "ComputeUtils.shader", &oldType, &newType);

  for (const string& kernelName : kernelNames)
  {
    logComputeMessage(kernelName.c_str());
    util.kernels.push_back(util.programs[0].createKernel(kernelName.c_str()));
  }

  logComputeMessage("\n");

  computeConfig.push_back(key);
  computeUtils.push_back(util);

  return computeUtils.size() - 1;
}

ComputeUtil* ComputeUtil::get(uint templateId)
{
  return &computeUtils[templateId];
}

void ComputeUtil::sum1D(ComputeInterface* compute, ComputeMemory* source, uint length, bool doMean)
{
  sum1D(compute, source, source, length, doMean);
}

//#define DEBUG_REDUCE

void ComputeUtil::sum1D(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, uint length, bool doMean)
{
  if (!localArrays[UtilTempReduceSum])
  {
    localArrays[UtilTempReduceSum] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempReduceSum])->create(compute, NULL, true);

    localArrays[UtilTempReduceStatus] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempReduceStatus])->create(compute, NULL, true);
  }

  DeviceArray<uint>* groupSum = (DeviceArray<uint>*)localArrays[UtilTempReduceSum];
  DeviceArray<uint>* groupStatus = (DeviceArray<uint>*)localArrays[UtilTempReduceStatus];

  uint groupCount = (length + this->maxWorkgroupSize * batchSize - 1) / (this->maxWorkgroupSize * batchSize);

  groupSum->resize(groupCount * 2 * this->structMemberSize/sizeof(uint), false);
  groupStatus->resize(groupCount, false);

  uint zero = REDUCE_STATUS_INVALID;
  uint mean = doMean ? 1 : 0;
  compute->setBuffer(groupStatus->device(), 0, groupCount * sizeof(uint), &zero, sizeof(uint));

  ComputeMemory* buffers[] = { destination, source, groupSum->device(), groupStatus->device() };

  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL];

  kernels[kernelIndex].setArgs(buffers, sizeof(buffers) / sizeof(ComputeMemory*));
  kernels[kernelIndex].setArg<uint>(&length, sizeof(buffers) / sizeof(ComputeMemory*));
  kernels[kernelIndex].setArg<uint>(&mean, 1 + sizeof(buffers) / sizeof(ComputeMemory*));

  size_t workgroupSize[3] = { 1, 1, 1 };
  size_t workgroupCount[3] = { 1, 1, 1 };
  workgroupSize[0] = this->maxWorkgroupSize;
  workgroupCount[0] = groupCount;

  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

#ifdef DEBUG_REDUCE
  groupSum->syncHost();
  compute->sync();
#endif

}

void ComputeUtil::sumRegular2D(ComputeInterface* compute, ComputeMemory* array2D, uint length, uint subArrayElements, bool doMean)
{
  const uint iterations = mCeilExpOf2(subArrayElements);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL];

  kernels[kernelIndex].setArg(array2D, 0);
  kernels[kernelIndex].setArg<uint>(&length, 1);
  kernels[kernelIndex].setArg<uint>(&subArrayElements, 2);
  kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 4);
  kernels[kernelIndex].setArg<uint>(&divideFlag, 5);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  for (uint i = 0; i < iterations; i++)
  {
    kernels[kernelIndex].setArg<uint>(&i, 3);

    maxLocalIterations = ((iterations - i) > maxThreadsPerGroupExponent) ? 1 : (iterations - i);
    divideFlag = ((iterations - i) <= maxThreadsPerGroupExponent) ? 1 : 0;

    // set only if needed
    if (maxLocalIterations != 1)
    {
      kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 4);
    }
    if (divideFlag && doMean)
    {
      kernels[kernelIndex].setArg<uint>(&divideFlag, 5);
    }

    const uint arraysPerGroup = compute->maxThreadsPerGroup() / subArrayElements;
    const uint subArrays = length / subArrayElements;
    const float totalthreads = compute->maxThreadsPerGroup() * (float(subArrays) / arraysPerGroup);

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(totalthreads / ((1 << i) * 2)));
    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

    if (iterations - i <= maxThreadsPerGroupExponent) break;
  }
}

void ComputeUtil::sumIrregular2D(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, ComputeMemory* identity, ComputeMemory* partitions, uint length, bool doMean)
{
  if (!localArrays[UtilTempReduceSum])
  {
    localArrays[UtilTempReduceSum] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempReduceSum])->create(compute, NULL, true);

    localArrays[UtilTempReduceStatus] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempReduceStatus])->create(compute, NULL, true);
  }

  DeviceArray<uint>* groupSum = (DeviceArray<uint>*)localArrays[UtilTempReduceSum];
  DeviceArray<uint>* groupStatus = (DeviceArray<uint>*)localArrays[UtilTempReduceStatus];

  uint groupCount = (length + this->maxWorkgroupSize - 1) / (this->maxWorkgroupSize);

  groupSum->resize(groupCount * this->structMemberSize/sizeof(uint), false);
  groupStatus->resize(groupCount, false);

  uint zero = REDUCE_STATUS_INVALID;
  compute->setBuffer(groupStatus->device(), 0, groupCount * sizeof(uint), &zero, sizeof(uint));

  uint divideFlag = doMean;
  uint threadGroupSizeExp = mCeilExpOf2(compute->maxThreadsPerGroup());
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL];

  kernels[kernelIndex].setArg(destination, 0);
  kernels[kernelIndex].setArg(source, 1);
  kernels[kernelIndex].setArg(identity, 2);
  kernels[kernelIndex].setArg(partitions, 3);
  kernels[kernelIndex].setArg(groupSum->device(), 4);
  kernels[kernelIndex].setArg(groupStatus->device(), 5);
  kernels[kernelIndex].setArg<uint>(&threadGroupSizeExp, 6);
  kernels[kernelIndex].setArg<uint>(&length, 7);
  kernels[kernelIndex].setArg<uint>(&divideFlag, 8);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  compute->configureSize(workgroupSize, workgroupCount, length);
  workgroupSize[0] = compute->maxThreadsPerGroup();
  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);
}

//#define DEBUG_PREFIX_SCAN

void ComputeUtil::compactSparseArray(ComputeInterface* compute, ComputeMemory* compactArrayCount, ComputeMemory* compactIndexArray, ComputeMemory* selectionArray, uint statusArrayLength)
{
  if (!localArrays[UtilTempPrefixGroupSum])
  {
    localArrays[UtilTempPrefixGroupSum] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempPrefixGroupSum])->create(compute, NULL, true);

    localArrays[UtilTempPrefixGroupStatus] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempPrefixGroupStatus])->create(compute, NULL, false);
  }

  DeviceArray<uint>* groupSum = (DeviceArray<uint>*)localArrays[UtilTempPrefixGroupSum];
  DeviceArray<uint>* groupStatus = (DeviceArray<uint>*)localArrays[UtilTempPrefixGroupStatus];

  uint groupCount = (statusArrayLength + this->maxWorkgroupSize * batchSize - 1) / (this->maxWorkgroupSize*batchSize);

  groupSum->resize(groupCount * 2 * this->structMemberSize/sizeof(uint), false);
  groupStatus->resize(groupCount, false);

  uint zero = PREFIX_SCAN_STATUS_INVALID;
  compute->setBuffer(groupStatus->device(), 0, groupCount * sizeof(uint), &zero, sizeof(uint));

  ComputeMemory* buffers[] = { compactArrayCount, compactIndexArray, selectionArray, groupSum->device(), groupStatus->device() };

  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_COMPACT_SPARSE_ARRAY];

  kernels[kernelIndex].setArgs(buffers, sizeof(buffers) / sizeof(ComputeMemory*));
  kernels[kernelIndex].setArg<uint>(&statusArrayLength, sizeof(buffers) / sizeof(ComputeMemory*));

  size_t workgroupSize[3] = { 1, 1, 1 };
  size_t workgroupCount[3] = { 1, 1, 1 };
  workgroupSize[0] = this->maxWorkgroupSize;
  workgroupCount[0] = groupCount;

  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

#ifdef DEBUG_PREFIX_SCAN
  groupSum->syncHost();
  compute->sync();
#endif
}

void ComputeUtil::consolidateFromPartitions(ComputeInterface* compute, ComputeMemory* source, ComputeMemory* destination, ComputeMemory* partitions, ComputeMemory* partitionsCount, uint partitionsCountHost)
{
  size_t workgroupSize[3];
  size_t workgroupCount[3];
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_CONSOLIDATE_FROM_PARTITIONS];

  compute->configureSize(workgroupSize, workgroupCount, partitionsCountHost);

  ComputeMemory* buffers[] = {
    source,
    destination,
    partitions,
    partitionsCount
  };
  kernels[kernelIndex].setArgs(buffers, 4);
  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);
}

void ComputeUtil::prefixScan1D(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, uint length)
{
  if (!localArrays[UtilTempPrefixGroupSum])
  {
    localArrays[UtilTempPrefixGroupSum] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempPrefixGroupSum])->create(compute, NULL, true);

    localArrays[UtilTempPrefixGroupStatus] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempPrefixGroupStatus])->create(compute, NULL, false);
  }

  DeviceArray<uint>* groupSum = (DeviceArray<uint>*)localArrays[UtilTempPrefixGroupSum];
  DeviceArray<uint>* groupStatus = (DeviceArray<uint>*)localArrays[UtilTempPrefixGroupStatus];

  uint groupCount = (length + this->maxWorkgroupSize * batchSize - 1) / (this->maxWorkgroupSize*batchSize);

  groupSum->resize(groupCount * 2 * this->structMemberSize/sizeof(uint), false);
  groupStatus->resize(groupCount, false);

  uint zero = PREFIX_SCAN_STATUS_INVALID;
  compute->setBuffer(groupStatus->device(), 0, groupCount * sizeof(uint), &zero, sizeof(uint));

  ComputeMemory* buffers[] = { destination, source, groupSum->device(), groupStatus->device() };

  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL];

  kernels[kernelIndex].setArgs(buffers, sizeof(buffers) / sizeof(ComputeMemory*));
  kernels[kernelIndex].setArg<uint>(&length, sizeof(buffers) / sizeof(ComputeMemory*));

  size_t workgroupSize[3] = { 1, 1, 1 };
  size_t workgroupCount[3] = { 1, 1, 1 };
  workgroupSize[0] = this->maxWorkgroupSize;
  workgroupCount[0] = groupCount;

  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

#ifdef DEBUG_PREFIX_SCAN
  groupSum->syncHost();
  compute->sync();
#endif

}

void ComputeUtil::bitonicSort32Bit(ComputeInterface* compute, ComputeMemory* array1D, uint length)
{

}

//#define DEBUG_RADIX_SORT

void ComputeUtil::radixSort32Bit(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, uint length)
{
  const uint kernelIndex1 = kernelIndices[COMPUTE_UTIL_RADIX_SORT1];
  const uint kernelIndex2 = kernelIndices[COMPUTE_UTIL_RADIX_SORT2];

  size_t workgroupSize[3] = { 1, 1, 1 };
  size_t workgroupCount[3] = { 1, 1, 1 };

  const uint localSortThreads = compute->simdSize();
  const uint radixBlockInstances = 256 / localSortThreads;

  if (!localArrays[UtilTempRadixGroupSum])
  {

#ifdef DEBUG_RADIX_SORT
    localArrays[UtilTempRadixGroupSum] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempRadixGroupSum])->create(compute, NULL, true);
#else
    localArrays[UtilTempRadixGroupSum] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[UtilTempRadixGroupSum])->create(compute, NULL, false);
#endif

  }

  const uint groups = compute->maxCores();
  DeviceArray<uint>* localSumBuffer = (DeviceArray<uint>*)localArrays[UtilTempRadixGroupSum];

  localSumBuffer->resize(groups * radixBlockInstances * (1 << RADIX_SORT_BIT_COUNT), false);

  kernels[kernelIndex1].setArg(localSumBuffer->device(), 1);
  kernels[kernelIndex1].setArg<uint>(&length, 3);

  kernels[kernelIndex2].setArg(localSumBuffer->device(), 2);
  kernels[kernelIndex2].setArg<uint>(&length, 4);

  uint i;

  workgroupSize[0] = localSortThreads * radixBlockInstances;
  workgroupCount[0] = groups;

  for (i = 0; i < 32; i += RADIX_SORT_BIT_COUNT)
  {
    kernels[kernelIndex1].setArg((i & RADIX_SORT_BIT_COUNT) ? destination : source, 0);

    kernels[kernelIndex2].setArg((i & RADIX_SORT_BIT_COUNT) ? source : destination, 0);
    kernels[kernelIndex2].setArg((i & RADIX_SORT_BIT_COUNT) ? destination : source, 1);

    kernels[kernelIndex1].setArg<uint>(&i, 2);

    compute->execute(kernels[kernelIndex1], workgroupSize, workgroupCount);

#ifdef DEBUG_RADIX_SORT
    localSumBuffer->syncHost();
    compute->sync();
#endif

    prefixScan1D(compute, localSumBuffer->device(), localSumBuffer->device(), groups * radixBlockInstances * (1 << RADIX_SORT_BIT_COUNT));

#ifdef DEBUG_RADIX_SORT
    localSumBuffer->syncHost();
    compute->sync();
#endif

    kernels[kernelIndex2].setArg<uint>(&i, 3);

    compute->execute(kernels[kernelIndex2], workgroupSize, workgroupCount);

#ifdef DEBUG_RADIX_SORT
    compute->sync();
#endif

  }

  if ((i & RADIX_SORT_BIT_COUNT) == 0)
  {
    compute->copyBuffer(source, destination, 0, 0, length * sizeof(SortNode32));
  }
}

void ComputeUtil::showMatrix(ComputeInterface* compute, ComputeMemory* memory, uint rowSize, uint strideIn4Byte, uint length)
{
  size_t workgroupSize[3];
  size_t workgroupCount[3];
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SHOW_MATRIX_KERNEL];

  compute->configureSize(workgroupSize, workgroupCount, length / rowSize);
  kernels[kernelIndex].setArg(memory, 0);
  kernels[kernelIndex].setArg<uint>(&rowSize, 1);
  kernels[kernelIndex].setArg<uint>(&strideIn4Byte, 2);
  kernels[kernelIndex].setArg<uint>(&length, 3);
  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);
}
