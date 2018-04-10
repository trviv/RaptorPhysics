#include "ComputeUtils.h"

vector<ComputeUtil> computeUtils;
vector<string>      computeConfig;

#define COMPUTE_UTIL_SUM_1D_KERNEL                  0
#define COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL          1
#define COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL        2
#define COMPUTE_UTIL_SHOW_MATRIX_KERNEL             3
#define COMPUTE_UTIL_SECTION_OFFSET                 4
#define COMPUTE_UTIL_CONSOLIDATE_FROM_PARTITIONS    5
#define COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL1  6
#define COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL2  7
#define COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL3  8

string getKeyName(ComputeUtilKey key)
{
  switch (key)
  {
  case ComputeUtilStructType:
    return "StructType";
  case ComputeUtilStructMember:
    return "StructMember";

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
  }
  return "";
}

uint ComputeUtil::create(ComputeInterface* compute, map<ComputeUtilKey, string>& dataMap, const vector<string>* includeFiles)
{
  ComputeUtil util;
  vector<string> oldType;
  vector<string> newType;
  vector<string> kernelNames;

  util.includeFiles.insert(util.includeFiles.begin(), "ComputeHeader.shader");
  if (includeFiles)
  {
    for (auto i : *includeFiles)
    {
      util.includeFiles.push_back(i);
    }
  }

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

  util.includeFiles.insert(util.includeFiles.begin() + 1, "ComputeShared.h");
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

  if (dataMap.find(ComputeUtilStructType) != dataMap.end())
  {
    util.kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL] = kernelNames.size();
    kernelNames.push_back("sum1DKernel");

    util.kernelIndices[COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL] = kernelNames.size();
    kernelNames.push_back("sumRegular2DKernel");

    util.kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL1] = kernelNames.size();
    kernelNames.push_back("prefixGroupScanKernel");

    util.kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL2] = kernelNames.size();
    kernelNames.push_back("prefixTopScanKernel");

    util.kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL3] = kernelNames.size();
    kernelNames.push_back("prefixAddOffsetKernel");

    util.localArrays.clear();
    util.localArrays.reserve(1);
    util.localArrays.push_back(NULL);

    util.kernelIndices[COMPUTE_UTIL_CONSOLIDATE_FROM_PARTITIONS] = kernelNames.size();
    kernelNames.push_back("consolidateFromPartitionsKernel");

    if (dataMap.find(ComputeUtilIdentityStructType) != dataMap.end())
    {
      util.kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL] = kernelNames.size();
      kernelNames.push_back("sumIrregular2DKernel");
    }
  }

  for (const string& kernelName : kernelNames)
  {
    logComputeMessage(kernelName.c_str());
    util.programs.push_back(compute->createTemplateProgram("ComputeUtils.shader", &oldType, &newType, &util.includeFiles));
    util.kernels.push_back(util.programs[0].createKernel(kernelName.c_str()));
  }
  logComputeMessage("\n");

  computeUtils.push_back(util);

  return computeUtils.size() - 1;
}

ComputeUtil* ComputeUtil::get(uint templateId)
{
  return &computeUtils[templateId];
}

void ComputeUtil::sum1D(ComputeInterface* compute, ComputeMemory* array1D, uint length, bool doMean)
{
  const uint iterations = mCeilExpOf2(length);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL];

  kernels[kernelIndex].setArg(array1D, 0);
  kernels[kernelIndex].setArg<uint>(&length, 1);
  kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 3);
  kernels[kernelIndex].setArg<uint>(&divideFlag, 4);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  for (uint i = 0; i < iterations; i++)
  {
    maxLocalIterations = ((iterations - i) > maxThreadsPerGroupExponent) ? 1 : (iterations - i);

    kernels[kernelIndex].setArg<uint>(&i, 2);
    if (maxLocalIterations != 1)
    {
      kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 3);
    }

    divideFlag = ((iterations - i) <= maxThreadsPerGroupExponent) ? 1 : 0;
    if (divideFlag && doMean) // set only if needed
    {
      kernels[kernelIndex].setArg<uint>(&divideFlag, 4);
    }

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(float(length) / ((1 << i) * 2)));
    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

    if (iterations - i <= maxThreadsPerGroupExponent) break;
  }
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

void ComputeUtil::sumIrregular2D(ComputeInterface* compute, ComputeMemory* array2D, ComputeMemory* identity, ComputeMemory* partitions, ComputeMemory* partitionCount, uint length, uint maxPartitionLength, bool doMean)
{
  sumIrregular2D(compute, array2D, array2D, identity, partitions, partitionCount, length, maxPartitionLength, doMean);
}

void ComputeUtil::sumIrregular2D(ComputeInterface* compute, ComputeMemory* array2D, ComputeMemory* consolidatedArray, ComputeMemory* identity, ComputeMemory* partitions, ComputeMemory* partitionCount, uint length, uint maxPartitionLength, bool doMean)
{
  const uint iterations = mCeilExpOf2(maxPartitionLength);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL];

  kernels[kernelIndex].setArg(array2D, 0);
  kernels[kernelIndex].setArg(consolidatedArray, 1);
  kernels[kernelIndex].setArg(identity, 2);
  kernels[kernelIndex].setArg(partitions, 3);
  kernels[kernelIndex].setArg(partitionCount, 4);
  kernels[kernelIndex].setArg<uint>(&length, 5);
  kernels[kernelIndex].setArg<uint>(&maxPartitionLength, 6);
  kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 8);
  kernels[kernelIndex].setArg<uint>(&divideFlag, 9);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  for (uint i = 0; i < iterations; i++)
  {
    kernels[kernelIndex].setArg<uint>(&i, 7);

    maxLocalIterations = ((iterations - i) > maxThreadsPerGroupExponent) ? 1 : (iterations - i);
    divideFlag = ((iterations - i) <= maxThreadsPerGroupExponent) ? 1 : 0;

    // set only if needed
    if (maxLocalIterations != 1)
    {
      kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 8);
    }
    if (divideFlag && doMean)
    {
      kernels[kernelIndex].setArg<uint>(&divideFlag, 9);
    }

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(float(length) / ((1 << i))));
    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

    if (iterations - i <= maxThreadsPerGroupExponent) break;
  }
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

#define PREFIX_TEMP_GROUP_SUM   0
//#define DEBUG_PREFIX_SCAN

void ComputeUtil::prefixScan1D(ComputeInterface* compute, ComputeMemory* array1D, uint length)
{
  if (!localArrays[PREFIX_TEMP_GROUP_SUM])
  {
    localArrays[PREFIX_TEMP_GROUP_SUM] = new DeviceArray<uint>();
    ((DeviceArray<uint>*)localArrays[PREFIX_TEMP_GROUP_SUM])->create(compute, NULL, true);
  }

  DeviceArray<uint>* groupSum = (DeviceArray<uint>*)localArrays[PREFIX_TEMP_GROUP_SUM];

  uint groupCount = (length + (compute->maxThreadsPerGroup() << 1) - 1) / (compute->maxThreadsPerGroup() << 1);
  uint maxGroupCount = 2 << mCeilExpOf2(groupCount);

  groupSum->resize(maxGroupCount, false);

  {
    size_t workgroupSize[3];
    size_t workgroupCount[3];

    compute->configureSize(workgroupSize, workgroupCount, length);

    ComputeMemory* buffers[] = { array1D, groupSum->device(), array1D };

    const uint kernelIndex = kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL1];

    kernels[kernelIndex].setArgs(buffers, sizeof(buffers) / sizeof(ComputeMemory*));
    kernels[kernelIndex].setArg<uint>(&length, sizeof(buffers) / sizeof(ComputeMemory*));

    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

#ifdef DEBUG_PREFIX_SCAN
    groupSum->syncHost();
    compute->sync();
#endif

  }

  {
    size_t workgroupSize[3];
    size_t workgroupCount[3];

    compute->configureSize(workgroupSize, workgroupCount, compute->maxThreadsPerGroup());

    const uint kernelIndex = kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL2];
    //uint topIterations = ceil(float(length) / (compute->maxThreadsPerGroup() * 2048));
    kernels[kernelIndex].setArg(groupSum->device(), 0);
    kernels[kernelIndex].setArg<uint>(&groupCount, 1);
    kernels[kernelIndex].setArg<uint>(&maxGroupCount, 2);
    //kernels[kernelIndex].setArg<uint>(&topIterations, 3);
    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

#ifdef DEBUG_PREFIX_SCAN
    groupSum->syncHost();
    compute->sync();
#endif

  }

  if (groupCount > 2)
  {
    size_t workgroupSize[3];
    size_t workgroupCount[3];

    compute->configureSize(workgroupSize, workgroupCount, compute->maxThreadsPerGroup() * ((2 * groupCount) - 2));

    const uint kernelIndex = kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL3];

    kernels[kernelIndex].setArg(array1D, 0);
    kernels[kernelIndex].setArg(groupSum->device(), 1);
    kernels[kernelIndex].setArg<uint>(&length, 2);
    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);
  }
}


void ComputeUtil::radixSort32Bit(ComputeInterface* compute, ComputeMemory* array1D, uint length)
{

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