#include "ComputeUtils.h"

vector<ComputeUtil> computeUtils;
vector<string>      computeConfig;

#define COMPUTE_UTIL_SUM_1D_KERNEL                  0
#define COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL          1
#define COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL        2
#define COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL  3
#define COMPUTE_UTIL_SHOW_MATRIX_KERNEL             4
#define COMPUTE_UTIL_SECTION_OFFSET                 5
#define COMPUTE_UTIL_COPY_FROM_OFFSETS_KERNEL       6

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
  case ComputeUtilStructIdentity:
    return "StructIdentity";
  case ComputeUtilIndexStructType:
    return "IndexStructType";
  case ComputeUtilIndexStructMember:
    return "IndexStructMember";
  case ComputeUtilCustomFunctionSuffix:
    return "FunctionSuffix";
  }
  return "";
}

uint ComputeUtil::create(ComputeInterface* compute, map<ComputeUtilKey, string>& dataMap, const vector<string>* includeFiles)
{
  ComputeUtil util;
  vector<string> oldType;
  vector<string> newType;

  util.includeFiles.push_back("ComputeHeader.shader");
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

  vector<string> kernelNames;

  logComputeMessage("Adding utility kernels");
  if (dataMap.find(ComputeUtilStructType) != dataMap.end())
  {
    util.kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL] = kernelNames.size();
    kernelNames.push_back("sum1DKernel");
    util.kernelIndices[COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL] = kernelNames.size();
    kernelNames.push_back("sumRegular2DKernel");
    //util.kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL] = kernelNames.size();
    //kernelNames.push_back("parallelPrefixSum1D");
  }

  if (dataMap.find(ComputeUtilIndexStructType) != dataMap.end())
  {
    util.kernelIndices[COMPUTE_UTIL_SECTION_OFFSET] = kernelNames.size();
    kernelNames.push_back("sectionOffsetsKernel");

    if (dataMap.find(ComputeUtilStructType) != dataMap.end())
    {
      util.kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL] = kernelNames.size();
      kernelNames.push_back("sumIrregular2DKernel");

      util.kernelIndices[COMPUTE_UTIL_COPY_FROM_OFFSETS_KERNEL] = kernelNames.size();
      kernelNames.push_back("copyFromOffsetsKernel");
    }
  }

  if (dataMap.find(ComputeUtilCustomFunctionSuffix) != dataMap.end())
  {
    oldType.push_back("AddFunction");
    newType.push_back("add" + dataMap[ComputeUtilCustomFunctionSuffix]);

    oldType.push_back("DivFunction");
    newType.push_back("div" + dataMap[ComputeUtilCustomFunctionSuffix]);

    oldType.push_back("CopyFunction");
    newType.push_back("copy" + dataMap[ComputeUtilCustomFunctionSuffix]);
  }

  util.kernelIndices[COMPUTE_UTIL_SHOW_MATRIX_KERNEL] = kernelNames.size();
  kernelNames.push_back("showMatrix");

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

void ComputeUtil::sum1D(ComputeInterface* compute, ComputeMemory* memory, uint length, bool doMean)
{
  const uint iterations = mCeilExpOf2(length);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL];

  kernels[kernelIndex].setArg(memory, 0);
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

void ComputeUtil::sumRegular2D(ComputeInterface* compute, ComputeMemory* memory, uint length, uint subArrayElements, bool doMean)
{
  const uint iterations = mCeilExpOf2(subArrayElements);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL];

  kernels[kernelIndex].setArg(memory, 0);
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

void ComputeUtil::sumIrregular2D(ComputeInterface* compute, ComputeMemory* memory, ComputeMemory* partitions, uint length, uint maxPartitionLength, bool doMean)
{
  sumIrregular2D(compute, memory, NULL, partitions, length, maxPartitionLength, doMean);
}

void ComputeUtil::sumIrregular2D(ComputeInterface* compute, ComputeMemory* memory, ComputeMemory* identity, ComputeMemory* partitions, uint length, uint maxPartitionLength, bool doMean)
{
  const uint iterations = mCeilExpOf2(maxPartitionLength);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL];

  uint index = 0;
  kernels[kernelIndex].setArg(memory, 0);
  if (identity)
  {
    kernels[kernelIndex].setArg(identity, ++index);
  }

  kernels[kernelIndex].setArg(partitions, index + 1);
  kernels[kernelIndex].setArg<uint>(&length, index + 2);
  kernels[kernelIndex].setArg<uint>(&maxPartitionLength, index + 3);
  kernels[kernelIndex].setArg<uint>(&maxLocalIterations, index + 5);
  kernels[kernelIndex].setArg<uint>(&divideFlag, index + 6);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  for (uint i = 0; i < iterations; i++)
  {
    kernels[kernelIndex].setArg<uint>(&i, index + 4);

    maxLocalIterations = ((iterations - i) > maxThreadsPerGroupExponent) ? 1 : (iterations - i);
    divideFlag = ((iterations - i) <= maxThreadsPerGroupExponent) ? 1 : 0;

    // set only if needed
    if (maxLocalIterations != 1)
    {
      kernels[kernelIndex].setArg<uint>(&maxLocalIterations, index + 5);
    }
    if (divideFlag && doMean)
    {
      kernels[kernelIndex].setArg<uint>(&divideFlag, index + 6);
    }

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(float(length) / ((1 << i))));
    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

    if (iterations - i <= maxThreadsPerGroupExponent) break;
  }
}

void ComputeUtil::prefixSum1D(ComputeInterface* compute, ComputeMemory* memory, uint length, bool doMean)
{
  const uint iterations = mCeilExpOf2(length);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint backwards = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL];

  kernels[kernelIndex].setArg(memory, 0);
  kernels[kernelIndex].setArg<uint>(&length, 1);
  kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 3);

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

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(float(length) / ((1 << (i / 2)) * 2)));
    compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);

    if (iterations - i <= maxThreadsPerGroupExponent) break;
  }
}

void ComputeUtil::createSectionOffsets(ComputeInterface* compute, ComputeMemory* sectionOffsets, ComputeMemory* sectionOffsetCount, ComputeMemory* sections, uint sectionCount)
{
  size_t workgroupSize[3];
  size_t workgroupCount[3];
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SECTION_OFFSET];

  compute->configureSize(workgroupSize, workgroupCount, compute->maxThreadsPerGroup());
  kernels[kernelIndex].setArg(sectionOffsets, 0);
  kernels[kernelIndex].setArg(sectionOffsetCount, 1);
  kernels[kernelIndex].setArg(sections, 2);
  kernels[kernelIndex].setArg<uint>(&sectionCount, 3);
  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);
}

void ComputeUtil::copySectionOffsets(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, ComputeMemory* sectionOffsets, ComputeMemory* sectionOffsetCount, uint sectionOffsetCountHost)
{
  size_t workgroupSize[3];
  size_t workgroupCount[3];
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_COPY_FROM_OFFSETS_KERNEL];

  compute->configureSize(workgroupSize, workgroupCount, sectionOffsetCountHost);

  ComputeMemory* buffers[] = {
    destination,
    source,
    sectionOffsets,
    sectionOffsetCount
  };
  kernels[kernelIndex].setArgs(buffers, 4);
  compute->execute(kernels[kernelIndex], workgroupSize, workgroupCount);
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