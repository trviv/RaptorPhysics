#include "ComputeUtils.h"

vector<ComputeUtil> computeUtils;
vector<string>      computeConfig;

#define COMPUTE_UTIL_KERNEL_SUM           0
#define COMPUTE_UTIL_KERNEL_SHOW_MATRIX   1
#define COMPUTE_UTIL_KERNEL_SUM_PARTITION 2


string getKeyName(ComputeUtilKey key)
{
  switch (key)
  {
  case ComputeUtilStructType:
    return "StructType";
  case ComputeUtilStructMember:
    return "StructMember";
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

uint ComputeUtil::create(ComputeInterface* compute, const vector<ComputeUtilTuple>& tuples, const vector<string>* includeFiles)
{
  ComputeUtil util;

  util.includeFiles.push_back("ComputeHeader.shader");
  if (includeFiles)
  {
    for (auto i : *includeFiles)
    {
      util.includeFiles.push_back(i);
    }
  }

  string key;

  for (uint i = 0; i < tuples.size(); i++)
  {
    key += ":" + tuples[i].value;
  }

  for (const string& include : util.includeFiles)
  {
    key += ":" + include;
  }

  vector<string>::iterator pos = find(computeConfig.begin(), computeConfig.end(), key);
  if (pos != computeConfig.end())
  {
    return pos - computeConfig.begin();
  }

  vector<string> kernelNames;
  vector<string> oldType;
  vector<string> newType;

  for (uint i = 0; i < tuples.size(); i++)
  {
    oldType.push_back(getKeyName(tuples[i].key));
    newType.push_back(tuples[i].value);
    if (tuples[i].key == ComputeUtilStructType)
    {
      util.kernelIndices[COMPUTE_UTIL_KERNEL_SUM] = kernelNames.size();
      kernelNames.push_back("calculateSum");
    }
    else if (tuples[i].key == ComputeUtilIndexStructType)
    {
      util.kernelIndices[COMPUTE_UTIL_KERNEL_SUM_PARTITION] = kernelNames.size();
      kernelNames.push_back("calculateSumPartitions");
    }
    else if (tuples[i].key == ComputeUtilCustomFunctionSuffix)
    {
      oldType.push_back("AddFunction");
      newType.push_back("add" + tuples[i].value);

      oldType.push_back("DivFunction");
      newType.push_back("div" + tuples[i].value);
    }
  }

  util.kernelIndices[COMPUTE_UTIL_KERNEL_SHOW_MATRIX] = kernelNames.size();
  kernelNames.push_back("showMatrix");

  util.programs.push_back(compute->createTemplateProgram("ComputeUtils.shader", &oldType, &newType, &util.includeFiles));

  for (const string& kernelName : kernelNames)
  {
    util.kernels.push_back(util.programs[0].createKernel(kernelName.c_str()));
  }

  computeUtils.push_back(util);

  return computeUtils.size() - 1;
}

ComputeUtil* ComputeUtil::get(uint templateId)
{
  return &computeUtils[templateId];
}

void ComputeUtil::calculateSum(ComputeInterface* compute, ComputeMemory* memory, uint length, bool doMean)
{
  const uint iterations = mExpOf2(length);
  const uint maxThreadsPerGroupExponent = mExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  kernels[COMPUTE_UTIL_KERNEL_SUM].setArg(memory, 0);
  kernels[COMPUTE_UTIL_KERNEL_SUM].setArg<uint>(&length, 1);
  kernels[COMPUTE_UTIL_KERNEL_SUM].setArg<uint>(&maxLocalIterations, 3);
  kernels[COMPUTE_UTIL_KERNEL_SUM].setArg<uint>(&divideFlag, 4);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  for (uint i = 0; i < iterations; i++)
  {
    maxLocalIterations = ((iterations - i) > maxThreadsPerGroupExponent) ? 1 : (iterations - i);

    kernels[COMPUTE_UTIL_KERNEL_SUM].setArg<uint>(&i, 2);
    if (maxLocalIterations != 1)
    {
      kernels[COMPUTE_UTIL_KERNEL_SUM].setArg<uint>(&maxLocalIterations, 3);
    }

    divideFlag = ((iterations - i) <= maxThreadsPerGroupExponent) ? 1 : 0;
    if (divideFlag && doMean) // set only if needed
    {
      kernels[COMPUTE_UTIL_KERNEL_SUM].setArg<uint>(&divideFlag, 4);
    }

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(float(length) / ((1 << i) * 2)));
    compute->execute(kernels[COMPUTE_UTIL_KERNEL_SUM], workgroupSize, workgroupCount);

    if (iterations - i <= maxThreadsPerGroupExponent) break;
  }
}

void ComputeUtil::calculateSum(ComputeInterface* compute, ComputeMemory* memory, ComputeMemory* partitions, uint length, uint maxPartitionLength, bool doMean)
{
  const uint iterations = mExpOf2(maxPartitionLength);// << 1;// mExpOf2(length);
  const uint maxThreadsPerGroupExponent = mExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg(memory, 0);
  kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg(partitions, 1);
  kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg<uint>(&length, 2);
  kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg<uint>(&maxPartitionLength, 3);
  kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg<uint>(&maxLocalIterations, 5);
  kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg<uint>(&divideFlag, 6);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  for (uint i = 0; i < iterations; i++)
  {
    kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg<uint>(&i, 4);

    maxLocalIterations = ((iterations - i) > maxThreadsPerGroupExponent) ? 1 : (iterations - i);
    divideFlag = ((iterations - i) <= maxThreadsPerGroupExponent) ? 1 : 0;

    // set only if needed
    if (maxLocalIterations != 1)
    {
      kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg<uint>(&maxLocalIterations, 5);
    }
    if (divideFlag && doMean)
    {
      kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION].setArg<uint>(&divideFlag, 6);
    }

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(float(length) / ((1 << i) * 2)));
    compute->execute(kernels[COMPUTE_UTIL_KERNEL_SUM_PARTITION], workgroupSize, workgroupCount);

    if (iterations - i <= maxThreadsPerGroupExponent) break;
  }
}

void ComputeUtil::showMatrix(ComputeInterface* compute, ComputeMemory* memory, uint rowSize, uint strideIn4Byte, uint length)
{
  size_t workgroupSize[3];
  size_t workgroupCount[3];

  compute->configureSize(workgroupSize, workgroupCount, length / rowSize);
  kernels[COMPUTE_UTIL_KERNEL_SHOW_MATRIX].setArg(memory, 0);
  kernels[COMPUTE_UTIL_KERNEL_SHOW_MATRIX].setArg<uint>(&rowSize, 1);
  kernels[COMPUTE_UTIL_KERNEL_SHOW_MATRIX].setArg<uint>(&strideIn4Byte, 2);
  kernels[COMPUTE_UTIL_KERNEL_SHOW_MATRIX].setArg<uint>(&length, 3);
  compute->execute(kernels[COMPUTE_UTIL_KERNEL_SHOW_MATRIX], workgroupSize, workgroupCount);
}