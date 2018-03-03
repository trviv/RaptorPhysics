#include "ComputeUtils.h"

vector<ComputeUtil> computeUtils;
vector<string>      computeConfig;

#define COMPUTE_UTIL_SUM_1D_KERNEL                  0
#define COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL          1
#define COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL        2
#define COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL  3
#define COMPUTE_UTIL_SHOW_MATRIX_KERNEL             4

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
      util.kernelIndices[COMPUTE_UTIL_SUM_1D_KERNEL] = kernelNames.size();
      kernelNames.push_back("sum1DKernel");
      util.kernelIndices[COMPUTE_UTIL_SUM_REGULAR_2D_KERNEL] = kernelNames.size();
      kernelNames.push_back("sumRegular2DKernel");
      util.kernelIndices[COMPUTE_UTIL_PARALLEL_PREFIX_SUM_1D_KERNEL] = kernelNames.size();
      kernelNames.push_back("parallelPrefixSum1D");
    }
    else if (tuples[i].key == ComputeUtilIndexStructType)
    {
      util.kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL] = kernelNames.size();
      kernelNames.push_back("sumIrregular2DKernel");
    }
    else if (tuples[i].key == ComputeUtilCustomFunctionSuffix)
    {
      oldType.push_back("AddFunction");
      newType.push_back("add" + tuples[i].value);

      oldType.push_back("DivFunction");
      newType.push_back("div" + tuples[i].value);
    }
  }

  util.kernelIndices[COMPUTE_UTIL_SHOW_MATRIX_KERNEL] = kernelNames.size();
  kernelNames.push_back("showMatrix");

  for (const string& kernelName : kernelNames)
  {
    //oldType.push_back("GroupSize");
    //newType.push_back("GroupSize");
    util.programs.push_back(compute->createTemplateProgram("ComputeUtils.shader", &oldType, &newType, &util.includeFiles));
    util.kernels.push_back(util.programs[0].createKernel(kernelName.c_str()));
  }

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

/*void ComputeUtil::sum1D(ComputeInterface* compute, ComputeMemory* memory, uint length, bool doMean)
{
ComputeMemory* temp = compute->heap.alloc(length);
compute->copyBuffer(memory, temp, 0, 0, length);

const uint iterations = mExpOf2(length);
const uint maxThreadsPerGroupExponent = mExpOf2(compute->maxThreadsPerGroup() << 1);

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
}*/

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
  const uint iterations = mCeilExpOf2(maxPartitionLength);
  const uint maxThreadsPerGroupExponent = mCeilExpOf2(compute->maxThreadsPerGroup() << 1);

  uint divideFlag = 0;
  uint maxLocalIterations = 1;
  const uint kernelIndex = kernelIndices[COMPUTE_UTIL_SUM_IRREGULAR_2D_KERNEL];

  kernels[kernelIndex].setArg(memory, 0);
  kernels[kernelIndex].setArg(partitions, 1);
  kernels[kernelIndex].setArg<uint>(&length, 2);
  kernels[kernelIndex].setArg<uint>(&maxPartitionLength, 3);
  kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 5);
  kernels[kernelIndex].setArg<uint>(&divideFlag, 6);

  size_t workgroupSize[3];
  size_t workgroupCount[3];

  for (uint i = 0; i < iterations; i++)
  {
    kernels[kernelIndex].setArg<uint>(&i, 4);

    maxLocalIterations = ((iterations - i) > maxThreadsPerGroupExponent) ? 1 : (iterations - i);
    divideFlag = ((iterations - i) <= maxThreadsPerGroupExponent) ? 1 : 0;

    // set only if needed
    if (maxLocalIterations != 1)
    {
      kernels[kernelIndex].setArg<uint>(&maxLocalIterations, 5);
    }
    if (divideFlag && doMean)
    {
      kernels[kernelIndex].setArg<uint>(&divideFlag, 6);
    }

    compute->configureSize(workgroupSize, workgroupCount, (uint)mCeil(float(length) / ((1 << i) * 2)));
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

void ComputeUtil::determineGroups(ComputeMemory* group, ComputeMemory* partitions, uint partitionCount)
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