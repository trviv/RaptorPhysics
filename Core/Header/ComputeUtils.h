#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

#include "Math.h"
#include "ShaderEntity.h"
#include "DeviceArray.h"

enum ComputeUtilKey
{
  ComputeUtilStructType,
  ComputeUtilStructMember,
  ComputeUtilStructTypeIntegral,

  ComputeUtilIdentityStructType,
  ComputeUtilIdentityStructMember,
  ComputeUtilIdentityFunction,

  ComputeUtilCustomAddFunction,
  ComputeUtilCustomCopyFunction,
  ComputeUtilCustomDivFunction,

  ComputeUtilMaxKey
};

/*
@class Class representing utility functions.
*/
class ComputeUtil : protected ShaderEntity
{
  uint kernelIndices[12];
  vector<void*> localArrays;

public:

  static uint create(ComputeInterface* compute, map<ComputeUtilKey, string>& dataMap, const vector<string>* includeFiles = NULL);

  static ComputeUtil* get(uint templateId);

  void sum1D(ComputeInterface* compute, ComputeMemory* array1D, uint length, bool doMean = false);

  void sumRegular2D(ComputeInterface* compute, ComputeMemory* array2D, uint length, uint subArrayElements, bool doMean = false);

  void sumIrregular2D(ComputeInterface* compute, ComputeMemory* array2D, ComputeMemory* identity, ComputeMemory* partitions, ComputeMemory* partitionCount, uint length, uint maxPartitionLength, bool doMean = false);

  void sumIrregular2D(ComputeInterface* compute, ComputeMemory* array2D, ComputeMemory* consolidatedArray, ComputeMemory* identity, ComputeMemory* partitions, ComputeMemory* partitionCount, uint length, uint maxPartitionLength, bool doMean = false);

  void consolidateFromPartitions(ComputeInterface* compute, ComputeMemory* source, ComputeMemory* destination, ComputeMemory* partitions, ComputeMemory* partitionsCount, uint partitionsCountHost);

  void prefixScan1D(ComputeInterface* compute, ComputeMemory* array1D, uint length);

  void bitonicSort32Bit(ComputeInterface* compute, ComputeMemory* array1D, uint length);

  void radixSort32Bit(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* array1D, uint length);

  void showMatrix(ComputeInterface* compute, ComputeMemory* memory, uint rowSize, uint strideIn4Byte, uint length);
};

#endif