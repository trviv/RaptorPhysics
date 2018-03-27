#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

#include "Math.h"
#include "ShaderEntity.h"

enum ComputeUtilKey
{
  ComputeUtilStructType,
  ComputeUtilStructMember,

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
  uint kernelIndices[7];

public:

  static uint create(ComputeInterface* compute, map<ComputeUtilKey, string>& dataMap, const vector<string>* includeFiles = NULL);

  static ComputeUtil* get(uint templateId);

  void sum1D(ComputeInterface* compute, ComputeMemory* memory, uint length, bool doMean = false);

  void sumRegular2D(ComputeInterface* compute, ComputeMemory* memory, uint length, uint subArrayElements, bool doMean = false);

  void sumIrregular2D(ComputeInterface* compute, ComputeMemory* memory, ComputeMemory* identity, ComputeMemory* partitions, uint length, uint maxPartitionLength, bool doMean = false);

  //void prefixSum1D(ComputeInterface* compute, ComputeMemory* memory, uint length, bool doMean = false);

  void consolidateFromPartitions(ComputeInterface* compute, ComputeMemory* source, ComputeMemory* destination, ComputeMemory* partitions, ComputeMemory* partitionsCount, uint partitionsCountHost);

  void showMatrix(ComputeInterface* compute, ComputeMemory* memory, uint rowSize, uint strideIn4Byte, uint length);
};

#endif