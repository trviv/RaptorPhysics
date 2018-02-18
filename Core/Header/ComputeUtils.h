#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

#include "Math.h"
#include "ShaderEntity.h"

enum ComputeUtilKey
{
  ComputeUtilStructType,
  ComputeUtilStructMember,
  ComputeUtilStructIdentity,
  ComputeUtilIndexStructType,
  ComputeUtilIndexStructMember,
  ComputeUtilCustomFunctionSuffix
};

struct ComputeUtilTuple
{
  ComputeUtilKey  key;
  string          value;

  ComputeUtilTuple(ComputeUtilKey key, const string& value)
  {
    this->key = key;
    this->value = value;
  }
};

/*
@class Class representing utility functions.
*/
class ComputeUtil : protected ShaderEntity
{
  uint kernelIndices[3];

public:

  static uint create(ComputeInterface* compute, const vector<ComputeUtilTuple>& tuples, const vector<string>* includeFiles = NULL);

  static ComputeUtil* get(uint templateId);

  void calculateSum(ComputeInterface* compute, ComputeMemory* memory, uint length, bool doMean = false);

  void calculateSum(ComputeInterface* compute, ComputeMemory* memory, ComputeMemory* partitions, uint length, uint maxPartitionLength, bool doMean = false);

  void showMatrix(ComputeInterface* compute, ComputeMemory* memory, uint rowSize, uint strideIn4Byte, uint length);
};

#endif