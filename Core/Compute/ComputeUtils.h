#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

#include <Header/ShaderEntity.h>
#include "DeviceArray.h"

enum ComputeUtilKey
{
  // use to specify structure name
  ComputeUtilStructType,
  // use to specify the datatype size in bytes for the structure type
  ComputeUtilStructSize,
  // use to specify if the operation has to be done on a member of the structure type
  ComputeUtilStructMember,
  // use to specify the datatype size in bytes for the member of the structure type
  ComputeUtilStructMemberSize,
  // use to specify the datatype of the member of structure
  ComputeUtilStructMemberType,
  // use to specify if the structure type is integral
  ComputeUtilStructTypeIntegral,

  ComputeUtilIdentityStructType,
  ComputeUtilIdentityStructMember,
  ComputeUtilIdentityFunction,

  ComputeUtilCustomAddFunction,
  ComputeUtilCustomCopyFunction,
  ComputeUtilCustomDivFunction,
  ComputeUtilCustomClearFunction,
  ComputeUtilCustomReduceFunction,
  ComputeUtilCustomScanFunction,

  ComputeUtilBatchSize,
  ComputeUtilSkipParallelPrimitives,
  ComputeUtilOnlyReduce,
  ComputeUtilOnlyCompaction,
  ComputeUtilMaxWorkgroupSize,

  ComputeUtilMaxKey
};

/*!
@class Class representing utility functions.
*/
class ComputeUtil : protected ShaderEntity
{
  uint kernelIndices[15];
  vector<void*> localArrays;
  uint batchSize;
  uint maxWorkgroupSize;
  uint structSize;
  uint structMemberSize;

public:

  static uint create(ComputeInterface* compute, map<ComputeUtilKey, string>& dataMap, const vector<string>* includeFiles = NULL);

  static ComputeUtil* get(uint templateId);

  static uint getUIntUtil(ComputeInterface* compute);

  static uint getUInt4Util(ComputeInterface* compute);

  static uint getXABUtil(ComputeInterface* compute);

  void configureWorkgroupCount(ComputeInterface* compute, ComputeMemory* workgroupCount, const ComputeMemory* threadCount, const size_t workgroupSize[3]);

  void configureWorkgroupCount(ComputeInterface* compute, ComputeMemory* workgroupCount, const size_t threadCount[3], const size_t workgroupSize[3]);

  void sum1D(ComputeInterface* compute, ComputeMemory* source, uint length, bool doMean = false);

  void sum1D(ComputeInterface* compute, ComputeMemory* destination, const ComputeMemory* source, uint length, bool doMean = false);

  void sumRegular2D(ComputeInterface* compute, ComputeMemory* array2D, uint length, uint subArrayElements, bool doMean = false);

  void sumIrregular2D(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, ComputeMemory* identity, ComputeMemory* partitions, uint length, bool doMean = false);

  void compactSparseArray(ComputeInterface* compute, ComputeMemory* compactLength, ComputeMemory* compactIndices, ComputeMemory* sparseArray, uint sparseLength);

  void compactSparseArrayAndCopy(ComputeInterface* compute, ComputeMemory* compactLength, ComputeMemory* compactArray, ComputeMemory* sparseArray, uint sparseLength);

  void compactSparseArrayAndCopy(ComputeInterface* compute, ComputeMemory* compactLength, ComputeMemory* compactArray, ComputeMemory* sparseArray, const ComputeMemory* sparseLength, uint maxSparseLength);

  void consolidateFromPartitions(ComputeInterface* compute, ComputeMemory* source, ComputeMemory* destination, ComputeMemory* partitions, ComputeMemory* partitionsCount, uint partitionsCountHost);

  void prefixScan1D(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, uint length);

  void bitonicSort32Bit(ComputeInterface* compute, ComputeMemory* array1D, uint length);

  void radixSort32Bit(ComputeInterface* compute, ComputeMemory* destination, ComputeMemory* source, uint length);

  void showMatrix(ComputeInterface* compute, ComputeMemory* memory, uint rowSize, uint strideIn4Byte, uint length, bool showOnlyFaults = false);

  void clearBuffer(ComputeInterface* compute, ComputeMemory* destination, uint length, uint value = 0);

  void copyBuffer(ComputeInterface* compute, const ComputeMemory* source, ComputeMemory* destination, uint sourceOffset, uint destinationOffset, uint sizeInBytes);
};

#endif
