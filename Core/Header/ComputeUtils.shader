
#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

// initialize usage macros according to input

#ifdef StructMember
#define STRUCT_MEMBER .StructMember
#else
#define STRUCT_MEMBER
#endif

#ifdef IdentityStructMember
#define IDENTITY_STRUCT_MEMBER .IdentityStructMember
#else
#define IDENTITY_STRUCT_MEMBER
#endif

#ifdef IdentityFunction
#define IDENTITY_FUNCTION(x) IdentityFunction(x)
#else
#define IDENTITY_FUNCTION(x)
#endif

#ifdef AddFunction
#define ADD_FUNCTION(x, y) AddFunction(&(x), &(y))
#else
#define ADD_FUNCTION(x, y) x += y
#endif

#ifdef CopyFunction
#define COPY_FUNCTION(x, y) CopyFunction(&(x), &(y))
#else
#define COPY_FUNCTION(x, y) x = y
#endif

#ifdef DivFunction
#define DIV_FUNCTION(x, y) DivFunction(&(x), &(y))
#else
#define DIV_FUNCTION(x, y) x /= y
#endif

#ifdef StructType

/*@kernel Sum all the array elements.*/
Kernel void sum1DKernel(
  Device StructType* array,
  const uint length,
  const uint iteration,
  const uint maxLocalIterations,
  const uint divideFlag)
{
  uint index = threadIndex();
  index = index << (1 + iteration);
  for (uint i = 0; i < maxLocalIterations; i++)
  {
    uint index2 = index + (1 << (i + iteration));
    if (index2 < length)
    {
      ADD_FUNCTION(array[index]STRUCT_MEMBER, array[index2]STRUCT_MEMBER);
    }
    index <<= 1;
    if (maxLocalIterations > 1)
    {
      globalMemBarrier();
    }
  }
  if (divideFlag && index == 0)
  {
    float divisor = length;
    DIV_FUNCTION(array[index]STRUCT_MEMBER, divisor);
  }
}

/*@kernel Sum all the elements of a flat 2d array.*/
Kernel void sumRegular2DKernel(
  Device StructType* array2D,
  const uint length,
  const uint subArrayElements,
  const uint iteration,
  uint       maxLocalIterations,
  const uint divideFlag)
{
  uint maxPower = 1;
  bool backwards = false;
  const uint originalIndex = threadLocalIndex();
  maxLocalIterations = maxLocalIterations << 1;

  const uint arraysPerGroup = ((COMPUTE_MAX_THREADS << 1) / subArrayElements);
  const uint maxIdentity = (threadGroupIndex() + 1) * arraysPerGroup;
  const uint offset = threadGroupIndex() * arraysPerGroup * subArrayElements;

  for (uint i = 0; i < maxLocalIterations; i++)
  {
    const uint width = (1 << maxPower);
    uint index1 = (originalIndex << maxPower) + offset;
    uint index2 = index1 + (width >> 1);

    if (index1 < length)
    {
      const uint identity1 = index1 / subArrayElements;

      if (identity1 < maxIdentity)
      {
        if (backwards) // add the remaining elements which are located at 2^ locations
        {
          // treat this index as second
          index2 = index1;
          // treat partition as the destination
          index1 = identity1 * subArrayElements;
        }

        if (index2 < length)
        {
          if (identity1 == (index2 / subArrayElements))
          {
            bool add = !backwards;
            if (backwards)
            {
              const uint diff = index2 - index1;
              if (diff < width && diff >= (width >> 1) && ((index1 - offset) & ((width << 1) - 1)))
              {
                add = true;
              }
            }

            if (add)
            {
              ADD_FUNCTION(array2D[index1]STRUCT_MEMBER, array2D[index2]STRUCT_MEMBER);
            }
          }
        }
      }
    }

    if (!backwards)
    {
      if (width > subArrayElements)
      {
        backwards = true;
      }
      else
      {
        maxPower++;
      }
    }
    else
    {
      maxPower--;
    }

    if (maxLocalIterations > 1)
    {
      globalMemBarrier();
    }
  }

  if (divideFlag && originalIndex < arraysPerGroup)
  {
    const float divisor = subArrayElements;
    DIV_FUNCTION(array2D[(threadGroupIndex() * arraysPerGroup + originalIndex) * subArrayElements]STRUCT_MEMBER, divisor);
  }
}

Kernel void consolidateFromPartitionsKernel(
  const Device StructType* source,
  Device StructType* destination,
  const Device PartitionInfo* partitions,
  const Device uint* partitionsCount)
{
  const uint index = threadIndex();

  if (index < partitionsCount[0])
  {
    COPY_FUNCTION(destination[index]STRUCT_MEMBER, source[partitions[index].offset]STRUCT_MEMBER);
  }
}

#ifdef IdentityStructType

Kernel void sumIrregular2DKernel(
  Device StructType* array2D,
  Device StructType* consolidatedArray,
  const Device IdentityStructType* array2DIdentity,
  const Device PartitionInfo* partitionArray,
  const Device uint* partitionCount,
  const uint length,
  const uint maxPartitionLength,
  const uint iteration,
  uint maxLocalIterations,
  const uint divideFlag)
{
  uint maxPower = 1;
  bool backwards = false;
  const int originalIndex = threadLocalIndex();
  maxLocalIterations = maxLocalIterations << 1;

  const uint perGroupPartitions = ((COMPUTE_MAX_THREADS << 1) / maxPartitionLength);
  const uint minIdentity = threadGroupIndex() * perGroupPartitions;
  const uint maxIdentity = minIdentity + perGroupPartitions;

  if (minIdentity < partitionCount[0])
  {
    const int offset = partitionArray[minIdentity].offset;

    for (uint i = 0; i < maxLocalIterations; i++)
    {
      uint width = (1 << maxPower);
      int index1 = (originalIndex << maxPower) + offset;
      int index2 = index1 + (width >> 1);

      if (index1 < length)
      {
        uint identity1 = IDENTITY_FUNCTION(array2DIdentity[index1]IDENTITY_STRUCT_MEMBER);

        if (identity1 < maxIdentity)
        {
          if (backwards) // add the remaining elements which are located at 2^ locations
          {
            // treat this index as second
            index2 = index1;
            // treat partition as the destination
            index1 = partitionArray[identity1].offset;
          }

          if (index2 < length)
          {
            if (identity1 == IDENTITY_FUNCTION(array2DIdentity[index2]IDENTITY_STRUCT_MEMBER))
            {
              bool add = !backwards;
              const int diff = index2 - index1;
              if (backwards)
              {
                if (diff < width && diff >= (width >> 1) && ((index1 - offset) & ((width << 1) - 1)))
                {
                  add = true;
                }
              }

              if (add)
              {
                ADD_FUNCTION(array2D[index1]STRUCT_MEMBER, array2D[index2]STRUCT_MEMBER);
              }

              if (maxPower == 1 && backwards && diff < width)
              {
                if (divideFlag)
                {
                  float div = partitionArray[identity1].count;
                  DIV_FUNCTION(array2D[index1]STRUCT_MEMBER, div);
                }
                if (consolidatedArray != array2D)
                {
                  COPY_FUNCTION(consolidatedArray[identity1]STRUCT_MEMBER, array2D[index1]STRUCT_MEMBER);
                }
              }
            }
          }
        }
      }

      if (!backwards)
      {
        if (width >= maxPartitionLength)
        {
          backwards = true;
        }
        else
        {
          maxPower++;
        }
      }
      else
      {
        maxPower--;
      }

      if (maxLocalIterations > 1)
      {
        barrier(CLK_GLOBAL_MEM_FENCE);
      }
    }
  }


  /*if (divideFlag)
  {
  const uint offset = originalIndex + minIdentity;
  if (offset < maxIdentity && offset < partitionCount[0])
  {
  const uint identity1 = IDENTITY_FUNCTION(array2DIdentity[offset]IDENTITY_STRUCT_MEMBER);
  const uint index1 = partitionArray[identity1].offset;
  const float div = partitionArray[identity1].count;
  DIV_FUNCTION(array2D[index1]STRUCT_MEMBER, div);

  if (consolidatedArray != array2D)
  {
  COPY_FUNCTION(consolidatedArray[identity1]STRUCT_MEMBER, array2D[index1]STRUCT_MEMBER);
  }
  }
  }*/
}


#endif

#endif

/*
@kernel Store offsets in an array from section data.
@param sectionOffsets Section offsets output.
@param entityLocation Section data.
@param length Current particle position.
*/
/*
Kernel void sectionOffsetsKernel(
Device uint* sectionOffsets,
Device uint* sectionOffsetCount,
const Device EntityLocation* entityLocation,
const uint length)
{
const uint localIndex = threadLocalIndex();
const uint multiplier = ceil(((float)length) / threadGroupSize());

Shared uint localSectionOffsets[COMPUTE_MAX_THREADS];
Shared uint localSectionCounts[COMPUTE_MAX_THREADS];
Shared uint compactOffsets[COMPUTE_MAX_THREADS + 1];

compactOffsets[COMPUTE_MAX_THREADS - 1] = 0;

for (uint multiple = 0; multiple < multiplier; multiple++)
{
const uint gridOffset = COMPUTE_MAX_THREADS * multiple;
const uint index = gridOffset + localIndex;

if (localIndex == 0)
{
compactOffsets[COMPUTE_MAX_THREADS] = compactOffsets[COMPUTE_MAX_THREADS - 1];
if (multiple > 0)
{
compactOffsets[COMPUTE_MAX_THREADS] += getInstanceId(entityLocation[gridOffset].identity);
}
}
localMemBarrier();

if (index < length)
{
localSectionOffsets[localIndex] = entityLocation[index].offsets[SECTION_DATA_NODE];
localSectionCounts[localIndex] = entityLocation[index].counts[SECTION_DATA_NODE];
const uint instanceCount = getInstanceId(entityLocation[index].identity);

compactOffsets[localIndex] = compactOffsets[COMPUTE_MAX_THREADS];
for (uint i = 0; i < localIndex; i++)
{
compactOffsets[localIndex] += getInstanceId(entityLocation[gridOffset + i].identity);
}

const uint nodeCount = localSectionCounts[localIndex] / instanceCount;

for (uint i = 0; i < instanceCount; i++)
{
sectionOffsets[compactOffsets[localIndex] + i] = localSectionOffsets[localIndex] + nodeCount * i;
}
}
localMemBarrier();
if (index == length)
{
sectionOffsetCount[0] = compactOffsets[localIndex - 1] + getInstanceId(entityLocation[index - 1].identity);
}
}
}
*/

#ifdef StructTypeIntegral

const StructType subGroupPrefixScan(volatile Shared StructType* localArray1D, const uint localIndex, const uint subGroupLocalIndex)
{
  if (subGroupLocalIndex >= 1)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 1];
  }
  if (subGroupLocalIndex >= 2)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 2];
  }
  if (subGroupLocalIndex >= 4)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 4];
  }
  if (subGroupLocalIndex >= 8)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 8];
  }
  if (subGroupLocalIndex >= 16)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 16];
  }

  return localArray1D[localIndex];
}

StructType groupPrefixScan(Shared StructType* localArray1D, const uint localIndex, const uint elements)
{
  const uint subGroupLocalIndex = localIndex & (COMPUTE_SUB_GROUP_SIZE - 1);
  const uint subGroupIndex = localIndex >> COMPUTE_SUB_GROUP_EXP;

  const StructType subGroupSum = subGroupPrefixScan(localArray1D, localIndex, subGroupLocalIndex);
  localMemBarrier();

  // copy last element from each sub group to first sub group's local space
  if (subGroupLocalIndex == (COMPUTE_SUB_GROUP_SIZE - 1))
  {
    localArray1D[subGroupIndex] = subGroupSum;
  }
  localMemBarrier();

  // prefix scan first sub group
  if (localIndex < (elements >> COMPUTE_SUB_GROUP_EXP))
  {
    const StructType prev = localArray1D[localIndex];
    subGroupPrefixScan(localArray1D, localIndex, localIndex);
    localArray1D[localIndex] -= prev;
  }
  localMemBarrier();

  // add scanned values to each return value
  return subGroupSum + localArray1D[subGroupIndex];
}

void subGroupReduce(volatile Shared StructType* localArray1D, const uint localIndex, const uint groupSize)
{
  const uint subGroupLocalIndex = localIndex & (groupSize - 1);

  //if (subGroupLocalIndex < 16)
  {
    const StructType temp = localArray1D[localIndex + 16];
    localArray1D[localIndex] += temp;// localArray1D[subGroupLocalIndex + 16];
  }

  //if (subGroupLocalIndex < 8)
  {
    const StructType temp = localArray1D[localIndex + 8];
    localArray1D[localIndex] += temp;// localArray1D[subGroupLocalIndex + 8];
  }

  //if (subGroupLocalIndex < 4)
  {
    const StructType temp = localArray1D[localIndex + 4];
    localArray1D[localIndex] += temp;// localArray1D[subGroupLocalIndex + 4];
  }

  //if (subGroupLocalIndex < 2)
  {
    const StructType temp = localArray1D[localIndex + 2];
    localArray1D[localIndex] += temp;// localArray1D[subGroupLocalIndex + 2];
  }

  //if (subGroupLocalIndex < 1)
  {
    const StructType temp = localArray1D[localIndex + 1];
    localArray1D[localIndex] += temp;// localArray1D[subGroupLocalIndex + 1];
  }
  //}
}

void groupReduce(volatile Shared StructType* localArray1D, const uint localIndex, const uint elements)
{
  /*const uint subGroupLocalIndex = localIndex & (COMPUTE_SUB_GROUP_SIZE - 1);
  const uint subGroupIndex = localIndex >> COMPUTE_SUB_GROUP_EXP;

  // per sub group prefix scan
  if (localIndex < elements)
  subGroupReduce(localArray1D, localIndex, COMPUTE_SUB_GROUP_SIZE);

  localMemBarrier();

  // copy last element from each sub group to first sub group's local space
  if (subGroupLocalIndex == 0)
  {
  //localArray1D[subGroupIndex] = localArray1D[localIndex];
  localArray1D[subGroupIndex] = (subGroupIndex < (elements >> COMPUTE_SUB_GROUP_EXP)) ? localArray1D[localIndex] : 0;
  //localArray1D[subGroupIndex] = subGroupIndex < (elements >> COMPUTE_SUB_GROUP_EXP) ? localArray1D[localIndex] : 0;
  }
  localMemBarrier();

  // prefix scan first sub group
  if (subGroupIndex == 0 && localIndex < (elements >> COMPUTE_SUB_GROUP_EXP))
  {
  subGroupReduce(localArray1D, localIndex, COMPUTE_SUB_GROUP_SIZE);
  }
  localMemBarrier();

  return;*/

  if (localIndex < 512 && elements > 512)
  {
    localArray1D[localIndex] += localArray1D[localIndex + 512];
    localMemBarrier();
  }

  if (localIndex < 256 && elements > 256)
  {
    localArray1D[localIndex] += localArray1D[localIndex + 256];
    localMemBarrier();
  }

  if (localIndex < 128 && elements > 128)
  {
    localArray1D[localIndex] += localArray1D[localIndex + 128];
  }
  localMemBarrier();

  if (localIndex < 64 && elements > 64)
  {
    localArray1D[localIndex] += localArray1D[localIndex + 64];
  }
  localMemBarrier();

  if (localIndex < 32)
  {
    localArray1D[localIndex] += localArray1D[localIndex + 32];
    localArray1D[localIndex] += localArray1D[localIndex + 16];
    localArray1D[localIndex] += localArray1D[localIndex + 8];
    localArray1D[localIndex] += localArray1D[localIndex + 4];
    localArray1D[localIndex] += localArray1D[localIndex + 2];
    localArray1D[localIndex] += localArray1D[localIndex + 1];
  }
}

/*
@kernel Parallel prefix scan all the elements within the group.
@param destination Output array.
@param sumBuffer Buffer storing output of the last group element.
@param array1D Input array.
@param length Total number of array elements.
*/
Kernel void prefixGroupScanKernel(
  Device StructType* sumBuffer,
  const Device StructType* array1D,
  const uint length)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();

  Shared StructType localArray1D[COMPUTE_MAX_THREADS];

  localArray1D[localIndex] = (index < length) ? array1D[index] : 0;

  localMemBarrier();

  groupReduce(localArray1D, localIndex, COMPUTE_MAX_THREADS);

  if (localIndex == 0)
  {
    sumBuffer[threadGroupIndex()] = localArray1D[0];
  }
}

/*
@kernel Parallel prefix scan all the elements within the group.
@param sumBuffer Output array.
@param prefixGroupCount .
@param maxPrefixGroupCount .
*/
Kernel void prefixTopScanKernel(
  Device StructType* sumBuffer,
  const uint prefixGroupCount,
  const uint maxPrefixGroupCount)
{
  Shared StructType localData[COMPUTE_MAX_THREADS];
  const uint localIndex = threadLocalIndex();
  const uint groupSize = threadGroupSize();

  for (uint i = localIndex; i < prefixGroupCount; i += groupSize)
  {
    localData[paddedIndex(i)] = (i < prefixGroupCount) ? sumBuffer[i] : 0;
    sumBuffer[i] = groupPrefixScan(localData, localIndex, COMPUTE_MAX_THREADS);
  }
}

/*
@kernel Parallel prefix scan all the elements within the group.
@param destination Output array.
@param blockSum Buffer storing output of the last group element..
@param length Total number of array elements.
*/
Kernel void prefixAddOffsetKernel(
  Device StructType* destination,
  Device StructType* blockSum,
  const uint length)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();
  const StructType localBlockSum = (threadGroupIndex() > 0) ? blockSum[threadGroupIndex() - 1] : 0;

  Shared StructType localArray1D[COMPUTE_MAX_THREADS];

  const StructType originalValue = (index < length) ? destination[index] : 0;
  localArray1D[localIndex] = originalValue;

  const StructType sum = groupPrefixScan(localArray1D, localIndex, COMPUTE_MAX_THREADS);

  if (index < length)
  {
    destination[index] = sum + localBlockSum - originalValue;
  }
}


#define PrefixPackingShift 2
#define PrefixPackingShifted 4
#define PrefixPackingBits  (32 / PrefixPackingShifted)
#define RadixPrefixScanType uint
#define RadixPrefixScanTypeSize 1

const RadixPrefixScanType subGroupPrefixScanRadix(volatile Shared RadixPrefixScanType* localArray1D, const uint localIndex, const uint subGroupLocalIndex)
{
  if (subGroupLocalIndex >= 1)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 1];
  }
  if (subGroupLocalIndex >= 2)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 2];
  }
  if (subGroupLocalIndex >= 4)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 4];
  }
  if (subGroupLocalIndex >= 8)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 8];
  }
  if (subGroupLocalIndex >= 16)
  {
    localArray1D[localIndex] += localArray1D[localIndex - 16];
  }

  return localArray1D[localIndex];
}

RadixPrefixScanType groupPrefixScanRadix(Shared RadixPrefixScanType* localArray1D, const uint localIndex, const uint elements)
{
  const uint subGroupLocalIndex = localIndex & (COMPUTE_SUB_GROUP_SIZE - 1);
  const uint subGroupIndex = localIndex >> COMPUTE_SUB_GROUP_EXP;

  const RadixPrefixScanType subGroupSum = subGroupPrefixScanRadix(localArray1D, localIndex, subGroupLocalIndex);
  localMemBarrier();

  // copy last element from each sub group to first sub group's local space
  if (subGroupLocalIndex == (COMPUTE_SUB_GROUP_SIZE - 1))
  {
    localArray1D[subGroupIndex] = subGroupSum;
  }
  localMemBarrier();

  // prefix scan first sub group
  if (localIndex < (elements >> COMPUTE_SUB_GROUP_EXP))
  {
    const RadixPrefixScanType prev = localArray1D[localIndex];
    subGroupPrefixScanRadix(localArray1D, localIndex, localIndex);
    localArray1D[localIndex] -= prev;
  }
  localMemBarrier();

  // add scanned values to each return value
  return subGroupSum + localArray1D[subGroupIndex];
}

void groupReduceRadix(volatile Shared RadixPrefixScanType* localArray1D, const uint localIndex, const uint elements)
{
  /*if (localIndex < 512 && elements > 512)
  {
  localArray1D[localIndex] += localArray1D[localIndex + 512];
  }
  localMemBarrier();

  if (localIndex < 256 && elements > 256)
  {
  localArray1D[localIndex] += localArray1D[localIndex + 256];
  }
  localMemBarrier();
  */
  /*if (localIndex < 128 && elements > 128)
  {
  localArray1D[localIndex] += localArray1D[localIndex + 128];
  }
  localMemBarrier();*/

  if (localIndex < 64 && elements > 64)
  {
    localArray1D[localIndex] += localArray1D[localIndex + 64];
  }
  localMemBarrier();

  if (localIndex < 32)
  {
    localArray1D[localIndex] += localArray1D[localIndex + 32];
    localArray1D[localIndex] += localArray1D[localIndex + 16];
    localArray1D[localIndex] += localArray1D[localIndex + 8];
    localArray1D[localIndex] += localArray1D[localIndex + 4];
    localArray1D[localIndex] += localArray1D[localIndex + 2];
    localArray1D[localIndex] += localArray1D[localIndex + 1];
  }
}

uint setLocalCount(const uint localKey, const uint keyPrefix)
{
  return 1 << (localKey << 3);
  // (localKey == keyPrefix) | ((localKey == (keyPrefix + 1)) << 8) | ((localKey == (keyPrefix + 2)) << 16) | ((localKey == (keyPrefix + 3)) << 24);
}

Kernel void radixSort32BitLocalSortKernel(
  const Device SortNode32* array1D,
  Device uint* localSumBuffer,
  const uint rightShift,
  const uint length)
{
  const uint localIndex = threadLocalIndex();
  const uint maxBlocks = (uint)ceil((float)(length) / BlockSize);
  const uint indexStride = BlockSize * threadGroupCount();

  Shared RadixPrefixScanType localCount[BlockSize];

  for (uint index = threadIndex(); index < length; index += indexStride)
  {
    const uint localKey = (((index < length) ? array1D[index] : defaultSortNode()).key >> rightShift) & (SortBitValue - 1);
    //RadixPrefixScanType first;

    //for (uint j = 0; j < RadixPrefixScanTypeSize; j++)
    //((Shared uint*)(localCount + localIndex))[j] = setLocalCount(localKey, PrefixPackingShifted * j);
    localCount[localIndex] = setLocalCount(localKey, 0);

    //if (localIndex == 0)
    {
      //first = localCount[0];
    }
    localMemBarrier();

    groupReduceRadix(localCount, localIndex, BlockSize);

    if (localIndex == 0)
    {
      const uint blockOffset = (index / BlockSize);
      //const RadixPrefixScanType privateCount = localCount[0] - first;
      const RadixPrefixScanType privateCount = localCount[0];
      for (uint i = 0; i < RadixPrefixScanTypeSize; i++)
      {
        //const uint4 count = convert_uint4(as_uchar4(((const Thread uint*)&privateCount)[i])) + convert_uint4(as_uchar4(((const Thread uint*)&first)[i]));
        const uint4 count = convert_uint4(as_uchar4(((const Thread uint*)&privateCount)[i]));
        const uint* countPtr = (const Thread uint*)&count;
        for (uint p = 0; p < PrefixPackingShifted; p++)
        {
          localSumBuffer[maxBlocks * (i * PrefixPackingShifted + p) + blockOffset] = countPtr[p];
        }
      }
    }
    localMemBarrier();
  }
}

Kernel void radixSort32BitGlobalShuffleKernel(
  Device SortNode32* destination,
  const Device SortNode32* array1D,
  const Device uint* localSumBuffer,
  const uint rightShift,
  const uint length)
{
  const uint localIndex = threadLocalIndex();
  const uint maxBlocks = (uint)ceil((float)(length) / BlockSize);
  const uint indexStride = BlockSize * threadGroupCount();

  Shared RadixPrefixScanType localCount[BlockSize];

  RadixPrefixScanType localSums;
  SortNode32 localSortNode;

  for (uint index = threadIndex(); index < length; index += indexStride)
  {
    localSortNode = (index < length) ? array1D[index] : defaultSortNode();
    const uint localKey = (localSortNode.key >> rightShift) & (SortBitValue - 1);

    //for (uint j = 0; j < RadixPrefixScanTypeSize; j++)
    //((Thread uint*)&localSums)[j] = -setLocalCount(localKey, PrefixPackingShifted * j);
    localSums = -setLocalCount(localKey, 0);

    localCount[localIndex] = -localSums;
    localSums += groupPrefixScanRadix(localCount, localIndex, BlockSize);

    if (index < length)
    {
      const uint localSum = localSumBuffer[(maxBlocks * localKey) + (index / BlockSize)];

      uint count = ((Thread uint*)&localSums)[localKey >> PrefixPackingShift];
      count >>= ((localKey & (PrefixPackingShifted - 1)) * PrefixPackingBits);
      count &= ((1 << PrefixPackingBits) - 1);

      //if ((rightShift == 0) || (index != (count + localSum)))
      //{
      destination[count + localSum] = localSortNode;
      //}
    }
    localMemBarrier();
  }
}

#endif

Kernel void bitonicSort32BitKernel(
  Device SortNode32* array1D,
  const uint multiplier,
  const uint minDepth,
  const uint maxDepth,
  const uint length)
{
  // the number of threads for the kernel
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();

#define MAX_LOCAL_NODES (COMPUTE_MAX_THREADS * 4)

  Shared SortNode32 localNode[MAX_LOCAL_NODES];

  for (uint m = 0; m < multiplier; m++)
  {
    const uint index1 = (m*MAX_LOCAL_NODES + index) << 1;
    const uint localIndex1 = (m*MAX_LOCAL_NODES + localIndex) << 1;
    //printf("_ %d _", index1);
    if (index1 < length)
    {
      localNode[localIndex1] = array1D[index1];
    }
    else
    {
      localNode[localIndex1].key = -1;
    }
    if ((index1 + 1) < length)
    {
      localNode[localIndex1 + 1] = array1D[index1 + 1];
    }
    else
    {
      localNode[localIndex1 + 1].key = -1;
    }
  }

  localMemBarrier();

  for (uint d = minDepth; d < maxDepth; d++)
  {
    for (int d2 = d; d2 >= minDepth; d2--)
    {
      const uint offset = 1 << d2;

      for (uint m = 0; m < multiplier; m++)
      {
        const uint index1 = (localIndex + m * MAX_LOCAL_NODES) << d2;
        const uint index2 = index1 + offset;

        //printf("_ %d _", index1);

        if (localNode[index1].key < localNode[index2].key)
        {
          const SortNode32 temp = localNode[index1];
          localNode[index1] = localNode[index2];
          localNode[index2] = temp;
        }
      }

      localMemBarrier();
    }
  }

  for (uint m = 0; m < multiplier; m++)
  {
    const uint index1 = (m*MAX_LOCAL_NODES + index) << 1;
    const uint localIndex1 = (m*MAX_LOCAL_NODES + localIndex) << 1;
    array1D[index1] = localNode[localIndex1];
    array1D[index1 + 1] = localNode[localIndex1 + 1];
  }
}

Kernel void showMatrix(Device float* array, const uint rowLength, const uint strideIn4Byte, const uint length)
{
  uint index = threadIndex();
  if (index < length / 3)
  {
    for (uint i = 0; i < rowLength / 3; i++)
    {
      printf("%f %f %f\n", array[index * strideIn4Byte + i * 3], array[index * strideIn4Byte + i * 3 + 1], array[index * strideIn4Byte + i * 3 + 2]);
    }
  }
}

#endif