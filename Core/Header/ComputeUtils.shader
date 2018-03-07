#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

// initialize usage macros according to input 

#ifdef StructMember
#define STRUCT_MEMBER .StructMember
#else
#define STRUCT_MEMBER
#endif

#ifdef StructIdentity
#define STRUCT_IDENTITY .StructIdentity
#else
#define STRUCT_IDENTITY
#endif

#ifdef IndexStructMember
#define INDEX_STRUCT_MEMBER .IndexStructMember
#else
#define INDEX_STRUCT_MEMBER
#endif

#ifdef AddFunction
#define ADD_FUNCTION(x, y) AddFunction(&(x), &(y))
#else
#define ADD_FUNCTION(x, y) x += y
#endif

#ifdef DivFunction
#define DIV_FUNCTION(x, y) DivFunction(&(x), &(y))
#else
#define DIV_FUNCTION(x, y) x /= y
#endif

// requires ReductionFunction
// GroupSize
/*@kernel Sum all the array elements.*/
#define GroupSize 64
#define ReductionFunction ADD_FUNCTION

Kernel void sum1DKernel(Device StructType* array,
  uint length,
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
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
  if (divideFlag && index == 0)
  {
    float divisor = length;
    DIV_FUNCTION(array[index]STRUCT_MEMBER, divisor);
  }
}

/*
Kernel void sum1DKernel2(
Device StructType* outputData,
Device StructType* inputData,
const uint length,
const uint divideFlag,
Shared StructType* groupData,
const uint GroupSize)
{
uint localIndex = threadLocalIndex();
uint globalIndex = localIndex + (groupIndex() * GroupSize << 1);
uint groupStride = groupSize() * (GroupSize << 1);

groupData[localIndex] = 0;
while (globalIndex < length)
{
ReductionFunction(groupData[localIndex], inputData[globalIndex]);
ReductionFunction(groupData[localIndex], inputData[globalIndex + GroupSize]);
globalIndex += groupStride;
}

#if (GroupSize >= 512)
localMemBarrier();
if (localIndex < 256)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 256]);
}
#endif

#if (GroupSize >= 256)
localMemBarrier();
if (localIndex < 128)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 128]);
}
#endif

#if (GroupSize >= 128)
localMemBarrier();
if (localIndex < 64)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 64]);
}
#endif

#if (GroupSize >= 64)
localMemBarrier();
if (localIndex < 32)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 32]);
}
#endif

#if (GroupSize >= 32)
localMemBarrier();
if (localIndex < 16)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 16]);
}
#endif

#if (GroupSize >= 16)
localMemBarrier();
if (localIndex < 8)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 8]);
}
#endif

#if (GroupSize >= 8)
localMemBarrier();
if (localIndex < 4)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 4]);
}
#endif

#if (GroupSize >= 4)
localMemBarrier();
if (localIndex < 2)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 2]);
}
#endif

#if (GroupSize >= 2)
localMemBarrier();
if (localIndex < 1)
{
ReductionFunction(groupData[localIndex], groupData[localIndex + 1]);
}
#endif

localMemBarrier();
if (localIndex == 0)
{
outputData[groupIndex()] = groupData[0];
}

if (divideFlag && globalIndex == 0)
{
float divisor = length;
//DIV_FUNCTION(array[index]STRUCT_MEMBER, divisor);
}
}*/

//#define DEBUG_SUM_PARTITION

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
  const uint maxIdentity = (groupIndex() + 1) * arraysPerGroup;
  const uint offset = groupIndex() * arraysPerGroup * subArrayElements;

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
    float divisor = subArrayElements;
    DIV_FUNCTION(array2D[(groupIndex() * arraysPerGroup + originalIndex) * subArrayElements]STRUCT_MEMBER, divisor);
  }
}

#ifdef IndexStructType

Kernel void sumIrregular2DKernel(Device StructType* array, const Device IndexStructType* partitionArray,
  const uint length, const uint maxPartitionLength, const uint iteration, uint maxLocalIterations, const uint divideFlag)
{
  uint maxPower = 1;
  bool backwards = false;
  const int originalIndex = threadLocalIndex();
  maxLocalIterations = maxLocalIterations << 1;

  const int maxIdentity = (groupIndex() + 1) * ((COMPUTE_MAX_THREADS << 1) / maxPartitionLength);
  const int offset = partitionArray[groupIndex()  * ((COMPUTE_MAX_THREADS << 1) / maxPartitionLength)]INDEX_STRUCT_MEMBER;

  for (uint i = 0; i < maxLocalIterations; i++)
  {
    uint width = (1 << maxPower);
    int index1 = (originalIndex << maxPower) + offset;
    int index2 = index1 + (width >> 1);

    if (index1 < length)
    {
      uint identity1 = array[index1]STRUCT_IDENTITY;

      if (identity1 < maxIdentity)
      {
        if (backwards) // add the remaining elements which are located at 2^ locations
        {
          // treat this index as second
          index2 = index1;
          // treat partition as the destination
          index1 = partitionArray[identity1]INDEX_STRUCT_MEMBER;
        }

        if (index2 < length)
        {

#ifdef DEBUG_SUM_PARTITION
          if (identity1 == 32)
          {
            printf("1:     %d %d %d %d\n", index1, index2, width, array[index2]STRUCT_IDENTITY);
          }
#endif

          if (identity1 == array[index2]STRUCT_IDENTITY)
          {
            bool add = !backwards;
            if (backwards)
            {
              int diff = index2 - index1;
              if (diff < width && diff >= (width >> 1) && ((index1 - offset) & ((width << 1) - 1)))
              {
                add = true;
              }
            }

            if (add)
            {

#ifdef DEBUG_SUM_PARTITION
              if (identity1 == 32)
              {
                printf("2:     %d %d %f\n", index1, index2, array[index1]STRUCT_MEMBER.x);
              }
#endif

              ADD_FUNCTION(array[index1]STRUCT_MEMBER, array[index2]STRUCT_MEMBER);
            }
          }
        }
      }
    }

    if (!backwards)
    {
      if (width > maxPartitionLength)
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
  //if (divideFlag && index == 0)
  {
    //array[index]STRUCT_MEMBER /= length;
  }
}

#endif

/*@kernel Sum all the array elements.*/
Kernel void parallelPrefixSum1D(
  Device StructType* array,
  const uint length,
  const uint iteration,
  const uint maxLocalIterations)
{
  /*
  Group StructType localArray[2 * COMPUTE_MAX_THREADS];
  */

  const uint originalIndex = threadIndex();

  for (uint i = 0; i < maxLocalIterations; i++)
  {
    const uint currentIteration = iteration + i;
    const uint stride = (1 << currentIteration);
    //uint index1 = originalIndex << (1 + currentIteration);
    uint index1 = ((1 + originalIndex) << (((currentIteration & 1) ^ 1) + currentIteration)) - 1;
    uint index2 = index1 - stride;

    //printf("1: %d %d %d\n", originalIndex, index1, index2);
    if (index1 < length && ((currentIteration & 1) == 0 || (originalIndex & 1)))
    {
      printf("2: %d %d %d\n", originalIndex, index1, index2);
      ADD_FUNCTION(array[index1]STRUCT_MEMBER, array[index2]STRUCT_MEMBER);
    }

    if (currentIteration)
    {
      //printf("3: %d %d\n", index1, index2);
      index1 = ((originalIndex) << ((((currentIteration - 1) & 1) ^ 1) + currentIteration - 1)) + 1;
      index2 = index1 + (stride >> 1);
      if (index2 < length)
      {
        printf("4: %d %d %d\n", originalIndex, index2, index1);
        ADD_FUNCTION(array[index2]STRUCT_MEMBER, array[index1]STRUCT_MEMBER);
      }
    }

    if (maxLocalIterations > 1)
    {
      globalMemBarrier();
    }
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