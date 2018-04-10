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

/*@kernel Sum all the array elements.*/
/*
Kernel void parallelPrefixSum1D(
Device StructType* array,
const uint length,
const uint iteration,
const uint maxLocalIterations)
{
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
*/

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

uint scanExclusive(Shared uint* localArray1D, const uint elements, const int localIndex, const uint groupSize)
{
  uint blockSum;
  uint offset = 1;

  // build sum in place up the tree
  for (uint eIndex = elements >> 1; eIndex > 0; eIndex >>= 1, offset <<= 1)
  {
    localMemBarrier();
    for (uint i = localIndex; i < eIndex; i += groupSize)
    {
      const uint index1 = (offset * ((i << 1) + 1)) - 1;

      localArray1D[paddedIndex(index1 + offset)] += localArray1D[paddedIndex(index1)];
    }
  }

  localMemBarrier();
  if (localIndex == 0)
  {
    const uint lastElement = paddedIndex(elements - 1);
    blockSum = localArray1D[lastElement];
    localArray1D[lastElement] = 0;
  }

  localMemBarrier();
  offset >>= 1;

  for (uint eIndex = 1; eIndex < elements; eIndex <<= 1, offset >>= 1)
  {
    localMemBarrier();
    for (uint i = localIndex; i < eIndex; i += groupSize)
    {
      uint index1 = (offset * ((i << 1) + 1)) - 1;
      uint index2 = index1 + offset;
      index1 = paddedIndex(index1);
      index2 = paddedIndex(index2);

      uint temp = localArray1D[index1];

      localArray1D[index1] = localArray1D[index2];
      localArray1D[index2] += temp;
    }
  }

  localMemBarrier();
  return blockSum;
}

/*
@kernel Parallel prefix scan all the elements within the group.
@param destination Output array.
@param sumBuffer Buffer storing output of the last group element.
@param array1D Input array.
@param length Total number of array elements.
*/
Kernel void prefixGroupScanKernel(
  Device uint* destination,
  Device uint* sumBuffer,
  const Device uint* array1D,
  const uint length)
{
  const uint index = threadIndex() << 1;
  const uint localIndex = threadLocalIndex() << 1;
  const uint localIndex1 = paddedIndex(localIndex);
  const uint localIndex2 = paddedIndex(localIndex + 1);

  const uint groupSize = threadGroupSize();
  const uint groupIndex = threadGroupIndex();

  Shared uint localArray1D[COMPUTE_MAX_THREADS << 1];

  localArray1D[localIndex1] = (index < length) ? array1D[index] : 0;
  localArray1D[localIndex2] = ((index + 1) < length) ? array1D[index + 1] : 0;

  const uint sum = scanExclusive(localArray1D, COMPUTE_MAX_THREADS << 1, localIndex >> 1, groupSize);

  if (localIndex == 0)
  {
    sumBuffer[groupIndex] = sum;
  }
  if (index < length)
  {
    destination[index] = localArray1D[localIndex1];
  }
  if ((index + 1) < length)
  {
    destination[index + 1] = localArray1D[localIndex2];
  }
}

/*
@kernel Parallel prefix scan all the elements within the group.
@param sumBuffer Output array.
@param prefixGroupCount .
@param maxPrefixGroupCount .
*/
Kernel void prefixTopScanKernel(
  Device uint* sumBuffer,
  const uint prefixGroupCount,
  const uint maxPrefixGroupCount)
{
  Shared uint localData[8 * 1024];
  uint localIndex = threadLocalIndex();
  const uint groupSize = threadGroupSize();

  for (uint i = localIndex; i < maxPrefixGroupCount; i += prefixGroupCount)
  {
    localData[paddedIndex(i)] = (i < prefixGroupCount) ? sumBuffer[i] : 0;
  }
  localMemBarrier();

  const uint sum = scanExclusive(localData, maxPrefixGroupCount, localIndex, groupSize);
  for (uint i = localIndex; i < prefixGroupCount; i += groupSize)
  {
    sumBuffer[i] = localData[paddedIndex(i)];
  }
  localIndex += groupSize;

  const uint index = threadIndex();
  if (index == 0)
  {
    sumBuffer[prefixGroupCount] = sum;
  }
}

/*
@kernel Parallel prefix scan all the elements within the group.
@param destination Output array.
@param blockSum Buffer storing output of the last group element..
@param length Total number of array elements.
*/
Kernel void prefixAddOffsetKernel(
  Device uint* destination,
  Device uint* blockSum,
  const uint length)
{
  const uint index = threadIndex() + (threadGroupSize() << 1);
  const uint groupIndex = 2 + threadGroupIndex();
  const uint localBlockSum = blockSum[(groupIndex >> 1)];

  if (index < length)
  {
    destination[index] += localBlockSum;
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