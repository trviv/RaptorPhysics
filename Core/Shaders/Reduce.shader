#ifndef COMPUTE_UTILS_REDUCE_H
#define COMPUTE_UTILS_REDUCE_H

#if (!defined(SkipParallelPrimitives) || defined(OnlyReduce)) && defined(StructType)

#ifndef USE_SIMD_COMPUTE

void subGroupReduce(volatile Shared MemberStructType* localArray, const ushort localIndex, const ushort subGroupLocalIndex)
{
#if ComputeSimdWidth > 32
  if (subGroupLocalIndex < 32)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex + 32]);
  }
#endif

  if (subGroupLocalIndex < 16)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex + 16]);
  }

  if (subGroupLocalIndex < 8)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex + 8]);
  }

  if (subGroupLocalIndex < 4)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex + 4]);
  }

  if (subGroupLocalIndex < 2)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex + 2]);
  }

  if (subGroupLocalIndex < 1)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex + 1]);
  }
}

void groupReduce(volatile Shared MemberStructType* localArray, const ushort localIndex)
{
//  // log n iterations
//  for (uchar i=0; i<8; i++)
//  {
//    localMemBarrier();
//    const short stride = (1 << i);
//
//    // add in strides of 2, 4, 8 ...
//    short localIndex2 = localIndex * (2 << i);
//
//    // if within the bounds
//    if (localIndex2 < REDUCE_COMPUTE_THREADS)
//    {
//      // index offset 1, 2, 4 ...
//      const short otherIndex = localIndex2 + stride;
//      // add elements
//      ADD_FUNCTION(localArray[paddedIndex(localIndex2)], localArray[paddedIndex(otherIndex)]);
//    }
//    //localMemBarrier();
//  }
//
//  return;

  const ushort subGroupLocalIndex = localIndex & (ComputeSimdWidth - 1);
  const ushort subGroupIndex = localIndex >> ComputeSimdWidthExp;

  // per sub group reduce
  subGroupReduce(localArray, localIndex, subGroupLocalIndex);
  localMemBarrier();

  // copy last element from each sub group to first sub group's local space
  if (subGroupLocalIndex == 0)
  {
    COPY_FUNCTION(localArray[subGroupIndex], localArray[localIndex]);
  }
  // set non copied elements to zero
  if (subGroupIndex == 0 && subGroupLocalIndex >= (REDUCE_COMPUTE_THREADS >> ComputeSimdWidthExp))
  {
    CLEAR_FUNCTION(localArray[subGroupLocalIndex], 0);
  }
  localMemBarrier();

  // reduce first sub group
  if (subGroupIndex == 0)
  {
    subGroupReduce(localArray, localIndex, subGroupLocalIndex);
  }
}

#else

inline MemberStructType simdGroupReduce(MemberStructType reduceSum, volatile Shared MemberStructType* localArray, const ushort localIndex)
{
  const ushort subGroupLocalIndex = localIndex & (ComputeSimdWidth - 1);
  const ushort subGroupIndex = localIndex >> ComputeSimdWidthExp;

  // per sub group reduce
  REDUCE_FUNCTION(localArray[localIndex], reduceSum);

  localMemBarrier();

  if (subGroupIndex == 0)
  {
    // clear all elements to zero
    CLEAR_FUNCTION(reduceSum, 0);

    // copy last element from each sub group to first sub group's local space
    if (subGroupLocalIndex < (REDUCE_COMPUTE_THREADS >> ComputeSimdWidthExp))
    {
      COPY_FUNCTION(reduceSum, localArray[subGroupLocalIndex * ComputeSimdWidth]);
    }

    // reduce first sub group
    REDUCE_FUNCTION(reduceSum, reduceSum);
  }

  return reduceSum;
}

#endif

Kernel void reduce(
  Device StructType*                destination,
  const Device StructType*          source,
  volatile Device MemberStructType* sumBuffer,
  atomicKernelInput(uint,           statusBuffer),
  constantKernelInput(uint,         length),
  constantKernelInput(uint,         divideFlag)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint index = threadIndex();
  const ushort localIndex = threadLocalIndex();

  Shared MemberStructType localArray[REDUCE_COMPUTE_THREADS];

  MemberStructType originalValues[BatchSize];
  batchRead(originalValues, source, index, length);

  const MemberStructType reduceSum = localReduce(originalValues);

  // reduce threadgroup elements
#ifndef USE_SIMD_COMPUTE
  localArray[localIndex] = reduceSum;
  groupReduce(localArray, localIndex);
  MemberStructType sum = localArray[localIndex];
#else
  MemberStructType sum = simdGroupReduce(reduceSum, localArray, localIndex);
#endif

  // for last thread in the threadgroup
  if (localIndex == 0)
  {
    // save current value as partial sum, or final sum for the first threadgroup
    if (threadGroupIndex())
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2], sum);
      atomicStore(&statusBuffer[threadGroupIndex()], REDUCE_STATUS_PARTIAL);
    }
    else
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2 + 1], sum);
      atomicStore(&statusBuffer[threadGroupIndex()], REDUCE_STATUS_FINAL);
    }

    int prevGroupIndex = threadGroupIndex() - 1;

    INIT_POLL();
    // get reduce sum from previous threadgroups
    while (threadGroupIndex() && prevGroupIndex > -1 && !POLL_TIMEOUT())
    {
      const uint status = atomicLoad(statusBuffer + prevGroupIndex);
      MemberStructType temp;
      if (status == REDUCE_STATUS_PARTIAL)
      {
        temp = ATOMIC_LOAD_FUNCTION(&sumBuffer[prevGroupIndex * 2]);
        ADD_FUNCTION(sum, temp);
        prevGroupIndex--;
        RESET_POLL();
      }
      else if (status == REDUCE_STATUS_FINAL)
      {
        temp = ATOMIC_LOAD_FUNCTION(&sumBuffer[prevGroupIndex * 2 + 1]);
        ADD_FUNCTION(sum, temp);
        break;
      }
    }

    // save final sum for this threadgroup, if not first or very last
    if (threadGroupIndex() && threadGroupIndex() < (threadGroupCount() - 1))
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2 + 1], sum);
      atomicStore(&statusBuffer[threadGroupIndex()], REDUCE_STATUS_FINAL);
    }

    // save the reduced sum
    if (threadGroupCount() == (threadGroupIndex() + 1))
    {
      if (divideFlag)
      {
        DIV_FUNCTION(sum, length);
        COPY_FUNCTION(destination[0]STRUCT_MEMBER, sum);
      }
      else
      {
        COPY_FUNCTION(destination[0]STRUCT_MEMBER, sum);
      }
    }
  }
}

/*!@kernel Sum all the elements of a flat 2d array.*/
Kernel void reduce2DKernel(
  Device StructType*  array2D,
  constantKernelInput(uint, length),
  constantKernelInput(uint, subArrayElements),
  constantKernelInput(uint, iteration),
  constantKernelInput(uint, maxLocalIterations2),
  constantKernelInput(uint, divideFlag)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  uint maxPower = 1;
  bool backwards = false;
  const uint originalIndex = threadLocalIndex();
  uint maxLocalIterations = maxLocalIterations2 << 1;

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

#endif

#endif
