#ifndef COMPUTE_UTILS_PREFIX_SCAN_H
#define COMPUTE_UTILS_PREFIX_SCAN_H

#if !defined(SkipParallelPrimitives) && defined(StructType)

#ifndef USE_SIMD_COMPUTE

const MemberStructType subGroupPrefixScan(volatile Shared MemberStructType* localArray, const ushort localIndex, const ushort subGroupLocalIndex)
{
  if (subGroupLocalIndex >= 1)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex - 1]);
  }
  if (subGroupLocalIndex >= 2)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex - 2]);
  }
  if (subGroupLocalIndex >= 4)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex - 4]);
  }
  if (subGroupLocalIndex >= 8)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex - 8]);
  }
  if (subGroupLocalIndex >= 16)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex - 16]);
  }
#if COMPUTE_SUB_GROUP_SIZE > 32
  if (subGroupLocalIndex >= 32)
  {
    ADD_FUNCTION(localArray[localIndex], localArray[localIndex - 32]);
  }
#endif

  return localArray[localIndex];
}

MemberStructType groupPrefixScan(Shared MemberStructType* localArray, const ushort localIndex, const ushort elements)
{
  const ushort subGroupLocalIndex = localIndex & (COMPUTE_SUB_GROUP_SIZE - 1);
  const ushort subGroupIndex = localIndex >> COMPUTE_SUB_GROUP_EXP;

  const MemberStructType subGroupSum = subGroupPrefixScan(localArray, localIndex, subGroupLocalIndex);
  localMemBarrier();

  // copy last element from each sub group to first sub group's local space
  if (subGroupLocalIndex == (COMPUTE_SUB_GROUP_SIZE - 1))
  {
    localArray[subGroupIndex] = subGroupSum;
  }
  localMemBarrier();

  // prefix scan first sub group
  if (localIndex < (elements >> COMPUTE_SUB_GROUP_EXP))
  {
    const MemberStructType prev = localArray[localIndex];
    subGroupPrefixScan(localArray, localIndex, localIndex);
    localArray[localIndex] -= prev;
  }
  localMemBarrier();

  // add scanned values to each return value
  return subGroupSum + localArray[subGroupIndex];
}

#else

inline MemberStructType simdGroupPrefixScan(MemberStructType reduceSum, volatile Shared MemberStructType* localArray, const ushort localIndex)
{
  const ushort subGroupLocalIndex = localIndex & (COMPUTE_SUB_GROUP_SIZE - 1);
  const ushort subGroupIndex = localIndex >> COMPUTE_SUB_GROUP_EXP;

  // per sub group reduce
  SCAN_FUNCTION(reduceSum, reduceSum);
  localArray[localIndex] = reduceSum;

  localMemBarrier();

  if (subGroupIndex == 0)
  {
    MemberStructType prefixSum;

    // clear all elements to zero
    CLEAR_FUNCTION(prefixSum, 0);

    // copy last element from each sub group to first sub group's local space
    if (subGroupLocalIndex < (PREFIX_SCAN_COMPUTE_THREADS >> COMPUTE_SUB_GROUP_EXP))
    {
      COPY_FUNCTION(prefixSum, localArray[(subGroupLocalIndex+1) * COMPUTE_SUB_GROUP_SIZE - 1]);
    }

    const MemberStructType prev = prefixSum;

    // reduce first sub group
    SCAN_FUNCTION(prefixSum, prefixSum);
    prefixSum -= prev;

    if (subGroupLocalIndex < (PREFIX_SCAN_COMPUTE_THREADS >> COMPUTE_SUB_GROUP_EXP))
    {
      COPY_FUNCTION(localArray[subGroupLocalIndex * COMPUTE_SUB_GROUP_SIZE], prefixSum);
    }
  }

  localMemBarrier();

  return reduceSum + localArray[subGroupIndex * COMPUTE_SUB_GROUP_SIZE];
}

#endif

void localExclusiveScan(Thread MemberStructType *elements, MemberStructType prev)
{
  for (uint i = 0; i < BatchSize; i++)
  {
    MemberStructType temp = elements[i];
    elements[i] = prev;
    prev += temp;
  }
}


/*
@kernel Parallel prefix scan array of elements.
@param destination Output array.
@param source Input array.
@param sumBuffer Buffer storing output of elements in a threadgroup.
@param statusBuffer Buffer storing status of elements in a threadgroup.
@param length Total number of array elements.
*/
Kernel void prefixGroupScanKernel(
  Device StructType*                destination,
  const Device StructType*          source,
  volatile Device MemberStructType* sumBuffer,
  atomicKernelInput(uint,           statusBuffer),
  constantKernelInput(uint,         length)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint index = threadIndex();
  const ushort localIndex = threadLocalIndex();

  Shared MemberStructType localArray1D[PREFIX_SCAN_COMPUTE_THREADS];

  // read the values
  MemberStructType originalValues[BatchSize];
  batchRead(originalValues, source, index, length);

  const MemberStructType reduceSum = localReduce(originalValues);

#ifndef USE_SIMD_COMPUTE
  localArray1D[localIndex] = reduceSum;
  // calculate prefix sum for the threadgroup
  MemberStructType prefixSum = groupPrefixScan(localArray1D, localIndex, PREFIX_SCAN_COMPUTE_THREADS);
  localMemBarrier();
#else
  MemberStructType prefixSum = simdGroupPrefixScan(reduceSum, localArray1D, localIndex);
#endif

  // for last thread in the threadgroup
  if (localIndex == (PREFIX_SCAN_COMPUTE_THREADS - 1))
  {
    MemberStructType lastSum = prefixSum;
    MemberStructType previousSum = 0;

    // save current value as partial sum, or final sum for the first threadgroup
    if (threadGroupIndex())
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2], lastSum);
      atomicStore(&statusBuffer[threadGroupIndex()], PREFIX_SCAN_STATUS_PARTIAL);
    }
    else
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2 + 1], lastSum);
      atomicStore(&statusBuffer[threadGroupIndex()], PREFIX_SCAN_STATUS_FINAL);
    }

    int prevGroupIndex = threadGroupIndex() - 1;
    INIT_POLL();
    // get prefix sum from previous threadgroups
    while (threadGroupIndex() && prevGroupIndex > -1 && !POLL_TIMEOUT())
    {
      const uint status = atomicLoad(&statusBuffer[prevGroupIndex]);
      if (status == PREFIX_SCAN_STATUS_PARTIAL)
      {
        ADD_FUNCTION(previousSum, ATOMIC_LOAD_FUNCTION(&sumBuffer[prevGroupIndex * 2]));
        prevGroupIndex--;
        RESET_POLL();
      }
      else if (status == PREFIX_SCAN_STATUS_FINAL)
      {
        ADD_FUNCTION(previousSum, ATOMIC_LOAD_FUNCTION(&sumBuffer[prevGroupIndex * 2 + 1]));
        break;
      }
    }

    // save final sum for this threadgroup, if not first or very last
    if (threadGroupIndex() && threadGroupIndex() < (threadGroupCount() - 1))
    {
      ADD_FUNCTION(lastSum, previousSum);
      writeAndWait(&sumBuffer[threadGroupIndex() * 2 + 1], lastSum);
      atomicStore(&statusBuffer[threadGroupIndex()], PREFIX_SCAN_STATUS_FINAL);
    }

    localArray1D[0] = previousSum;
  }

  localMemBarrier();

  ADD_FUNCTION(prefixSum, -reduceSum);
  ADD_FUNCTION(prefixSum, localArray1D[0]);

  localExclusiveScan(originalValues, prefixSum);

  batchWrite(originalValues, destination, index, length);
}

#ifdef StructTypeIntegral

Kernel void compactSparseArray(
  Device uint*                      compactArrayCount,
  Device uint*                      compactIndexArray,
  const Device StructType*          selectionArray,
  volatile Device MemberStructType* sumBuffer,
  atomicKernelInput(uint,           statusBuffer),
  constantKernelInput(uint,         length)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  const uint index = threadIndex();
  const ushort localIndex = threadLocalIndex();

  Shared MemberStructType localArray1D[PREFIX_SCAN_COMPUTE_THREADS];

  // read the values
  MemberStructType originalValues[BatchSize];
  uint statusFlag = 0;
  batchRead(originalValues, selectionArray, index, length);

  // make values binary
  for (uint i = 0; i < BatchSize; i++)
  {
    originalValues[i] = originalValues[i] > 0;
    statusFlag <<= 1;
    statusFlag |= originalValues[i];
  }

  const MemberStructType reduceSum = localReduce(originalValues);

#ifndef USE_SIMD_COMPUTE
  localArray1D[localIndex] = reduceSum;
  // calculate prefix sum for the threadgroup
  MemberStructType prefixSum = groupPrefixScan(localArray1D, localIndex, PREFIX_SCAN_COMPUTE_THREADS);
  localMemBarrier();
#else
  MemberStructType prefixSum = simdGroupPrefixScan(reduceSum, localArray1D, localIndex);
#endif

  // for last thread in the threadgroup
  if (localIndex == (PREFIX_SCAN_COMPUTE_THREADS - 1))
  {
    localArray1D[0] = 0;

    // save current value as partial sum, or final sum for the first threadgroup
    if (threadGroupIndex())
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2], prefixSum);
      atomicStore(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_PARTIAL);
    }
    else
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2 + 1], prefixSum);
      atomicStore(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_FINAL);
    }

    int prevGroupIndex = threadGroupIndex() - 1;
    INIT_POLL();
    // get prefix sum from previous threadgroups
    while (threadGroupIndex() && prevGroupIndex > -1 && !POLL_TIMEOUT())
    {
      const uint status = atomicLoad(statusBuffer + prevGroupIndex);
      if (status == PREFIX_SCAN_STATUS_PARTIAL)
      {
        ADD_FUNCTION(localArray1D[0], ATOMIC_LOAD_FUNCTION(&sumBuffer[prevGroupIndex * 2]));
        prevGroupIndex--;
        RESET_POLL();
      }
      else if (status == PREFIX_SCAN_STATUS_FINAL)
      {
        ADD_FUNCTION(localArray1D[0], ATOMIC_LOAD_FUNCTION(&sumBuffer[prevGroupIndex * 2 + 1]));
        break;
      }
    }

    // save final sum for this threadgroup, if not first or very last
    if (threadGroupIndex() && threadGroupIndex() < (threadGroupCount() - 1))
    {
      writeAndWait(&sumBuffer[threadGroupIndex() * 2 + 1], localArray1D[0] + prefixSum);
      atomicStore(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_FINAL);
    }

    // save the sum from last threadgroup to the output array
    if (threadGroupIndex() == (threadGroupCount() - 1))
    {
      compactArrayCount[0] = localArray1D[0] + prefixSum;
    }
  }

  localMemBarrier();

  ADD_FUNCTION(prefixSum, -reduceSum);
  ADD_FUNCTION(prefixSum, localArray1D[0]);

  localExclusiveScan(originalValues, prefixSum);

  const uint indexOffset = index * BatchSize;
  const uint writeCount = min(select((uint)0, length - indexOffset, length > indexOffset), (uint)BatchSize);

  uint mask = (1 << (BatchSize - 1));
  for (uint i = 0; i < writeCount; i++, mask >>= 1)
  {
    if (statusFlag & mask)
    {
      compactIndexArray[originalValues[i]] = indexOffset + i;
    }
  }
}

#endif

#endif

#endif
