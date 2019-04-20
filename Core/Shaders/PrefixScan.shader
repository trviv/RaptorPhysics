#ifndef COMPUTE_UTILS_PREFIX_SCAN_H
#define COMPUTE_UTILS_PREFIX_SCAN_H

#if !defined(SkipParallelPrimitives) && defined(StructType)

const MemberStructType subGroupPrefixScan(volatile Shared MemberStructType* localArray, const uint localIndex, const uint subGroupLocalIndex)
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

  return localArray[localIndex];
}

MemberStructType groupPrefixScan(Shared MemberStructType* localArray, const uint localIndex, const uint elements)
{
  const uint subGroupLocalIndex = localIndex & (COMPUTE_SUB_GROUP_SIZE - 1);
  const uint subGroupIndex = localIndex >> COMPUTE_SUB_GROUP_EXP;

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
  volatile Device uint*             statusBuffer,
  const uint                        length)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();

  Shared MemberStructType localArray1D[PREFIX_SCAN_COMPUTE_THREADS];

  // read the values
  MemberStructType originalValues[BatchSize];
  batchRead(originalValues, source, index, length);

  const MemberStructType reduceSum = localReduce(originalValues);
  localArray1D[localIndex] = reduceSum;

  // calculate prefix sum for the threadgroup
  MemberStructType prefixSum = groupPrefixScan(localArray1D, localIndex, PREFIX_SCAN_COMPUTE_THREADS);

  // for last thread in the threadgroup
  if (localIndex == (PREFIX_SCAN_COMPUTE_THREADS - 1))
  {
    localArray1D[0] = 0;

    // save current value as partial sum, or final sum for the first threadgroup
    if (threadGroupIndex())
    {
      sumBuffer[threadGroupIndex() * 2] = prefixSum;
      atomicSave(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_PARTIAL);
    }
    else
    {
      sumBuffer[threadGroupIndex() * 2 + 1] = prefixSum;
      atomicSave(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_FINAL);
    }

    int prevGroupIndex = threadGroupIndex() - 1;
    // get prefix sum from previous threadgroups
    while (prevGroupIndex > -1)
    {
      const uint status = atomicLoad(statusBuffer + prevGroupIndex);
      if (status == PREFIX_SCAN_STATUS_PARTIAL)
      {
        ADD_FUNCTION(localArray1D[0], sumBuffer[prevGroupIndex * 2]);
        prevGroupIndex--;
      }
      else if (status == PREFIX_SCAN_STATUS_FINAL)
      {
        ADD_FUNCTION(localArray1D[0], sumBuffer[prevGroupIndex * 2 + 1]);
        break;
      }
    }

    // save final sum for this threadgroup
    if (threadGroupIndex())
    {
      sumBuffer[threadGroupIndex() * 2 + 1] = localArray1D[0] + prefixSum;
      atomicSave(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_FINAL);
    }
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
  volatile Device uint*             statusBuffer,
  const uint                        length)
{
  const uint index = threadIndex();
  const uint localIndex = threadLocalIndex();

  Shared MemberStructType localArray1D[PREFIX_SCAN_COMPUTE_THREADS];

  // read the values
  MemberStructType originalValues[BatchSize];
  uchar statusFlag[BatchSize];
  batchRead(originalValues, selectionArray, index, length);

  // make values binary
  for (uint i = 0; i < BatchSize; i++)
  {
    originalValues[i] = select((MemberStructType)(0), (MemberStructType)(1), originalValues[i] > (MemberStructType)(0));
    statusFlag[i] = originalValues[i];
  }

  const MemberStructType reduceSum = localReduce(originalValues);
  localArray1D[localIndex] = reduceSum;

  // calculate prefix sum for the threadgroup
  MemberStructType prefixSum = groupPrefixScan(localArray1D, localIndex, PREFIX_SCAN_COMPUTE_THREADS);

  // for last thread in the threadgroup
  if (localIndex == (PREFIX_SCAN_COMPUTE_THREADS - 1))
  {
    localArray1D[0] = 0;

    // save current value as partial sum, or final sum for the first threadgroup
    if (threadGroupIndex())
    {
      sumBuffer[threadGroupIndex() * 2] = prefixSum;
      atomicSave(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_PARTIAL);
    }
    else
    {
      sumBuffer[threadGroupIndex() * 2 + 1] = prefixSum;
      atomicSave(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_FINAL);
    }

    int prevGroupIndex = threadGroupIndex() - 1;
    // get prefix sum from previous threadgroups
    while (prevGroupIndex > -1)
    {
      const uint status = atomicLoad(statusBuffer + prevGroupIndex);
      if (status == PREFIX_SCAN_STATUS_PARTIAL)
      {
        ADD_FUNCTION(localArray1D[0], sumBuffer[prevGroupIndex * 2]);
        prevGroupIndex--;
      }
      else if (status == PREFIX_SCAN_STATUS_FINAL)
      {
        ADD_FUNCTION(localArray1D[0], sumBuffer[prevGroupIndex * 2 + 1]);
        break;
      }
    }

    // save final sum for this threadgroup
    if (threadGroupIndex())
    {
      sumBuffer[threadGroupIndex() * 2 + 1] = localArray1D[0] + prefixSum;
      atomicSave(statusBuffer + threadGroupIndex(), PREFIX_SCAN_STATUS_FINAL);
    }
  }

  if (index * BatchSize == (length - 1))
  {
    compactArrayCount[0] = localArray1D[0] + prefixSum;
  }

  localMemBarrier();

  ADD_FUNCTION(prefixSum, -reduceSum);
  ADD_FUNCTION(prefixSum, localArray1D[0]);

  localExclusiveScan(originalValues, prefixSum);

  const uint indexOffset = index * BatchSize;
  const uint writeCount = min((length > indexOffset) ? length - indexOffset : 0, (uint)BatchSize);

  for (uint i = 0; i < writeCount; i++)
  {
    if (statusFlag[i])
    {
      compactIndexArray[originalValues[i]] = index * BatchSize;
    }
  }
}

#endif

#endif

#endif