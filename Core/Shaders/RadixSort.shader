/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef COMPUTE_UTILS_RADIX_SORT_H
#define COMPUTE_UTILS_RADIX_SORT_H

#if !defined(SkipParallelPrimitives) && defined(StructTypeIntegral)

#define BusAlignedFetch
#define UsePackedKeys
//#define CoalescedWrites

#if defined(BusAlignedFetch) && defined(CoalescedWrites)
#undef BankConflictShift
#define BankConflictShift   RadixPrefixScanPackingExp
#endif

#define PackedParts         4
#define PackedBits          (32 / PackedParts)
#define RadixScanIterations (1 << (SortBits-2))
#define RadixPackedType     uint
#define RadixBlockInstances (256>>ComputeSimdWidthExp)
#define SortBitValue        (1<<SortBits)

#if RadixPrefixScanPackingExp > RadixReductionPackingExp
#define FetchAlignmentExp (1<<RadixPrefixScanPackingExp)
#else
#define FetchAlignmentExp (1<<RadixReductionPackingExp)
#endif

#ifndef USE_SIMD_COMPUTE
// Per-lane Hillis-Steele inclusive scan, barrier-split read-then-write so
// non-Apple OpenCL compilers (NVIDIA, RustICL) can't reorder cross-lane
// reads/writes. Every thread in the workgroup calls these helpers and
// reaches every barrier.
const RadixPackedType lanePrefixScanRadix(volatile Shared RadixPackedType* localArray1D, const ushort localIndex)
{
  volatile Shared RadixPackedType *localArray1DPtr = localArray1D + localIndex;

  RadixPackedType v;
  v = *(localArray1DPtr - 1);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr - 2);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr - 4);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr - 8);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr - 16); localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
#if ComputeSimdWidth > 32
  v = *(localArray1DPtr - 32); localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
#endif

  return localArray1D[localIndex];
}

void laneReduceRadix(volatile Shared StructType* localArray1D, const ushort localIndex)
{
  volatile Shared RadixPackedType *localArray1DPtr = localArray1D + localIndex;

  RadixPackedType v;
#if ComputeSimdWidth > 32
  v = *(localArray1DPtr + 32); localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
#endif
  v = *(localArray1DPtr + 16); localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr + 8);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr + 4);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr + 2);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
  v = *(localArray1DPtr + 1);  localMemBarrier(); *localArray1DPtr += v; localMemBarrier();
}
#endif

inline uint setLocalCount(const uchar localKey, const uchar keyPrefix)
{
#if RadixScanIterations == 1
  return 1 << (localKey << 3);
#else
  return select(0, (1 << ((localKey & 3) << 3)), (localKey & 0xFC) == keyPrefix);
#endif
}

#ifdef UsePackedKeys
#define getKey(keys, index)       ((keys >> (index * SortBits)) & (SortBitValue - 1))
#define setKey(keys, index, key)  keys |= ((key & (SortBitValue - 1)) << (index * SortBits))
#else
#define getKey(keys, index)       keys[i]
#define setKey(keys, index, key)  keys[i] = key
#endif

Kernel void radixSort32BitReduceKernel(
  Device SortNode32* source,
  volatile Device uint* localSumBuffer,
  constantKernelInput(uint, rightShift),
  constantKernelInput(uint, length)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  // local index of the thread within a lane instance
  const uchar localIndexInLane = threadLocalIndex() & (ComputeSimdWidth - 1);
  // lane instance id
  const uchar laneIndex = threadLocalIndex() >> ComputeSimdWidthExp;

#ifndef OneBlockPerGroup
  const uint maxBlocks = (length + ComputeSimdWidth - 1) / ComputeSimdWidth;
  const uint blocksPerGroup = (maxBlocks + FetchAlignmentExp * RadixBlockInstances * threadGroupCount() - 1) / (FetchAlignmentExp * RadixBlockInstances * threadGroupCount());
#else
  #define blocksPerGroup 1
#endif

  uint startIndex = ((laneIndex + threadGroupIndex() * RadixBlockInstances) * blocksPerGroup * FetchAlignmentExp) << ComputeSimdWidthExp;
  const uint endIndex = startIndex + ((FetchAlignmentExp * blocksPerGroup) << ComputeSimdWidthExp);
  startIndex += localIndexInLane;

  const ushort laneCountOffsetInTG = laneIndex << (ComputeSimdWidthExp + BankConflictShift);
  const ushort laneIndexOffsetInTG = laneIndex << SortBits;

#ifndef USE_SIMD_COMPUTE
  Shared RadixPackedType localCountHeap[(RadixBlockInstances << ComputeSimdWidthExp) << BankConflictShift];
  Shared RadixPackedType* localCount = ComputeSimdWidth + localCountHeap + laneCountOffsetInTG;
#endif
  Shared uint threadgroupLocalCountTotals[RadixBlockInstances << SortBits];

#ifndef USE_SIMD_COMPUTE
  localCount[localIndexInLane - ComputeSimdWidth] = 0;
#endif

  // clear block sums for the first thread of each group
  if (localIndexInLane < SortBitValue)
  {
    threadgroupLocalCountTotals[laneIndex * SortBitValue + localIndexInLane] = 0;
  }

  // for each sub block
#ifndef OneBlockPerGroup
  for (uint index = startIndex; index < endIndex;)
#else
  uint index = startIndex;
#endif
  {
#ifdef UsePackedKeys
    uint localKeys = 0;
#else
    uchar localKeys[1 << RadixReductionPackingExp];
#endif

    // accumulate the local key occurrence for multiple successive elements
    for (uchar i = 0; i < (1 << RadixReductionPackingExp); i++, index += ComputeSimdWidth)
    {
      uchar localKey;
      if (index < length)
      {
        localKey = source[index].key >> rightShift;
      }
      else
      {
        localKey = defaultSortNode().key >> rightShift;
      }
      setKey(localKeys, i, (localKey & (SortBitValue - 1)));
    }

    for (ushort j = 0; j < RadixScanIterations*PackedParts; j += PackedParts)
    {
      RadixPackedType count = 0;

      for (uchar i = 0; i < (1 << RadixReductionPackingExp); i++)
      {
        count += setLocalCount(getKey(localKeys, i), j);
      }

#ifndef USE_SIMD_COMPUTE
      localCount[localIndexInLane] = count;
      laneReduceRadix(localCount, localIndexInLane);
#else
      RadixPackedType reduce;
      REDUCE_FUNCTION(reduce, count);
#endif

      if (localIndexInLane == 0)
      {
        ushort offset = laneIndexOffsetInTG + j;

        // subtract first to account potential 8 bit overflow
        // add first to 32 bit converted value
#ifndef USE_SIMD_COMPUTE
        uchar4 temp = asUchar4(localCount[0] - count);
#else
        uchar4 temp = asUchar4(reduce - count);
#endif
        ushort4 sum = convertUshort4(temp);
        temp = asUchar4(count);
        sum += constructUshort4(temp.x, temp.y, temp.z, temp.w);

        threadgroupLocalCountTotals[offset++] += sum.x;
        threadgroupLocalCountTotals[offset++] += sum.y;
        threadgroupLocalCountTotals[offset++] += sum.z;
        threadgroupLocalCountTotals[offset++] += sum.w;
      }
    }
  }

  if (localIndexInLane < SortBitValue)
  {
    const uint sumBufferIndex = laneIndex + (localIndexInLane * threadGroupCount() + threadGroupIndex()) * RadixBlockInstances;
    const ushort offset = laneIndexOffsetInTG + localIndexInLane;

    localSumBuffer[sumBufferIndex] = threadgroupLocalCountTotals[offset];
  }
}

inline void fetchNodes(SortNode32 localSortNodes[], const Device SortNode32* source, const uint index, const uint length)
{
  if (index + ((1 << RadixPrefixScanPackingExp) - 1) < length)
  {
#if RadixPrefixScanPackingExp == 1
    *((Thread uint4*)localSortNodes) = *((const Device uint4*)(source + index));
#elif RadixPrefixScanPackingExp == 2
    *((Thread commonUint8*)localSortNodes) = *((const Device commonUint8*)(source + index));
#elif RadixPrefixScanPackingExp == 3
    *((Thread commonUint16*)localSortNodes) = *((const Device commonUint16*)(source + index));
#endif
  }
  else
  {
    for (uchar w = 0; w < (1 << RadixPrefixScanPackingExp); w++)
    {
      localSortNodes[w] = ((index + w) < length) ? source[index + w] : defaultSortNode();
    }
  }
}

Kernel void radixSort32BitSortKernel(
  Device SortNode32* destination,
  const Device SortNode32* source,
  const Device uint* localSumBuffer,
  constantKernelInput(uint, rightShift),
  constantKernelInput(uint, length)
  KERNEL_THREAD_ARGUMENTS
  KERNEL_THREADGROUP_ARGUMENTS)
{
  // local index of the thread within a lane instance
  const uchar localIndexInLane = threadLocalIndex() & (ComputeSimdWidth - 1);
  // lane instance id
  const uchar laneIndex = threadLocalIndex() >> ComputeSimdWidthExp;

#ifndef OneBlockPerGroup
  const uint maxBlocks = (length + ComputeSimdWidth - 1) / ComputeSimdWidth;
  const uint blocksPerGroup = (maxBlocks + FetchAlignmentExp * RadixBlockInstances * threadGroupCount() - 1) / (FetchAlignmentExp * RadixBlockInstances * threadGroupCount());
#else
  #define blocksPerGroup 1
#endif

  uint startIndex = ((laneIndex + threadGroupIndex() * RadixBlockInstances) * blocksPerGroup * FetchAlignmentExp) << ComputeSimdWidthExp;
  const uint endIndex = startIndex + ((FetchAlignmentExp * blocksPerGroup) << ComputeSimdWidthExp);
  startIndex += localIndexInLane;

  const ushort laneIndexOffsetInTG = laneIndex << SortBits;
  const ushort laneSortNodeOffsetInTG = laneIndex << (ComputeSimdWidthExp + RadixPrefixScanPackingExp);

  SortNode32 localSortNodes[1 << RadixPrefixScanPackingExp];

#ifdef CoalescedWrites
  Shared SortNode32 localSortNodesHeap[(RadixBlockInstances << ComputeSimdWidthExp) << RadixPrefixScanPackingExp];
  Shared SortNode32* swapSourceOffset = localSortNodesHeap + laneSortNodeOffsetInTG + (localIndexInLane << RadixPrefixScanPackingExp);
  Shared SortNode32* swapDestOffset = localSortNodesHeap + laneSortNodeOffsetInTG + localIndexInLane;
#endif

#if defined(BusAlignedFetch) && defined(CoalescedWrites)
  Shared RadixPackedType* localCountHeap = (Shared RadixPackedType*)localSortNodesHeap;
  const ushort instanceCountOffset = ComputeSimdWidth + (laneIndex << (ComputeSimdWidthExp + BankConflictShift + 1));
#else
  Shared RadixPackedType localCountHeap[(RadixBlockInstances << ComputeSimdWidthExp) << BankConflictShift];
  const ushort instanceCountOffset = ComputeSimdWidth + (laneIndex << (ComputeSimdWidthExp + BankConflictShift));
#endif

#ifndef BusAlignedFetch
  Shared SortNode32* offsettedLocalSortNodes = localSortNodesHeap + laneSortNodeOffsetInTG;
#endif

  Shared uint threadgroupLocalCountTotals[RadixBlockInstances << SortBits];
  Shared uint *threadgroupLocalCountTotalsForInstance = threadgroupLocalCountTotals + laneIndexOffsetInTG;
  Shared RadixPackedType *localPrefixCount = localCountHeap + instanceCountOffset;

#if !(defined(BusAlignedFetch) && defined(CoalescedWrites))
  localPrefixCount[localIndexInLane - ComputeSimdWidth] = 0;
#endif

  if (localIndexInLane < SortBitValue)
  {
    // prefix scan instance values per bit
    const uint sumBufferIndex = threadGroupIndex() * RadixBlockInstances + laneIndex + (localIndexInLane * threadGroupCount() * RadixBlockInstances);
    const ushort offset = laneIndexOffsetInTG + localIndexInLane;

    threadgroupLocalCountTotals[offset] = localSumBuffer[sumBufferIndex];
  }

#ifdef BusAlignedFetch
#ifndef OneBlockPerGroup
  for (uint index = startIndex + (localIndexInLane << RadixPrefixScanPackingExp) - localIndexInLane; index < endIndex; index += (ComputeSimdWidth << RadixPrefixScanPackingExp))
#else
  const uint index = startIndex + (localIndexInLane << RadixPrefixScanPackingExp) - localIndexInLane;
#endif
#else
#ifndef OneBlockPerGroup
  for (uint index = startIndex; index < endIndex; index += (ComputeSimdWidth << RadixPrefixScanPackingExp))
#else
  const uint index = startIndex;
#endif
#endif
  {

#ifdef BusAlignedFetch
    // accumulated local key occurrence
    fetchNodes(localSortNodes, source, index, length);
#else
    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      const uint indexOffset = index + (i * ComputeSimdWidth);
      offsettedLocalSortNodes[localIndexInLane + (i * ComputeSimdWidth)] = (indexOffset < length) ? source[indexOffset] : defaultSortNode();
    }

    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      localSortNodes[i] = offsettedLocalSortNodes[(localIndexInLane * (1 << RadixPrefixScanPackingExp)) + i];
    }
#endif

    uint destOffset[1 << RadixPrefixScanPackingExp];

    for (uint j = 0; j < RadixScanIterations*PackedParts; j += PackedParts)
    {
      RadixPackedType reduceSum = 0;

#ifdef UsePackedKeys
      uint localKeys = 0;
      uint localKeyValid = 0;
#else
      uchar localKeys[1 << RadixPrefixScanPackingExp];
      uchar localKeyValid[1 << RadixPrefixScanPackingExp];
#endif

      for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
      {
        const uchar localKey = (localSortNodes[i].key >> rightShift) & (SortBitValue - 1);
        setKey(localKeys, i, localKey);

        const uint sum = setLocalCount(localKey, j);
        reduceSum += sum;

#ifdef UsePackedKeys
        if (sum != 0)
          localKeyValid |= (1 << i);
#else
        localKeyValid[i] = (sum != 0);
#endif
      }

#ifndef USE_SIMD_COMPUTE
      localPrefixCount[localIndexInLane] = reduceSum;
      uint prefixScan = lanePrefixScanRadix(localPrefixCount, localIndexInLane) - reduceSum;
#else
      uint prefixScan;
      SCAN_FUNCTION(prefixScan, reduceSum);
      prefixScan -= reduceSum;
#endif

      for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
      {
#ifdef UsePackedKeys
        if (localKeyValid & (1 << i))
#else
        if (localKeyValid[i])
#endif
        {
          const uchar localKey = getKey(localKeys, i);

          uchar count = (prefixScan >> ((localKey & (PackedParts - 1)) * PackedBits));
          count &= ((1 << PackedBits) - 1);

          destOffset[i] = threadgroupLocalCountTotalsForInstance[localKey] + count;
          prefixScan += setLocalCount(localKey, j);
        }
      }

      if (localIndexInLane == (ComputeSimdWidth - 1))
      {
        prefixScan -= reduceSum;
        ushort start = laneIndexOffsetInTG + j;
        const ushort end = start + 4;
        for ( ; start < end; start++)
        {
          threadgroupLocalCountTotals[start] += ((prefixScan & ((1 << PackedBits) - 1)) + (reduceSum & ((1 << PackedBits) - 1)));
          prefixScan >>= PackedBits;
          reduceSum >>= PackedBits;
        }
      }
    }

#ifdef CoalescedWrites
    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      swapSourceOffset[i].key = destOffset[i];
    }

    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      destOffset[i] = swapDestOffset[i * ComputeSimdWidth].key;
    }

#ifdef BusAlignedFetch
    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      swapSourceOffset[i] = localSortNodes[i];
    }

    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      localSortNodes[i] = swapDestOffset[i * ComputeSimdWidth];
    }
#endif

#endif

    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      if (destOffset[i] < length)
      {
#ifdef BusAlignedFetch
        destination[destOffset[i]] = localSortNodes[i];
#else
#ifdef CoalescedWrites
        destination[destOffset[i]] = offsettedLocalSortNodes[localIndex + (i * ComputeSimdWidth)];
#else
        destination[destOffset[i]] = localSortNodes[i];
#endif
#endif
      }
    }
  }
}

#endif

#endif
