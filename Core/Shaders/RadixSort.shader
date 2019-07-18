#ifndef COMPUTE_UTILS_RADIX_SORT_H
#define COMPUTE_UTILS_RADIX_SORT_H

#if !defined(SkipParallelPrimitives) && defined(StructTypeIntegral)

#define BusAlignedFetch
#define UsePackedKeys
//#define CoalescedWrites

#if defined(BusAlignedFetch) && defined(CoalescedWrites)
#undef BankConflictShift
#define BankConflictShift RadixPrefixScanPackingExp
#endif

#define LaneWidthExp        COMPUTE_SUB_GROUP_EXP
#define PackedParts         4
#define PackedBits          (32 / PackedParts)
#define RadixScanIterations (1 << (SortBits-2))
#define RadixPackedType     uint
#define LaneWidth           (1<<LaneWidthExp)
#define RadixBlockInstances (256>>LaneWidthExp)
#define SortBitValue        (1<<SortBits)

#if RadixPrefixScanPackingExp > RadixReductionPackingExp
#define FetchAlignmentExp (1<<RadixPrefixScanPackingExp)
#else
#define FetchAlignmentExp (1<<RadixReductionPackingExp)
#endif

const RadixPackedType lanePrefixScanRadix(volatile Shared RadixPackedType* localArray1D, const ushort localIndex)
{
  volatile Shared RadixPackedType *localArray1DPtr = localArray1D + localIndex;

  *localArray1DPtr += *(localArray1DPtr - 1);
  *localArray1DPtr += *(localArray1DPtr - 2);
  *localArray1DPtr += *(localArray1DPtr - 4);
  *localArray1DPtr += *(localArray1DPtr - 8);
  *localArray1DPtr += *(localArray1DPtr - 16);
#if COMPUTE_SUB_GROUP_SIZE > 32
  *localArray1DPtr += *(localArray1DPtr - 32);
#endif

  return localArray1D[localIndex];
}

void laneReduceRadix(volatile Shared StructType* localArray1D, const ushort localIndex)
{
  volatile Shared RadixPackedType *localArray1DPtr = localArray1D + localIndex;

#if COMPUTE_SUB_GROUP_SIZE > 32
  *localArray1DPtr += *(localArray1DPtr + 32);
#endif
  *localArray1DPtr += *(localArray1DPtr + 16);
  *localArray1DPtr += *(localArray1DPtr + 8);
  *localArray1DPtr += *(localArray1DPtr + 4);
  *localArray1DPtr += *(localArray1DPtr + 2);
  *localArray1DPtr += *(localArray1DPtr + 1);
}

uint setLocalCount(const uchar localKey, const uchar keyPrefix)
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
  const uint rightShift,
  const uint length)
{
  // local index of the thread within a lane instance
  const ushort localIndexInLane = threadLocalIndex() & (LaneWidth - 1);
  // lane instance id
  const uchar laneIndex = threadLocalIndex() >> LaneWidthExp;

  const uint maxBlocks = (length + LaneWidth - 1) / LaneWidth;
  const uint blocksPerGroup = (maxBlocks + FetchAlignmentExp * RadixBlockInstances * threadGroupCount() - 1) / (FetchAlignmentExp * RadixBlockInstances * threadGroupCount());

  uint startIndex = ((laneIndex + threadGroupIndex() * RadixBlockInstances) * blocksPerGroup * FetchAlignmentExp) << LaneWidthExp;
  const uint endIndex = startIndex + ((FetchAlignmentExp * blocksPerGroup) << LaneWidthExp);
  startIndex += localIndexInLane;

  const ushort laneCountOffsetInTG = laneIndex << (LaneWidthExp + BankConflictShift);
  const ushort laneIndexOffsetInTG = laneIndex << SortBits;

  Shared RadixPackedType localCountHeap[(RadixBlockInstances << LaneWidthExp) << BankConflictShift];
  Shared RadixPackedType* localCount = LaneWidth + localCountHeap + laneCountOffsetInTG;
  Shared uint threadgroupLocalCountTotals[RadixBlockInstances << SortBits];

  localCount[localIndexInLane - LaneWidth] = 0;

  // clear block sums for the first thread of each group
  if (localIndexInLane < SortBitValue)
  {
    threadgroupLocalCountTotals[laneIndex * SortBitValue + localIndexInLane] = 0;
  }

  // for each sub block
  for (uint index = startIndex; index < endIndex;)
  {
#ifdef UsePackedKeys
    uint localKeys = 0;
#else
    uchar localKeys[1 << RadixReductionPackingExp];
#endif

    // accumulate the local key occurrence for multiple successive elements
    for (uchar i = 0; i < (1 << RadixReductionPackingExp); i++, index += LaneWidth)
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

    for (uint j = 0; j < RadixScanIterations*PackedParts; j += PackedParts)
    {
      RadixPackedType count = 0;

      for (uchar i = 0; i < (1 << RadixReductionPackingExp); i++)
      {
        count += setLocalCount(getKey(localKeys, i), j);
      }
      localCount[localIndexInLane] = count;

      laneReduceRadix(localCount, localIndexInLane);

      if (localIndexInLane == 0)
      {
        ushort offset = laneIndexOffsetInTG + j;

        // subtract first to account potential 8 bit overflow
        // add first to 32 bit converted value
        uchar4 temp = as_uchar4(localCount[0] - count);
        ushort4 sum = convert_ushort4(temp);
        temp = as_uchar4(count);
        sum += (ushort4)(temp.x, temp.y, temp.z, temp.w);

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
    *((Thread uint4*)localSortNodes) = *((Device uint4*)(source + index));
#elif RadixPrefixScanPackingExp == 2
    *((Thread uint8*)localSortNodes) = *((Device uint8*)(source + index));
#elif RadixPrefixScanPackingExp == 3
    *((Thread uint16*)localSortNodes) = *((Device uint16*)(source + index));
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
  volatile Device SortNode32* destination,
  const Device SortNode32* source,
  const Device uint* localSumBuffer,
  const uint rightShift,
  const uint length)
{
  // local index of the thread within a lane instance
  const ushort localIndexInLane = threadLocalIndex() & (LaneWidth - 1);
  // lane instance id
  const uchar laneIndex = threadLocalIndex() >> LaneWidthExp;

  const uint maxBlocks = (length + LaneWidth - 1) / LaneWidth;
  const uint blocksPerGroup = (maxBlocks + FetchAlignmentExp * RadixBlockInstances * threadGroupCount() - 1) / (FetchAlignmentExp * RadixBlockInstances * threadGroupCount());

  uint startIndex = ((laneIndex + threadGroupIndex() * RadixBlockInstances) * blocksPerGroup * FetchAlignmentExp) << LaneWidthExp;
  const uint endIndex = startIndex + ((FetchAlignmentExp * blocksPerGroup) << LaneWidthExp);
  startIndex += localIndexInLane;

  const ushort laneIndexOffsetInTG = laneIndex << SortBits;
  const ushort laneSortNodeOffsetInTG = laneIndex << (LaneWidthExp + RadixPrefixScanPackingExp);

  SortNode32 localSortNodes[1 << RadixPrefixScanPackingExp];

  Shared SortNode32 localSortNodesHeap[(RadixBlockInstances << LaneWidthExp) << RadixPrefixScanPackingExp];
#ifdef CoalescedWrites
  Shared SortNode32* swapSourceOffset = localSortNodesHeap + laneSortNodeOffsetInTG + (localIndexInLane << RadixPrefixScanPackingExp);
  Shared SortNode32* swapDestOffset = localSortNodesHeap + laneSortNodeOffsetInTG + localIndexInLane;
#endif

#if defined(BusAlignedFetch) && defined(CoalescedWrites)
  Shared RadixPackedType* localCountHeap = (Shared RadixPackedType*)localSortNodesHeap;
  const ushort instanceCountOffset = LaneWidth + (laneIndex << (LaneWidthExp + BankConflictShift + 1));
#else
  Shared RadixPackedType localCountHeap[(RadixBlockInstances << LaneWidthExp) << BankConflictShift];
  const ushort instanceCountOffset = LaneWidth + (laneIndex << (LaneWidthExp + BankConflictShift));
#endif

#ifndef BusAlignedFetch
  Shared SortNode32* offsettedLocalSortNodes = localSortNodesHeap + laneSortNodeOffsetInTG;
#endif

  Shared uint threadgroupLocalCountTotals[RadixBlockInstances << SortBits];
  Shared uint *threadgroupLocalCountTotalsForInstance = threadgroupLocalCountTotals + laneIndexOffsetInTG;
  Shared RadixPackedType *localPrefixCount = localCountHeap + instanceCountOffset;// LaneWidth * ((1 << BankConflictShift) - 1) + instanceCountOffset;

#if defined(BusAlignedFetch) && defined(CoalescedWrites)
#else
  localPrefixCount[localIndexInLane - LaneWidth] = 0;
#endif

  if (localIndexInLane < SortBitValue)
  {
    // prefix scan instance values per bit
    const uint sumBufferIndex = threadGroupIndex() * RadixBlockInstances + laneIndex + (localIndexInLane * threadGroupCount() * RadixBlockInstances);
    const ushort offset = laneIndexOffsetInTG + localIndexInLane;

    threadgroupLocalCountTotals[offset] = localSumBuffer[sumBufferIndex];
  }

#ifdef BusAlignedFetch
  for (uint index = startIndex + (localIndexInLane << RadixPrefixScanPackingExp) - localIndexInLane; index < endIndex; index += (LaneWidth << RadixPrefixScanPackingExp))
#else
  for (uint index = startIndex; index < endIndex; index += (LaneWidth << RadixPrefixScanPackingExp))
#endif
  {

#ifdef BusAlignedFetch
    // accumulated local key occurrence
    fetchNodes(localSortNodes, source, index, length);
#else
    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      const uint indexOffset = index + (i * LaneWidth);
      offsettedLocalSortNodes[localIndexInLane + (i * LaneWidth)] = (indexOffset < length) ? source[indexOffset] : defaultSortNode();
    }

    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      localSortNodes[i] = offsettedLocalSortNodes[(localIndexInLane * (1 << RadixPrefixScanPackingExp)) + i];
    }
#endif

    uint destOffset[1 << RadixPrefixScanPackingExp];

    for (uint j = 0; j < RadixScanIterations*PackedParts; j += PackedParts)
    {

#if defined(BusAlignedFetch) && defined(CoalescedWrites)
      localPrefixCount[localIndexInLane - LaneWidth] = 0;
#endif

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
        const char localKey = (localSortNodes[i].key >> rightShift) & (SortBitValue - 1);
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

      localPrefixCount[localIndexInLane] = reduceSum;

      uint prefixScan = lanePrefixScanRadix(localPrefixCount, localIndexInLane) - reduceSum;

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

      if (localIndexInLane == (LaneWidth - 1))
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
      destOffset[i] = swapDestOffset[i * LaneWidth].key;
    }

#ifdef BusAlignedFetch
    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      swapSourceOffset[i] = localSortNodes[i];
    }

    for (uchar i = 0; i < (1 << RadixPrefixScanPackingExp); i++)
    {
      localSortNodes[i] = swapDestOffset[i * LaneWidth];
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
        destination[destOffset[i]] = offsettedLocalSortNodes[localIndex + (i * LaneWidth)];
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
