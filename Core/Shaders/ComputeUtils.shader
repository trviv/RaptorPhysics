#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

#ifdef StructType

Kernel void bandwidthReadTest(
  const Device StructType* array,
  constantKernelInput(uint, length)
  KERNEL_GLOBAL_ARGUMENTS)
{
  StructType value;
  uint index = threadIndex();

  if (index < length)
  {
    value = array[index];
  }
}

Kernel void bandwidthWriteTest(
  Device StructType* array,
  constantKernelInput(uint, length)
  KERNEL_GLOBAL_ARGUMENTS)
{
  StructType value;
  uint index = threadIndex();

  if (index < length)
  {
    array[index] = value;
  }
}

Kernel void consolidateFromPartitionsKernel(
  const Device StructType* source,
  Device StructType* destination,
  const Device PartitionInfo* partitions,
  const Device uint* partitionsCount
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index < partitionsCount[0])
  {
    COPY_FUNCTION(destination[index]STRUCT_MEMBER, source[partitions[index].offset]STRUCT_MEMBER);
  }
}

#endif

Kernel void bitonicSort32BitKernel(
  Device SortNode32* array1D,
  constantKernelInput(uint, multiplier),
  constantKernelInput(int, minDepth),
  constantKernelInput(int, maxDepth),
  constantKernelInput(uint, length)
  KERNEL_GLOBAL_ARGUMENTS
  KERNEL_THREAD_ARGUMENTS)
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

  for (int d = minDepth; d < maxDepth; d++)
  {
    for (int d2 = d; d2 >= minDepth; d2--)
    {
      const uint offset = 1 << d2;

      for (uint m = 0; m < multiplier; m++)
      {
        const uint index1 = (localIndex + m * MAX_LOCAL_NODES) << d2;
        const uint index2 = index1 + offset;

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

Kernel void showMatrix(Device float* array,
  constantKernelInput(uint, rowLength),
  constantKernelInput(uint, strideIn4Byte),
  constantKernelInput(uint, length)
  KERNEL_GLOBAL_ARGUMENTS)
{
  uint index = threadIndex();
  if (index < length / 3)
  {
    for (uint i = 0; i < rowLength / 3; i++)
    {
#ifndef USE_METAL_COMPUTE
      printf("%f %f %f\n", array[index * strideIn4Byte + i * 3], array[index * strideIn4Byte + i * 3 + 1], array[index * strideIn4Byte + i * 3 + 2]);
#endif
    }
  }
}

// kernel to clear an integer buffer to all 0s
Kernel void clearIntegerBuffer(
  Device uint* destination,
  constantKernelInput(uint, value),
  constantKernelInput(int, length)
  KERNEL_GLOBAL_ARGUMENTS)
{
  int writeSize = length - (int)threadIndex() * BatchSize;

  if (writeSize >= BatchSize)
  {
#ifndef USE_METAL_COMPUTE
#if BatchSize == 4
    ((Device uint4*)destination)[threadIndex()] = value;
#elif BatchSize == 8
    ((Device uint8*)destination)[threadIndex()] = value;
#elif BatchSize == 16
    ((Device uint16*)destination)[threadIndex()] = value;
#elif BatchSize == 1
    destination[threadIndex()] = value;
#else
    assert;
#endif
#else
#if BatchSize == 4
    ((Device uint4*)destination)[threadIndex()] = value;
#elif BatchSize == 8
    ((Device dummy_uint8*)destination)[threadIndex()] = value;
#elif BatchSize == 16
    ((Device dummy_uint16*)destination)[threadIndex()] = value;
#elif BatchSize == 1
    destination[threadIndex()] = value;
#else
    assert;
#endif
#endif
  }
  else
  {
    for (int i = 0; i < writeSize; i++)
    {
      (destination + threadIndex() * BatchSize)[i] = value;
    }
  }
}

#endif
