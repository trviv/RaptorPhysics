#ifndef COMPUTE_UTILS_COMPLEX_REDUCE_H
#define COMPUTE_UTILS_COMPLEX_REDUCE_H

#if defined(StructType) && defined(IdentityStructType)

Kernel void sumIrregular2DKernel(
  Device StructType* array2D,
  Device StructType* consolidatedArray,
  Device IdentityStructType* array2DIdentity,
  const Device PartitionInfo* partitionArray,
  const Device uint* partitionCount,
  const int length,
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
      int width = (1 << maxPower);
      int index1 = (originalIndex << maxPower) + offset;
      int index2 = index1 + (width >> 1);

      if (index1 < length)
      {
        IdentityStructType backup = array2DIdentity[index1];
        uint identity1 = IDENTITY_FUNCTION(backup IDENTITY_STRUCT_MEMBER);

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
                // write identity if identity and data arrays are the same, to avoid packing relate issues
                if (array2DIdentity == array2D)
                {
                  array2DIdentity[index1]IDENTITY_STRUCT_MEMBER = (backup IDENTITY_STRUCT_MEMBER);
                }
              }

              if (maxPower == 1 && backwards && diff < width)
              {
                if (divideFlag)
                {
                  float div = partitionArray[identity1].count;
                  DIV_FUNCTION(array2D[index1]STRUCT_MEMBER, div);
                  // write identity if identity and data arrays are the same, to avoid packing relate issues
                  if (array2DIdentity == array2D)
                  {
                    array2DIdentity[index1]IDENTITY_STRUCT_MEMBER = (backup IDENTITY_STRUCT_MEMBER);
                  }
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
