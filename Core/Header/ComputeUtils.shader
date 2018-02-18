#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

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

Kernel void calculateSum(Device StructType* array, uint length, const uint iteration,
  const uint maxLocalIterations, const uint divideFlag)
{
  /*
  uint index = threadIndex();

  Group StructType localArray[2 * COMPUTE_MAX_THREADS];

  if (maxLocalIterations) // if all local copy to local memory
  {
  const uint index2 = index << 1;
  localArray[index2] = array[(index2 << 1) << maxLocalIterations]STRUCT_MEMBER;
  localArray[index2 + 1] = array[((index2 << 1) + 1) << maxLocalIterations]STRUCT_MEMBER;
  }

  // each iteration will do index + index^(2*iteration)
  index = index << (1 + iteration);

  for (uint i = 0; i < maxLocalIterations; i++)
  {
  uint index2 = index + (1 << (i + iteration));
  if (index2 < length)
  {
  //localArray[originalIndex] = array[index]STRUCT_MEMBER;
  array[index]STRUCT_MEMBER += array[index2]STRUCT_MEMBER;
  }
  index <<= 1;
  if (maxLocalIterations > 1)
  {
  barrier(CLK_GLOBAL_MEM_FENCE);
  }
  }
  if (divideFlag && index == 0)
  {
  array[index]STRUCT_MEMBER /= length;
  }
  */

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

#ifdef IndexStructType

#define COMPUTE_MAX_THREADS 1024

#define DEBUG_SUM_PARTITION

Kernel void calculateSumPartitions(Device StructType* array, const Device IndexStructType* partitionArray,
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
              if (diff < width && diff >= (width >> 1) && (index1 & ((width << 1) - 1)))
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

              ADD_FUNCTION(array[index]STRUCT_MEMBER, array[index2]STRUCT_MEMBER);
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