#ifndef COMPUTE_UTILS_SHARED_H
#define COMPUTE_UTILS_SHARED_H

#define PREFIX_SCAN_STATUS_INVALID  0
#define PREFIX_SCAN_STATUS_PARTIAL  1
#define PREFIX_SCAN_STATUS_FINAL    2

#define REDUCE_STATUS_INVALID       0
#define REDUCE_STATUS_PARTIAL       1
#define REDUCE_STATUS_FINAL         2

#ifdef COMPUTE_SHADER_SCOPE

#include "ComputeHeader.shader"
#include "ComputeShared.h"

#ifdef StructMember
#define STRUCT_MEMBER           .StructMember
#else
#define STRUCT_MEMBER
#endif

#ifndef MemberStructType
#define MemberStructType        StructType
#define NoMemberStruct
#endif

#ifdef IdentityStructMember
#define IDENTITY_STRUCT_MEMBER  .IdentityStructMember
#else
#define IDENTITY_STRUCT_MEMBER
#endif

#ifdef IdentityFunction
#define IDENTITY_FUNCTION(x)    IdentityFunction(x)
#else
#define IDENTITY_FUNCTION(x)
#endif

#ifdef AddFunction
#define ADD_FUNCTION(x, y)      AddFunction(&(x), &(y))
#else
#define ADD_FUNCTION(x, y)      x += y
#endif

#ifdef CopyFunction
#define COPY_FUNCTION(x, y)     CopyFunction(&(x), &(y))
#else
#define COPY_FUNCTION(x, y)     x = y
#endif

#ifdef DivFunction
#define DIV_FUNCTION(x, y)      DivFunction(&(x), &(y))
#else
#define DIV_FUNCTION(x, y)      x /= y
#endif

#ifdef ClearFunction
#define CLEAR_FUNCTION(x, y)    ClearFunction(&(x), y)
#else
#define CLEAR_FUNCTION(x, y)    (x) = y
#endif

#ifdef ReduceFunction
#define REDUCE_FUNCTION(o, i)   ReduceFunction(o, i)
#else
#define REDUCE_FUNCTION(o, i)   o = simdReduce(i)
#endif

#ifdef ScanFunction
#define SCAN_FUNCTION(o, i)     ScanFunction(o, i)
#else
#define SCAN_FUNCTION(o, i)     o = simdScan(i)
#endif

#define SHORT_BATCH_RW 1

// general atomics for structures
inline MemberStructType atomicLoadN(volatile Device MemberStructType* x)
{
  MemberStructType ret;
  for (int i = 0; i < (int)sizeof(MemberStructType)/4; i++)
  {
    ((Thread uint*)&ret)[i] = atomicLoad(((volatile Device uint*)x) + i);
  }
  return ret;
}

inline void atomicStoreN(volatile Device MemberStructType* x, const MemberStructType y)
{
  for (int i = 0; i < (int)sizeof(MemberStructType)/4; i++)
  {
    atomicStore(((volatile Device uint*)x) + i, ((const Thread uint*)&y)[i]);
  }
}


#define _STR(x) #x
#define STR(x) _STR(x)

#pragma message ("Struct type: " STR(StructType) ", Member Struct type: " STR(MemberStructType))
#pragma message ("Atomics for: " STR(MemberStructType))
#pragma message ("Batch Size: " STR(BatchSize))

#define COMPUTE_MAX_THREADS         MaxWorkgroupSize
#define REDUCE_COMPUTE_THREADS      MaxWorkgroupSize
#define REDUCE_SHARED_SIZE          paddedIndex(MaxWorkgroupSize)
#define PREFIX_SCAN_COMPUTE_THREADS MaxWorkgroupSize
#define PREFIX_SCAN_SHARED_SIZE     paddedIndex(MaxWorkgroupSize)

// function to write to a memory and wait until the written data is visible
// this helps to get consistent write memory ordering on AMD GPU
inline static void writeAndWait(volatile Device MemberStructType* location, const MemberStructType value)
{
//  COPY_FUNCTION(*location, value);

  INIT_POLL();
  MemberStructType temp;
  do
  {
    ATOMIC_STORE_FUNCTION(location, value);
    temp = ATOMIC_LOAD_FUNCTION(location);
  }
  while (((Thread uint*)&temp)[0] != ((Thread uint*)&value)[0] && !POLL_TIMEOUT());
}

inline static MemberStructType localReduce(const Thread MemberStructType *elements)
{
  MemberStructType ret = elements[0];
  for (uint i = 1; i < BatchSize; i++)
  {
    ADD_FUNCTION(ret, elements[i]);
  }
  return ret;
}

inline static void batchRead(Thread MemberStructType *elements, const Device StructType* array1D, const uint index, const uint length)
{
  const uint indexOffset = (index << BatchSizeExp);
  const uint readCount = min(select((uint)0, length - indexOffset, length > indexOffset), (uint)BatchSize);

#if defined(NoMemberStruct) && BatchSize > 1
#pragma message ("Using Batch Read")

#if SHORT_BATCH_RW
  if (readCount < BatchSize)
  {
    for (ushort i=0; i<readCount; i++)
    {
      elements[i] = ((Device MemberStructType*)&(array1D[indexOffset+i]STRUCT_MEMBER))[0];
    }
    for (ushort i=readCount; i<BatchSize; i++)
    {
      elements[i] = (MemberStructType)(0);
    }
  }
  else
  {
#if BatchSize == 16
    *((Thread MemberStructType16*)elements) = *((const Device MemberStructType16*)(array1D + indexOffset));
#elif BatchSize == 8
    *((Thread MemberStructType8*)elements)  = *((const Device MemberStructType8*)(array1D + indexOffset));
#elif BatchSize == 4
    *((Thread MemberStructType4*)elements)  = *((const Device MemberStructType4*)(array1D + indexOffset));
#else
    for (ushort i=0; i<readCount; i++)
    {
      elements[i] = ((Device MemberStructType*)&(array1D[indexOffset+i]STRUCT_MEMBER))[0];
    }
#endif
  }

#else
  if (readCount < BatchSize)
#if BatchSize == 16
  * ((Thread MemberStructType16*)elements) = (MemberStructType16)(0);
#elif BatchSize == 8
  * ((Thread MemberStructType8*)elements) = (MemberStructType8)(0);
#elif BatchSize == 4
  * ((Thread MemberStructType4*)elements) = (MemberStructType4)(0);
#endif

  switch (readCount)
  {
#if BatchSize >= 16
  case 16:
  {
    *((Thread MemberStructType16*)elements)       = *((const Device MemberStructType16*)(array1D + indexOffset));
    break;
  }
  case 15:
  {
    *((Thread MemberStructType8*)(elements + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(elements + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    *((Thread MemberStructType2*)(elements + 12)) = *((const Device MemberStructType2*)(array1D + indexOffset + 12));
    *((Thread MemberStructType*)(elements + 14))  = *((const Device MemberStructType*)(array1D + indexOffset + 14));
    break;
  }
  case 14:
  {
    *((Thread MemberStructType8*)(elements + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(elements + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    *((Thread MemberStructType2*)(elements + 12)) = *((const Device MemberStructType2*)(array1D + indexOffset + 12));
    break;
  }
  case 13:
  {
    *((Thread MemberStructType8*)(elements + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(elements + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    *((Thread MemberStructType*)(elements + 12))  = *((const Device MemberStructType*)(array1D + indexOffset + 12));
    break;
  }
  case 12:
  {
    *((Thread MemberStructType8*)(elements + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(elements + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    break;
  }
  case 11:
  {
    *((Thread MemberStructType8*)(elements + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(elements + 8))  = *((const Device MemberStructType2*)(array1D + indexOffset + 8));
    *((Thread MemberStructType*)(elements + 10))  = *((const Device MemberStructType*)(array1D + indexOffset + 10));
    break;
  }
  case 10:
  {
    *((Thread MemberStructType8*)(elements + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(elements + 8))  = *((const Device MemberStructType2*)(array1D + indexOffset + 8));
    break;
  }
  case 9:
  {
    *((Thread MemberStructType8*)(elements + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType*)(elements + 8))   = *((const Device MemberStructType*)(array1D + indexOffset + 8));
    break;
  }
#endif
#if BatchSize >= 8
  case 8:
  {
    *((Thread MemberStructType8*)elements + 0)    = *((const Device MemberStructType8*)(array1D + indexOffset));
    break;
  }
  case 7:
  {
    *((Thread MemberStructType4*)(elements + 0))  = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(elements + 4))  = *((const Device MemberStructType2*)(array1D + indexOffset + 4));
    *((Thread MemberStructType*)(elements + 6))   = *((const Device MemberStructType*)(array1D + indexOffset + 6));
    break;
  }
  case 6:
  {
    *((Thread MemberStructType4*)(elements + 0))  = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(elements + 4))  = *((const Device MemberStructType2*)(array1D + indexOffset + 4));
    break;
  }
  case 5:
  {
    *((Thread MemberStructType4*)(elements + 0))  = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType*)(elements + 4))   = *((const Device MemberStructType*)(array1D + indexOffset + 4));
    break;
  }
#endif
  case 4:
  {
    *((Thread MemberStructType4*)elements + 0)    = *((const Device MemberStructType4*)(array1D + indexOffset));
    break;
  }
  case 3:
  {
    *((Thread MemberStructType2*)(elements + 0))  = *((const Device MemberStructType2*)(array1D + indexOffset));
    *((Thread MemberStructType*)(elements + 2))   = *((const Device MemberStructType*)(array1D + indexOffset + 2));
    break;
  }
  case 2:
  {
    *((Thread MemberStructType2*)elements + 0)    = *((const Device MemberStructType2*)(array1D + indexOffset));
    break;
  }
  case 1:
  {
    *((Thread MemberStructType*)elements + 0)     = *((const Device MemberStructType*)(array1D + indexOffset));
    break;
  }
  default:
    break;
  }
#endif

#else

#pragma message ("Using Loop Read")
  for (ushort i=0; i<BatchSize; i++)
  {
    CLEAR_FUNCTION(elements[i], 0);
  }

#if 0
  const ushort stepSize = max((ushort)1, (ushort)(sizeof(commonUint16) / sizeof(StructType)));
  commonUint16 localArr(0);
  for (ushort i=0; i<readCount; i+=stepSize)
  {
    if (stepSize > 1)
    {
      localArr = ((Device commonUint16*)&(array1D[indexOffset+i]))[0];
      Thread StructType* localArr2 = (Thread StructType*)&localArr;
      for (ushort j=0; j<stepSize; j++)
      {
        elements[i+j] = ((Thread MemberStructType*)&(localArr2[j]STRUCT_MEMBER))[0];
      }
    }
    else
    {
      elements[i] = ((Device MemberStructType*)&(array1D[indexOffset+i]STRUCT_MEMBER))[0];
    }
  }
#else
  for (ushort i=0; i<readCount; i++)
  {
    elements[i] = ((Device MemberStructType*)&(array1D[indexOffset+i]STRUCT_MEMBER))[0];
  }
#endif

#endif
}

inline static void batchWrite(const Thread MemberStructType *elements, Device StructType* array1D, const uint index, const uint length)
{
  const uint indexOffset = (index << BatchSizeExp);
  const uint writeCount = min(select((uint)0, length - indexOffset, length > indexOffset), (uint)BatchSize);

#if defined(NoMemberStruct) && BatchSize > 1
#pragma message ("Using Batch Write")

#if SHORT_BATCH_RW
  if (writeCount == BatchSize)
  {
#if BatchSize == 16
    *((Device MemberStructType16*)(array1D + indexOffset))  = *((const Thread MemberStructType16*)elements);
#elif BatchSize == 8
    *((Device MemberStructType8*)(array1D + indexOffset))   = *((const Thread MemberStructType8*)elements);
#elif BatchSize == 4
    *((Device MemberStructType4*)(array1D + indexOffset))   = *((const Thread MemberStructType4*)elements);
#else
    for (ushort i=0; i<writeCount; i++)
    {
      ((Device MemberStructType*)&(array1D[indexOffset+i]STRUCT_MEMBER))[0] = elements[i];
    }
#endif
  }
  else
  {
    for (ushort i=0; i<writeCount; i++)
    {
      ((Device MemberStructType*)&(array1D[indexOffset+i]STRUCT_MEMBER))[0] = elements[i];
    }
  }

#else
  switch (writeCount)
  {
#if BatchSize >= 16
  case 16:
  {
    *((Device MemberStructType16*)(array1D + indexOffset))      = *((const Thread MemberStructType16*)elements);
    break;
  }
  case 15:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)(elements));
    *((Device MemberStructType4*)(array1D + indexOffset + 8))   = *((const Thread MemberStructType4*)(elements + 8));
    *((Device MemberStructType2*)(array1D + indexOffset + 12))  = *((const Thread MemberStructType2*)(elements + 12));
    *((Device MemberStructType*)(array1D + indexOffset + 14))   = *((const Thread MemberStructType*)(elements + 14));
    break;
  }
  case 14:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)(elements));
    *((Device MemberStructType4*)(array1D + indexOffset + 8))   = *((const Thread MemberStructType4*)(elements + 8));
    *((Device MemberStructType2*)(array1D + indexOffset + 12))  = *((const Thread MemberStructType2*)(elements + 12));
    break;
  }
  case 13:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)(elements));
    *((Device MemberStructType4*)(array1D + indexOffset + 8))   = *((const Thread MemberStructType4*)(elements + 8));
    *((Device MemberStructType*)(array1D + indexOffset + 12))   = *((const Thread MemberStructType*)(elements + 12));
    break;
  }
  case 12:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)(elements));
    *((Device MemberStructType4*)(array1D + indexOffset + 8))   = *((const Thread MemberStructType4*)(elements + 8));
    break;
  }
  case 11:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)(elements));
    *((Device MemberStructType2*)(array1D + indexOffset + 8))   = *((const Thread MemberStructType2*)(elements + 8));
    *((Device MemberStructType*)(array1D + indexOffset + 10))   = *((const Thread MemberStructType*)(elements + 10));
    break;
  }
  case 10:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)(elements));
    *((Device MemberStructType2*)(array1D + indexOffset + 8))   = *((const Thread MemberStructType2*)(elements + 8));
    break;
  }
  case 9:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)(elements));
    *((Device MemberStructType*)(array1D + indexOffset + 8))    = *((const Thread MemberStructType*)(elements + 8));
    break;
  }
#endif
#if BatchSize >= 8
  case 8:
  {
    *((Device MemberStructType8*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType8*)elements);
    break;
  }
  case 7:
  {
    *((Device MemberStructType4*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType4*)(elements + 0));
    *((Device MemberStructType2*)(array1D + indexOffset + 4))   = *((const Thread MemberStructType2*)(elements + 4));
    *((Device MemberStructType*)(array1D + indexOffset + 6))    = *((const Thread MemberStructType*)(elements + 6));
    break;
  }
  case 6:
  {
    *((Device MemberStructType4*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType4*)(elements + 0));
    *((Device MemberStructType2*)(array1D + indexOffset + 4))   = *((const Thread MemberStructType2*)(elements + 4));
    break;
  }
  case 5:
  {
    *((Device MemberStructType4*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType4*)(elements + 0));
    *((Device MemberStructType*)(array1D + indexOffset + 4))    = *((const Thread MemberStructType*)(elements + 4));
    break;
  }
#endif
  case 4:
  {
    *((Device MemberStructType4*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType4*)elements);
    break;
  }
  case 3:
  {
    *((Device MemberStructType2*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType2*)(elements + 0));
    *((Device MemberStructType*)(array1D + indexOffset + 2))    = *((const Thread MemberStructType*)(elements + 2));
    break;
  }
  case 2:
  {
    *((Device MemberStructType2*)(array1D + indexOffset + 0))   = *((const Thread MemberStructType2*)elements);
    break;
  }
  case 1:
  {
    *((Device MemberStructType*)(array1D + indexOffset + 0))    = *((const Thread MemberStructType*)elements);
    break;
  }
  default:
    break;
  }
#endif

#else

#pragma message ("Using Loop Write")
  if (writeCount)
  {
    ((Device StructType*)(array1D + indexOffset))[0]STRUCT_MEMBER = *elements;
  }

#endif
}

#endif

#endif
