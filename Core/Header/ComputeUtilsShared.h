#ifndef COMPUTE_UTILS_SHARED_H
#define COMPUTE_UTILS_SHARED_H

#define PREFIX_SCAN_STATUS_INVALID  0
#define PREFIX_SCAN_STATUS_PARTIAL  1
#define PREFIX_SCAN_STATUS_FINAL    2

#define REDUCE_STATUS_INVALID       0
#define REDUCE_STATUS_PARTIAL       1
#define REDUCE_STATUS_FINAL         2

#ifdef COMPUTE_SHADER_SCOPE

#ifdef StructMember
#define STRUCT_MEMBER           .StructMember
#else
#define STRUCT_MEMBER
#endif

#ifndef MemberStructType
#define MemberStructType        StructType
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

// general atomics for structures
inline MemberStructType atomicLoadN(volatile Device MemberStructType* x)
{
  MemberStructType ret;
  for (int i = 0; i < sizeof(MemberStructType); i += 4)
  {
    ((Thread uint*)&ret)[i] = atomicLoad(((volatile Device uint*)x) + i);
  }
  return ret;
}

inline void atomicStoreN(volatile Device MemberStructType* x, const MemberStructType y)
{
  for (int i = 0; i < sizeof(MemberStructType); i += 4)
  {
    atomicStore(((volatile Device uint*)x) + i, ((const Thread uint*)&y)[i]);
  }
}


#define _STR(x) #x
#define STR(x) _STR(x)

#pragma message ("Struct type: " STR(StructType) ", Member Struct type: " STR(MemberStructType))

// optimized atomics for primitives

#if MemberStructType == XAB
#define ATOMIC_LOAD_FUNCTION(x)     atomicLoadN(x)
#define ATOMIC_STORE_FUNCTION(x, y) atomicStoreN(x, y)

#elif MemberStructType == uint
#define ATOMIC_LOAD_FUNCTION(x)     atomicLoad((volatile Device uint*)(x))
#define ATOMIC_STORE_FUNCTION(x, y) atomicStore((volatile Device uint*)(x), *((uint*)&(y)))

#elif MemberStructType == int
#define ATOMIC_LOAD_FUNCTION(x)     atomicLoad((volatile Device int*)(x))
#define ATOMIC_STORE_FUNCTION(x, y) atomicStore((volatile Device int*)(x), *((int*)&(y)))

#elif MemberStructType == float
#define ATOMIC_LOAD_FUNCTION(x)     as_float(atomicLoad((volatile Device uint*)(x)))
#define ATOMIC_STORE_FUNCTION(x, y) atomicStore((volatile Device uint*)(x), *((uint*)&(y)))

#endif

#define COMPUTE_MAX_THREADS         MaxWorkgroupSize
#define REDUCE_COMPUTE_THREADS      MaxWorkgroupSize
#define PREFIX_SCAN_COMPUTE_THREADS MaxWorkgroupSize

#if MemberStructType == uint
#define MemberStructType16 uint16
#define MemberStructType8 uint8
#define MemberStructType4 uint4
#define MemberStructType2 uint2
#elif MemberStructType == int
#define MemberStructType16 int16
#define MemberStructType8 int8
#define MemberStructType4 int4
#define MemberStructType2 int2
#elif MemberStructType == float
#define MemberStructType16 float16
#define MemberStructType8 float8
#define MemberStructType4 float4
#define MemberStructType2 float2
#endif

// function to write to a memory and wait until the written data is visible
// this helps to get consistent write memory ordering on AMD GPU
void writeAndWait(volatile Device MemberStructType* location, const MemberStructType value)
{
  //ATOMIC_STORE_FUNCTION(location, value);
  //COPY_FUNCTION(*location, value);

  MemberStructType temp;
  do
  {
    ATOMIC_STORE_FUNCTION(location, value);
    temp = ATOMIC_LOAD_FUNCTION(location);
    //COPY_FUNCTION(temp, *location);
  }
  while (((Thread uint*)&temp)[0] != ((Thread uint*)&value)[0]);
}

MemberStructType localReduce(const Thread MemberStructType *elements)
{
  MemberStructType ret = elements[0];
  for (uint i = 1; i < BatchSize; i++)
  {
    ADD_FUNCTION(ret, elements[i]);
  }
  return ret;
}

void batchRead(Thread MemberStructType *elements, const Device StructType* array1D, const uint index, const uint length)
{
  const uint indexOffset = index * BatchSize;
  const uint readCount = min((length > indexOffset) ? length - indexOffset : 0, (uint)BatchSize);

#if MemberStructType == StructType
  Thread MemberStructType *data = elements;
#else
  StructType data[BatchSize];
#endif

#if BatchSize == 16
  * ((Thread MemberStructType16*)elements) = (MemberStructType16)(0);
#elif BatchSize == 8
  * ((Thread MemberStructType8*)elements) = (MemberStructType8)(0);
#elif BatchSize == 4
  * ((Thread MemberStructType4*)elements) = (MemberStructType4)(0);
#elif BatchSize == 1
  CLEAR_FUNCTION(*elements, 0);
#endif

  switch (readCount)
  {
#if BatchSize >= 16
  case 16:
  {
    *((Thread MemberStructType16*)data)       = *((const Device MemberStructType16*)(array1D + indexOffset));
    break;
  }
  case 15:
  {
    *((Thread MemberStructType8*)(data + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(data + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    *((Thread MemberStructType2*)(data + 12)) = *((const Device MemberStructType2*)(array1D + indexOffset + 12));
    *((Thread MemberStructType*)(data + 14))  = *((const Device MemberStructType*)(array1D + indexOffset + 14));
    break;
  }
  case 14:
  {
    *((Thread MemberStructType8*)(data + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(data + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    *((Thread MemberStructType2*)(data + 12)) = *((const Device MemberStructType2*)(array1D + indexOffset + 12));
    break;
  }
  case 13:
  {
    *((Thread MemberStructType8*)(data + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(data + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    *((Thread MemberStructType*)(data + 12))  = *((const Device MemberStructType*)(array1D + indexOffset + 12));
    break;
  }
  case 12:
  {
    *((Thread MemberStructType8*)(data + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType4*)(data + 8))  = *((const Device MemberStructType4*)(array1D + indexOffset + 8));
    break;
  }
  case 11:
  {
    *((Thread MemberStructType8*)(data + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(data + 8))  = *((const Device MemberStructType2*)(array1D + indexOffset + 8));
    *((Thread MemberStructType*)(data + 10))  = *((const Device MemberStructType*)(array1D + indexOffset + 10));
    break;
  }
  case 10:
  {
    *((Thread MemberStructType8*)(data + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(data + 8))  = *((const Device MemberStructType2*)(array1D + indexOffset + 8));
    break;
  }
  case 9:
  {
    *((Thread MemberStructType8*)(data + 0))  = *((const Device MemberStructType8*)(array1D + indexOffset));
    *((Thread MemberStructType*)(data + 8))   = *((const Device MemberStructType*)(array1D + indexOffset + 8));
    break;
  }
#endif
#if BatchSize >= 8
  case 8:
  {
    *((Thread MemberStructType8*)data + 0)    = *((const Device MemberStructType8*)(array1D + indexOffset));
    break;
  }
  case 7:
  {
    *((Thread MemberStructType4*)(data + 0))  = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(data + 4))  = *((const Device MemberStructType2*)(array1D + indexOffset + 4));
    *((Thread MemberStructType*)(data + 6))   = *((const Device MemberStructType*)(array1D + indexOffset + 6));
    break;
  }
  case 6:
  {
    *((Thread MemberStructType4*)(data + 0))  = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(data + 4))  = *((const Device MemberStructType2*)(array1D + indexOffset + 4));
    break;
  }
  case 5:
  {
    *((Thread MemberStructType4*)(data + 0))  = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType*)(data + 4))   = *((const Device MemberStructType*)(array1D + indexOffset + 4));
    break;
  }
#endif
  case 4:
  {
    *((Thread MemberStructType4*)data + 0)    = *((const Device MemberStructType4*)(array1D + indexOffset));
    break;
  }
  case 3:
  {
    *((Thread MemberStructType2*)(data + 0))  = *((const Device MemberStructType2*)(array1D + indexOffset));
    *((Thread MemberStructType*)(data + 2))   = *((const Device MemberStructType*)(array1D + indexOffset + 2));
    break;
  }
  case 2:
  {
    *((Thread MemberStructType2*)data + 0)    = *((const Device MemberStructType2*)(array1D + indexOffset));
    break;
  }
  case 1:
  {
    *((Thread MemberStructType*)data + 0)     = *((const Device MemberStructType*)(array1D + indexOffset));
    break;
  }
  default:
    break;
  }

#if MemberStructType != StructType

  for (uint i = 0; i < writeCount; i++)
  {
    elements[i] = data[i]STRUCT_MEMBER;
  }

#endif
}

void batchWrite(const MemberStructType *elements, Device StructType* array1D, const uint index, const uint length)
{
  const uint indexOffset = index * BatchSize;
  const uint writeCount = min((length > indexOffset) ? length - indexOffset : 0, (uint)BatchSize);

#if MemberStructType == StructType

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

#else

  for (uint i = 0; i < writeCount; i++)
  {
    ((Device StructType*)(array1D + indexOffset))[i]STRUCT_MEMBER = elements[i];
  }

#endif
}

#endif

#endif
