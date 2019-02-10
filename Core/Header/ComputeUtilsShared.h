#ifndef COMPUTE_UTILS_SHARED_H
#define COMPUTE_UTILS_SHARED_H

#define PREFIX_SCAN_STATUS_INVALID  0
#define PREFIX_SCAN_STATUS_PARTIAL  1
#define PREFIX_SCAN_STATUS_FINAL    2
#define PREFIX_SCAN_COMPUTE_THREADS 1024

#define REDUCE_STATUS_INVALID       0
#define REDUCE_STATUS_PARTIAL       1
#define REDUCE_STATUS_FINAL         2
#define REDUCE_COMPUTE_THREADS      1024

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

#if MemberStructType == uint
#define MemberStructType8 uint8
#define MemberStructType4 uint4
#define MemberStructType2 uint2
#elif MemberStructType == int
#define MemberStructType8 int8
#define MemberStructType4 int4
#define MemberStructType2 int2
#elif MemberStructType == float
#define MemberStructType8 float8
#define MemberStructType4 float4
#define MemberStructType2 float2
#endif

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

#if BatchSize == 8
  * ((Thread MemberStructType8*)elements) = (MemberStructType8)(0);
#elif BatchSize == 4
  * ((Thread MemberStructType4*)elements) = (MemberStructType4)(0);
#endif

  switch (readCount)
  {
#if BatchSize >= 8
  case 8:
  {
    *((Thread MemberStructType8*)data) = *((const Device MemberStructType8*)(array1D + indexOffset));
    break;
  }
  case 7:
  {
    *((Thread MemberStructType4*)(data + 0)) = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(data + 4)) = *((const Device MemberStructType2*)(array1D + indexOffset + 4));
    *((Thread MemberStructType*)(data + 6)) = *((const Device MemberStructType*)(array1D + indexOffset + 6));
    break;
  }
  case 6:
  {
    *((Thread MemberStructType4*)(data + 0)) = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType2*)(data + 4)) = *((const Device MemberStructType2*)(array1D + indexOffset + 4));
    break;
  }
  case 5:
  {
    *((Thread MemberStructType4*)(data + 0)) = *((const Device MemberStructType4*)(array1D + indexOffset));
    *((Thread MemberStructType*)(data + 4)) = *((const Device MemberStructType*)(array1D + indexOffset + 4));
    break;
  }
#endif
  case  4:
  {
    *((Thread MemberStructType4*)data) = *((const Device MemberStructType4*)(array1D + indexOffset));
    break;
  }
  case 3:
  {
    *((Thread MemberStructType2*)(data + 0)) = *((const Device MemberStructType2*)(array1D + indexOffset));
    *((Thread MemberStructType*)(data + 2)) = *((const Device MemberStructType*)(array1D + indexOffset + 2));
    break;
  }
  case 2:
  {
    *((Thread MemberStructType2*)data) = *((const Device MemberStructType2*)(array1D + indexOffset));
    break;
  }
  case 1:
  {
    *((Thread MemberStructType*)data) = *((const Device MemberStructType*)(array1D + indexOffset));
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
#if BatchSize >= 8
  case 8:
  {
    *((Device MemberStructType8*)(array1D + indexOffset)) = *((const Thread MemberStructType8*)elements);
    break;
  }
  case 7:
  {
    *((Device MemberStructType4*)(array1D + indexOffset)) = *((const Thread MemberStructType4*)(elements + 0));
    *((Device MemberStructType2*)(array1D + indexOffset + 4)) = *((const Thread MemberStructType2*)(elements + 4));
    *((Device MemberStructType*)(array1D + indexOffset + 6)) = *((const Thread MemberStructType*)(elements + 6));
    break;
  }
  case 6:
  {
    *((Device MemberStructType4*)(array1D + indexOffset)) = *((const Thread MemberStructType4*)(elements + 0));
    *((Device MemberStructType2*)(array1D + indexOffset + 4)) = *((const Thread MemberStructType2*)(elements + 4));
    break;
  }
  case 5:
  {
    *((Device MemberStructType4*)(array1D + indexOffset)) = *((const Thread MemberStructType4*)(elements + 0));
    *((Device MemberStructType*)(array1D + indexOffset + 4)) = *((const Thread MemberStructType*)(elements + 4));
    break;
  }
#endif
  case 4:
  {
    *((Device MemberStructType4*)(array1D + indexOffset)) = *((const Thread MemberStructType4*)elements);
    break;
  }
  case 3:
  {
    *((Device MemberStructType2*)(array1D + indexOffset)) = *((const Thread MemberStructType2*)(elements + 0));
    *((Device MemberStructType*)(array1D + indexOffset + 2)) = *((const Thread MemberStructType*)(elements + 2));
    break;
  }
  case 2:
  {
    *((Device MemberStructType2*)(array1D + indexOffset)) = *((const Thread MemberStructType2*)elements);
    break;
  }
  case 1:
  {
    *((Device MemberStructType*)(array1D + indexOffset)) = *((const Thread MemberStructType*)elements);
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