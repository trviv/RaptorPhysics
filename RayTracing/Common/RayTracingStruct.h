#ifndef RAY_TRACING_STRUCT_H
#define RAY_TRACING_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include "RayStructs.h"
#include "HitStructs.h"
#endif

#pragma pack(push, 4)

#define RAY_TRACING_TYPE_ID_MASK      0xF0000000
#define RAY_TRACING_TYPE_ID_SHIFT     28
#define RAY_TRACING_PRIM_OFFSET_MASK  0x0FFFFFFF

#ifdef COMPUTE_SHADER_SCOPE

#define MIN_TIME COMPUTE_EPSILON

// also known as slabs method
inline bool rayXABIntersectTest(const XAB xab, const float3 rayOrigin, const float3 invRayDirection)
{
  const float3 t0 = (xab.min - rayOrigin) * invRayDirection;
  const float3 t1 = (xab.max - rayOrigin) * invRayDirection;
  const float3 tmin = min(t0, t1), tmax = max(t0, t1);
  return maxCompFloat3(tmin) <= minCompFloat3(tmax);
}

inline bool rayXABIntersectInOut(Thread float* timeIn, Thread float* timeOut, const XAB xab, const float3 rayOrigin, const float3 invRayDirection, const bool3 sign)
{
  const float3 t0 = (xab.min - rayOrigin) * invRayDirection;
  const float3 t1 = (xab.max - rayOrigin) * invRayDirection;
  const float3 tmin = select(t0, t1, sign);
  const float3 tmax = select(t1, t0, sign);
  const float tmaxOut = minCompFloat3(tmax);
  const float tminOut = maxCompFloat3(tmin);

  if (tminOut > MIN_TIME && tminOut <= tmaxOut)
  {
    *timeIn  = tminOut;
    *timeOut = tmaxOut;
    return true;
  }
  return false;
}

inline bool rayXABIntersectEarliest(Thread float* timeIn, const XAB xab, const float3 rayOrigin, const float3 invRayDirection, const bool3 sign)
{
  const float3 t0 = (xab.min - rayOrigin) * invRayDirection;
  const float3 t1 = (xab.max - rayOrigin) * invRayDirection;
  const float3 tmin = select(t0, t1, sign);
  const float tminOut = maxCompFloat3(tmin);

  if (tminOut > MIN_TIME && *timeIn > tminOut)
  {
    const float3 tmax = select(t1, t0, sign);
    const float tmaxOut = minCompFloat3(tmax);
    if (tminOut <= tmaxOut)
    {
      *timeIn  = tminOut;
      return true;
    }
  }
  return false;
}


// Returns the i'th element of the Halton sequence using the d'th prime number as a
// base. The Halton sequence is a "low discrepency" sequence: the values appear
// random but are more evenly distributed then a purely random sequence. Each random
// value used to render the image should use a different independent dimension 'd',
// and each sample (frame) should use a different index 'i'. To decorrelate each
// pixel, a random offset can be applied to 'i'.
float halton(Const uint* primes, uint i, uint d)
{
  uint b = primes[d];

  float f = 1.0f;
  float invB = 1.0f / b;
  float r = 0;

  while (i > 0)
  {
    f = f * invB;
    r = r + f * (i % b);
    i = i / b;
  }

  return r;
}

#endif


struct PackingInfo_t
{
  ushort strideIn4Bytes;
  ushort offsetIn4Bytes;

#ifndef COMPUTE_SHADER_SCOPE
  PackingInfo_t(ushort strideIn4Bytes = 0, ushort offsetIn4Bytes = 0)
  {
    this->strideIn4Bytes = strideIn4Bytes;
    this->offsetIn4Bytes = offsetIn4Bytes;
  }
#endif
};

typedef struct PackingInfo_t PackingInfo;


/*!
@struct Base data for a ray traced primitive.
@note   Should have same structure as PositionStruct. May cause issues otherwise.
*/
struct DEFAULT_ALIGN PrimitiveStruct_t
{
  union
  {
    float3  position;
    struct
    {
      uint  reserved[3];
      uint  identity;
    };
  };
};

typedef struct PrimitiveStruct_t PrimitiveStruct;


struct ALIGN(4) PrimitiveAttrib_t
{
  float radius;
};

typedef struct PrimitiveAttrib_t PrimitiveAttrib;


enum RTPrimitiveType
{
  PrimitiveSphere,
  PrimitiveTriangle,
  RTPrimitiveCount
};


/*!
@struct Encoded primitive offsets.
*/
struct ALIGN(4) EncodedPrimitiveInfo_t
{
  union
  {
    uint primTypeAndIndexOffset;
    uint count;
  };
  uint vertexOffset;
};

typedef struct EncodedPrimitiveInfo_t EncodedPrimitiveInfo;


/*!
@struct Decoded primitive offsets.
*/
struct DecodedPrimitiveInfo_t
{
  ushort primType;
  union
  {
    uint indexOffset;
    uint indexCount;
  };
  union
  {
    uint vertexOffset;
    uint vertexCount;
  };
};

typedef struct DecodedPrimitiveInfo_t DecodedPrimitiveInfo;


struct DEFAULT_ALIGN RTSystemSettings_t
{
  XAB systemBound;
  EncodedPrimitiveInfo globalOffsets[RTPrimitiveCount];
};

typedef struct RTSystemSettings_t RTSystemSettings;


#ifndef COMPUTE_SHADER_SCOPE

inline static void setPrimitiveType(EncodedPrimitiveInfo& sys, RTPrimitiveType type)
{
  sys.primTypeAndIndexOffset = (sys.primTypeAndIndexOffset & RAY_TRACING_PRIM_OFFSET_MASK) | (type << RAY_TRACING_TYPE_ID_SHIFT);
}

inline static void setPrimitiveIndexOffset(EncodedPrimitiveInfo& sys, uint offset)
{
  sys.primTypeAndIndexOffset = (sys.primTypeAndIndexOffset & (-1 ^ RAY_TRACING_PRIM_OFFSET_MASK)) | (offset & RAY_TRACING_PRIM_OFFSET_MASK);
}

inline static void setPrimitiveVertexOffset(EncodedPrimitiveInfo& sys, uint offset)
{
  sys.vertexOffset = offset;
}

#endif

inline static DecodedPrimitiveInfo decodePrimitiveInfo(const EncodedPrimitiveInfo primInfo)
{
  const DecodedPrimitiveInfo ret = {
    (ushort)(primInfo.primTypeAndIndexOffset >> RAY_TRACING_TYPE_ID_SHIFT),
    primInfo.primTypeAndIndexOffset & RAY_TRACING_PRIM_OFFSET_MASK,
    primInfo.vertexOffset
  };
  return ret;
}

#ifdef COMPUTE_SHADER_SCOPE

float3 extractPackedFloat3(const Device float* buffer, const PackingInfo packingInfo, const uint index)
{
  return *((Device float3*)(buffer + index * packingInfo.strideIn4Bytes + packingInfo.offsetIn4Bytes));
}

float extractPackedFloat(const Device float* buffer, const PackingInfo packingInfo, const uint index)
{
  return buffer[index * packingInfo.strideIn4Bytes + packingInfo.offsetIn4Bytes];
}

inline static DecodedPrimitiveInfo decodePrimitiveInfoFromSystemSettings(Const RTSystemSettings* systemSettings, const uint index)
{
  for (ushort i=0; i<RTPrimitiveCount; i++)
  {
    const DecodedPrimitiveInfo primInfo = decodePrimitiveInfo(systemSettings->globalOffsets[i]);

    if (index < primInfo.indexOffset)
    {
      return primInfo;
    }
  }

  DecodedPrimitiveInfo ret;
  ret.primType     = -1;
  ret.indexOffset  = -1;
  ret.vertexOffset = -1;
  return ret;
}

#endif


/*!
@struct Shared Camera information.
*/
struct DEFAULT_ALIGN CameraStruct_t
{
  struct
  {
    uint  width;
    uint  height;
    float scale;
    uint  padding;
  };
#if defined(COMPUTE_SHADER_SCOPE) && defined(USE_METAL_COMPUTE)
  float4x4  viewMatrixInv;
#else
  float     viewMatrixInv[16];
#endif
};

typedef struct CameraStruct_t CameraStruct;


/*!
@struct Shared Light information.
*/
struct DEFAULT_ALIGN LightStruct_t
{
  Color3 color;
};

typedef struct LightStruct_t LightStruct;

#pragma pack(pop)

#endif
