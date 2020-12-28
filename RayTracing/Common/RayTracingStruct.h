#ifndef RAY_TRACING_STRUCT_H
#define RAY_TRACING_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include "RayStructs.h"
#include "HitStructs.h"
#include "MaterialStruct.h"
#endif

#pragma pack(push, 4)

#define RAY_TRACING_PRIM_TYPE_ID_SHIFT    28
#define RAY_TRACING_PRIM_OFFSET_MASK      0x0FFFFFFF

#define RAY_TRACING_ENTITY_TYPE_ID_MASK   0xF0000000
#define RAY_TRACING_ENTITY_TYPE_ID_SHIFT  28
#define RAY_TRACING_ENTITY_ID_MASK        0x0FFF0000
#define RAY_TRACING_ENTITY_ID_SHIFT       16
#define RAY_TRACING_INSTANCE_ID_MASK      0x0000FFFF

#ifdef COMPUTE_SHADER_SCOPE

#define MIN_TIME 0.0001f

// also known as slabs method
inline bool rayXABIntersectTest(const float timeIn, const XAB xab, const float3 rayOrigin, const float3 invRayDirection, const bool3 sign)
{
  const float3 t0 = (xab.min - rayOrigin) * invRayDirection;
  const float3 t1 = (xab.max - rayOrigin) * invRayDirection;
  const float3 tmin = select(t0, t1, sign);
  const float tminOut = maxCompFloat3(tmin);

  if (tminOut >= timeIn)
  {
    return false;
  }
  const float tmaxOut = minCompFloat3(select(t1, t0, sign));
  return tminOut < tmaxOut && tmaxOut > MIN_TIME;
}

inline bool rayXABIntersectInOut(Thread float* timeIn, Thread float* timeOut, const XAB xab, const float3 rayOrigin, const float3 invRayDirection, const bool3 sign)
{
  const float3 t0 = (xab.min - rayOrigin) * invRayDirection;
  const float3 t1 = (xab.max - rayOrigin) * invRayDirection;
  const float3 tmin = select(t0, t1, sign);
  const float3 tmax = select(t1, t0, sign);
  const float tmaxOut = minCompFloat3(tmax);
  const float tminOut = maxCompFloat3(tmin);

  if (tmaxOut > MIN_TIME && tminOut < *timeIn && tminOut < tmaxOut)
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

  if (*timeIn > tminOut)
  {
    const float3 tmax = select(t1, t0, sign);
    const float tmaxOut = minCompFloat3(tmax);
    if (tminOut < tmaxOut && tmaxOut > MIN_TIME)
    {
      *timeIn  = tminOut;
      return true;
    }
  }
  return false;
}

#endif


struct ALIGN(8) PackingInfo_t
{
  uint   elementOffset;
  ushort strideIn4Bytes;
  ushort offsetIn4Bytes;

#ifndef COMPUTE_SHADER_SCOPE
  PackingInfo_t(uint elementOffset = 0, ushort strideIn4Bytes = 0, ushort offsetIn4Bytes = 0)
  {
    this->elementOffset  = elementOffset;
    this->strideIn4Bytes = strideIn4Bytes;
    this->offsetIn4Bytes = offsetIn4Bytes;
  }
#endif
};

typedef struct PackingInfo_t PackingInfo;


typedef struct IdentityInfo_t RayTracingEntityId;


/*!
@struct Base data for a ray traced primitive.
@note   Should have same structure as PositionStruct. May cause issues otherwise.
*/
struct DEFAULT_ALIGN PrimitiveStruct_t
{
  union
  {
    struct
    {
      float3  position;
    };
    struct
    {
      uint          reserved[3];
      IdentityInfo  identity;
    };
  };
};

typedef struct PrimitiveStruct_t PrimitiveStruct;

#ifndef COMPUTE_SHADER_SCOPE
inline static void setRayTracingEntityId(IdentityInfo& identity, uint entityType, uint entityId)
{
  identity.identity = (identity.identity & RAY_TRACING_INSTANCE_ID_MASK) |
    ((entityType << RAY_TRACING_ENTITY_TYPE_ID_SHIFT) & RAY_TRACING_ENTITY_TYPE_ID_MASK) |
    ((entityId << RAY_TRACING_ENTITY_ID_SHIFT) & RAY_TRACING_ENTITY_ID_MASK);
}

inline static void setRayTracingInstanceId(IdentityInfo& identity, uint instanceId)
{
  identity.identity = (identity.identity & (-1 ^ RAY_TRACING_INSTANCE_ID_MASK)) | (instanceId & RAY_TRACING_INSTANCE_ID_MASK);
}

inline static void setIdentityTwoSided(IdentityInfo& identity, const bool twoSidedFlag)
{
  identity.identity = (identity.identity & 0x7FFFFFFF) | (twoSidedFlag?0x80000000:0);
}

inline static void setIdentityEntityNoShadow(IdentityInfo& identity, const bool noShadow)
{
  identity.identity = (identity.identity & 0xBFFFFFFF) | (noShadow?0x40000000:0);
}
#endif

inline static uint getRayTracingInstanceId(const IdentityInfo identity)
{
  return identity.identity & RAY_TRACING_INSTANCE_ID_MASK;
}

inline static ushort getRayTracingEntityId(const IdentityInfo identity)
{
  return (identity.identity & RAY_TRACING_ENTITY_ID_MASK) >> RAY_TRACING_ENTITY_ID_SHIFT;
}

inline static ushort getRayTracingEntityType(const IdentityInfo identity)
{
  return (identity.identity & RAY_TRACING_ENTITY_TYPE_ID_MASK) >> RAY_TRACING_ENTITY_TYPE_ID_SHIFT;
}

inline static bool isIdentityEntityTwoSided(const IdentityInfo identity)
{
  return identity.identity & 0x80000000;
}

inline static bool isIdentityEntityNoShadow(const IdentityInfo identity)
{
  return identity.identity & 0x40000000;
}

inline IdentityInfo removeIdentityFlags(const IdentityInfo identity)
{
  IdentityInfo ret = identity;
  ret.identity &= 0x3FFFFFFF;
  return ret;
}


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
  uint indexOffset;
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
    uint primOffset;
    uint primCount;
  };
  union
  {
    uint indexOffset;
    uint indexCount;
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
  sys.primTypeAndIndexOffset = (sys.primTypeAndIndexOffset & RAY_TRACING_PRIM_OFFSET_MASK) | (type << RAY_TRACING_PRIM_TYPE_ID_SHIFT);
}

inline static void setPrimitiveIndexOffset(EncodedPrimitiveInfo& sys, uint offset)
{
  sys.primTypeAndIndexOffset = (sys.primTypeAndIndexOffset & (-1 ^ RAY_TRACING_PRIM_OFFSET_MASK)) | (offset & RAY_TRACING_PRIM_OFFSET_MASK);
}

inline static void setPrimitiveVertexOffset(EncodedPrimitiveInfo& sys, uint offset)
{
  sys.indexOffset = offset;
}

#endif

inline static DecodedPrimitiveInfo decodePrimitiveInfo(const EncodedPrimitiveInfo primInfo)
{
  DecodedPrimitiveInfo ret;
  ret.primType    = (ushort)(primInfo.primTypeAndIndexOffset >> RAY_TRACING_PRIM_TYPE_ID_SHIFT);
  ret.primOffset  = primInfo.primTypeAndIndexOffset & RAY_TRACING_PRIM_OFFSET_MASK;
  ret.indexOffset = primInfo.indexOffset;
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

    if (index < primInfo.primOffset)
    {
      return primInfo;
    }
  }

  DecodedPrimitiveInfo ret;
  ret.primType    = -1;
  ret.primOffset  = -1;
  ret.indexOffset = -1;
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
    IdentityInfo identity;
  };
#if defined(COMPUTE_SHADER_SCOPE)
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
  union
  {
    struct
    {
      float3 position;
    };
    struct
    {
      uint res1[3];
      IdentityInfo identity;
    };
  };
  union
  {
    struct
    {
      float3 color;
    };
    struct
    {
      float res2[3];
      float width;
    };
  };
  union
  {
    struct
    {
      float3 normal;
    };
    struct
    {
      float res3[3];
      float height;
    };
  };
};

typedef struct LightStruct_t LightStruct;

#pragma pack(pop)

#endif
