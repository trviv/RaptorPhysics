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
  const float tmaxOut = minComp3(select(t1, t0, sign));
  if (tmaxOut <= MIN_TIME)
  {
    return false;
  }
  const float tminOut = maxComp3(select(t0, t1, sign));
  return tminOut < timeIn && tminOut < tmaxOut;
}

inline bool rayXABIntersectInOut(Thread float* timeIn, Thread float* timeOut, const XAB xab, const float3 rayOrigin, const float3 invRayDirection, const bool3 sign)
{
  const float3 t0 = (xab.min - rayOrigin) * invRayDirection;
  const float3 t1 = (xab.max - rayOrigin) * invRayDirection;
  const float3 tmin = select(t0, t1, sign);
  const float3 tmax = select(t1, t0, sign);
  const float tmaxOut = minComp3(tmax);
  const float tminOut = maxComp3(tmin);

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
  const float tminOut = maxComp3(tmin);

  if (*timeIn > tminOut)
  {
    const float3 tmax = select(t1, t0, sign);
    const float tmaxOut = minComp3(tmax);
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

#ifndef COMPUTE_SHADER_SCOPE
  PackingInfo_t(uint elementOffset = 0, ushort strideIn4Bytes = 0, ushort offsetIn4Bytes = 0)
  {
    this->elementOffset  = elementOffset + offsetIn4Bytes;
    this->strideIn4Bytes = strideIn4Bytes;
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

inline static void setIdentityEntityFlat(IdentityInfo& identity, const bool flatNormal)
{
  identity.identity = (identity.identity & 0xDFFFFFFF) | (flatNormal?0x20000000:0);
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

inline static bool isIdentityEntityFlat(const IdentityInfo identity)
{
  return identity.identity & 0x20000000;
}

inline IdentityInfo removeIdentityFlags(const IdentityInfo identity)
{
  IdentityInfo ret = identity;
  ret.identity &= 0x1FFFFFFF;
  return ret;
}


struct DEFAULT_ALIGN PrimitiveAttrib_t
{
#ifdef COMPUTE_SHADER_SCOPE
  union
  {
    struct
    {
      uint3 triangleIndex;
    };
    struct
    {
      uint4 quadIndex;
    };
    struct
    {
      int   dummy[3];
      float radius;
    };
  };
#else
  union
  {
    struct
    {
      uint  triangleIndex[3];
      float radius;
    };
    struct
    {
      uint  quadIndex[4];
    };
  };
#endif
};

typedef struct PrimitiveAttrib_t PrimitiveAttrib;


struct DEFAULT_ALIGN VertexAttrib_t
{
  union
  {
    struct
    {
      float3  normal;
    };
    struct
    {
      int     dummy[3];
      float   radius;
    };
  };
};

typedef struct VertexAttrib_t VertexAttrib;


enum EntityPrimitiveAttributeType
{
  EntityPrimitiveAttributePosition,
  EntityPrimitiveAttributeRadius,
  EntityPrimitiveAttributeIndex = EntityPrimitiveAttributeRadius,
  EntityPrimitiveAttributeNormal,
  EntityPrimitiveAttributeMax
};


enum RayTracingEntityType
{
  RayTracingEntityCamera      = 0,

  RayTracingEntityLight       = 1,
  RayTracingEntityLightPoint  = 1,
  RayTracingEntityLightArea   = 2,

  RayTracingEntityPrimArray   = 4,
  RayTracingEntitySpheres     = 4,
  RayTracingEntityTriangles   = 5,
  RayTracingEntityIndexedTriangles  = 6,
  RayTracingEntityIndexedQuads      = 7
};


enum RTPrimitiveType
{
  PrimitiveSphere,
  PrimitiveIndexedTriangle,
  PrimitiveIndexedQuad,
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
    uint primitiveTypeAndIndexOffset;
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
  ushort primitiveType;
  uint prevPrimitiveOffset;
  union
  {
    uint primitiveOffset;
    uint primitiveCount;
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
  sys.primitiveTypeAndIndexOffset = (sys.primitiveTypeAndIndexOffset & RAY_TRACING_PRIM_OFFSET_MASK) | (type << RAY_TRACING_PRIM_TYPE_ID_SHIFT);
}

inline static void setPrimitiveIndexOffset(EncodedPrimitiveInfo& sys, uint offset)
{
  sys.primitiveTypeAndIndexOffset = (sys.primitiveTypeAndIndexOffset & (-1 ^ RAY_TRACING_PRIM_OFFSET_MASK)) | (offset & RAY_TRACING_PRIM_OFFSET_MASK);
}

inline static void setPrimitiveVertexOffset(EncodedPrimitiveInfo& sys, uint offset)
{
  sys.vertexOffset = offset;
}

#endif

inline static DecodedPrimitiveInfo decodePrimitiveInfo(const EncodedPrimitiveInfo primInfo)
{
  DecodedPrimitiveInfo ret;
  ret.primitiveType   = (ushort)(primInfo.primitiveTypeAndIndexOffset >> RAY_TRACING_PRIM_TYPE_ID_SHIFT);
  ret.primitiveOffset = primInfo.primitiveTypeAndIndexOffset & RAY_TRACING_PRIM_OFFSET_MASK;
  ret.vertexOffset    = primInfo.vertexOffset;
  return ret;
}

#ifdef COMPUTE_SHADER_SCOPE

const Device float* extractPackedPointer(const Device float* buffer, const PackingInfo packingInfo)
{
  return buffer + packingInfo.elementOffset;
}

inline void decodePrimitiveInfoFromSystemSettings(Const RTSystemSettings* systemSettings, const uint index, Thread DecodedPrimitiveInfo* primInfo)
{
  if (index >= primInfo->prevPrimitiveOffset && index < primInfo->primitiveOffset && primInfo->primitiveType < RTPrimitiveCount)
  {
    return;
  }

  uint prevPrimitiveOffset = 0;

  for (ushort i=0; i<RTPrimitiveCount; i++)
  {
    *primInfo = decodePrimitiveInfo(systemSettings->globalOffsets[i]);
    primInfo->prevPrimitiveOffset = prevPrimitiveOffset;

    if (index < primInfo->primitiveOffset)
    {
      return;
    }

    prevPrimitiveOffset = primInfo->primitiveOffset;
  }

  primInfo->primitiveType       = -1;
  primInfo->primitiveOffset     = -1;
  primInfo->vertexOffset        = -1;
  primInfo->prevPrimitiveOffset = 0;
}

#endif


/*!
@struct Shared Camera information.
*/
typedef struct DEFAULT_ALIGN
{
  struct
  {
    uint  width;
    uint  height;
    float scale;
    uint  frameIndex;
  };
#if defined(COMPUTE_SHADER_SCOPE)
  float4x4  viewMatrixInv;
#else
  float     viewMatrixInv[16];
#endif
} CameraStruct;


/*!
@struct Shared Light information.
*/
typedef struct DEFAULT_ALIGN
{
  union
  {
    struct
    {
      float3 position;
      float3 color;
      float3 normal;
      float3 right;
      float3 up;
    };
    struct
    {
      uint res1[3];
      IdentityInfo identity;
      uint res2[4*4];
    };
  };
} LightStruct;

#pragma pack(pop)

#endif
