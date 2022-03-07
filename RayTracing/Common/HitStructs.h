#ifndef HIT_STRUCTS_H
#define HIT_STRUCTS_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#pragma pack(push, 4)

enum HitStructType
{
  HitStructDistanceIdentity,
  HitStructDistanceIndexIdentity,
  HitStructDistanceBVHHits,
  HitStructDistanceIndexIdentityNormal,
  HitStructDistanceIndexIdentityNormalUV,
  HitStructTypeMax
};


/*!
@struct Hit Info containing distance information.
*/
struct ALIGN(8) HitInfoIdentity
{
  float         distance;
  IdentityInfo  primitiveIdentity;
};


/*!
@struct Hit Info containing distance information.
*/
struct DEFAULT_ALIGN HitInfoDistanceIndexIdentity
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          primitiveInternalIndex;
};


/*!
@struct Hit Info containing distance and BVH information.
*/
struct DEFAULT_ALIGN HitInfoDistanceBVHHits
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          bvhHits;
};


/*!
@struct Hit Info containing distance and normal information.
*/
struct DEFAULT_ALIGN HitInfoDistanceIndexIdentityNormal
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          primitiveInternalIndex;
  float3        normal;
};


/*!
@struct Hit Info containing distance, normal and UV information.
*/
struct DEFAULT_ALIGN HitInfoDistanceIndexIdentityNormalUV
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          primitiveInternalIndex;
  float3        normal;
  float         u, v;
};


#ifdef COMPUTE_SHADER_SCOPE

#define setHitDistance(hitDistance, time) hitDistance = time

#ifdef HitStructIndex
#define setHitPrimitiveIndex(hitPrimitiveIndex, index) hitPrimitiveIndex = index
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index) primitiveInternalIndex = index
#else
#define setHitPrimitiveIndex(hitPrimitiveIndex, index)
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index)
#endif

#ifdef HitStructIdentity
#define setHitPrimitiveIdentity(hitPrimitiveIdentity, identity) hitPrimitiveIdentity = identity
#else
#define setHitPrimitiveIdentity(hitPrimitiveIdentity, identity)
#endif

#ifdef HitStructNormal
#define setHitNormal(hitNormal, normal) hitNormal = normal
#else
#define setHitNormal(hitNormal, normal)
#endif

#ifdef HitStructBVHHits
#define addBVHHit(hitBVHHits, hitCount) hitBVHHits += hitCount
#define setBVHHit(hitBVHHits, hitCount) hitBVHHits = hitCount
#else
#define addBVHHit(hitBVHHits, hitCount)
#define setBVHHit(hitBVHHits, hitCount)
#endif

#ifdef HitStructUV
#define setHitUV(hitUV, uv) hitUV = uv
#else
#define setHitUV(hitUV, uv)
#endif

inline HitStruct defaultHit(const float maxDistance = INFINITY)
{
  HitStruct hit;
  setHitDistance(hit.distance, maxDistance);
  setHitPrimitiveIndex(hit.primitiveIndex, -1);
  setHitPrimitiveIdentity(hit.primitiveIdentity.identity, -1);
  setBVHHit(hit.bvhHits, 0);
  return hit;
}

#else

extern uint getHitStructSize(HitStructType type);

extern string getHitStructName(HitStructType type);

extern void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type);

#endif

#pragma pack(pop)

#endif
