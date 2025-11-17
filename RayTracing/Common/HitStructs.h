/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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
typedef struct ALIGN(8)
{
  float         distance;
  IdentityInfo  primitiveIdentity;
} HitInfoIdentity;


/*!
@struct Hit Info containing distance information.
*/
typedef struct DEFAULT_ALIGN
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          dummy;
} HitInfoDistanceIndexIdentity;


/*!
@struct Hit Info containing distance and BVH information.
*/
typedef struct DEFAULT_ALIGN
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          bvhHits;
} HitInfoDistanceBVHHits;


/*!
@struct Hit Info containing distance and normal information.
*/
typedef struct ALIGN(8) 
{
  float         distance;
  uint          primitiveIndex;
#ifdef COMPUTE_SHADER_SCOPE
  packed_float3 normal;
  IdentityInfo  primitiveIdentity;
#else
  float3        normal;
#endif
} HitInfoDistanceIndexIdentityNormal;


/*!
@struct Hit Info containing distance, normal and UV information.
*/
typedef struct DEFAULT_ALIGN
{
  float         distance;
  uint          primitiveIndex;
  float         u, v;
#ifdef COMPUTE_SHADER_SCOPE
  packed_float3 normal;
  IdentityInfo  primitiveIdentity;
#else
  float3        normal;
#endif
} HitInfoDistanceIndexIdentityNormalUV;


#ifdef COMPUTE_SHADER_SCOPE

#define setHitDistance(hitDistance, time) hitDistance = time

#ifdef HitStructIndex
#define setHitPrimitiveIndex(hitPrimitiveIndex, index)              hitPrimitiveIndex = ((index & 0x7FFFFFFF) | (hitPrimitiveIndex & 0x80000000))
#define getHitPrimitiveIndex(hitPrimitiveIndex)                     (hitPrimitiveIndex & 0x7FFFFFFF)
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index) primitiveInternalIndex = ((primitiveInternalIndex & 0x7FFFFFFF) | (index << 31))
#define getHitPrimitiveInternalIndex(primitiveInternalIndex)        (primitiveInternalIndex >> 31)
#else
#define setHitPrimitiveIndex(hitPrimitiveIndex, index)
#define getHitPrimitiveIndex(hitPrimitiveIndex)
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index)
#define getHitPrimitiveInternalIndex(primitiveInternalIndex)
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

inline HitStruct defaultHit(const float maxDistance)
{
  HitStruct hit;
  setHitDistance(hit.distance, maxDistance);
  setHitPrimitiveIndex(hit.primitiveIndex, -1);
  setHitPrimitiveInternalIndex(hit.primitiveIndex, 1);
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
