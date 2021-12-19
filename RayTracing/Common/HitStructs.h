#ifndef HIT_STRUCTS_H
#define HIT_STRUCTS_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#pragma pack(push, 4)

enum HitStructType
{
  HitStructDistanceIndex,
  HitStructDistanceIdentity,
  HitStructDistanceIndexIdentity,
  HitStructDistanceBVHHits,
  HitStructTypeMax
};


/*!
@struct Hit Info containing distance information.
*/
typedef struct ALIGN(8)
{
  float distance;
  uint  primitiveIndex;
} HitInfoIndex;


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
struct DEFAULT_ALIGN HitInfoDistanceIndexIdentity_t
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          primitiveInternalIndex;
};

typedef struct HitInfoDistanceIndexIdentity_t HitInfoDistanceIndexIdentity;


/*!
@struct Hit Info containing distance and BVH information.
*/
struct DEFAULT_ALIGN HitInfoDistanceBVHHits_t
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          bvhHits;
};

typedef struct HitInfoDistanceBVHHits_t HitInfoDistanceBVHHits;


#ifdef COMPUTE_SHADER_SCOPE

#ifdef IntersectionTypeClosest
#define setHitDistance(hitDistance, time) hitDistance = time
#else
#define setHitDistance(hitDistance, time)
#endif

#ifndef HitStructIdentity
#define setHitPrimitiveIndex(hitPrimitiveIndex, index) hitPrimitiveIndex = index
#else
#define setHitPrimitiveIndex(hitPrimitiveIndex, index)
#endif

#ifndef HitStructIndex
#define setHitPrimitiveIdentity(hitPrimitiveIdentity, identity) hitPrimitiveIdentity = identity
#else
#define setHitPrimitiveIdentity(hitPrimitiveIdentity, identity)
#endif

#ifdef HitStructIndexIdentity
#define setHitNormal(hitNormal, normal) hitNormal = normal
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index) primitiveInternalIndex = index
#else
#define setHitNormal(hitNormal, normal)
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index)
#endif

#if defined(HitStructBVHHits)
#define addBVHHit(hitBVHHits, hitCount) hitBVHHits += hitCount
#define setBVHHit(hitBVHHits, hitCount) hitBVHHits = hitCount
#else
#define addBVHHit(hitBVHHits, hitCount)
#define setBVHHit(hitBVHHits, hitCount)
#endif


inline void initializeHit(Thread HitStruct* hit)
{
  hit->distance = INFINITY;
  setHitPrimitiveIndex(hit->primitiveIndex, -1);
  setHitPrimitiveIdentity(hit->primitiveIdentity.identity, -1);
#if defined(HitStructBVHHits)
  setBVHHit(hit->bvhHits, 0);
#endif
}

#else

extern uint getHitStructSize(HitStructType type);

extern string getHitStructName(HitStructType type);

extern void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type);

#endif

#pragma pack(pop)

#endif
