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
  HitStructDistanceIndexNormal,
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
struct DEFAULT_ALIGN HitInfoDistanceIndexNormal_t
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          primitiveInternalIndex;
};

typedef struct HitInfoDistanceIndexNormal_t HitInfoDistanceIndexNormal;


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

#ifdef HitStructNormal
#define setHitNormal(hitNormal, normal) hitNormal = normal
#else
#define setHitNormal(hitNormal, normal)
#endif

#ifdef HitStructNormal
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index) primitiveInternalIndex = index
#else
#define setHitPrimitiveInternalIndex(primitiveInternalIndex, index)
#endif


inline void initializeHit(Thread HitStruct* hit)
{
  hit->distance = INFINITY;
  setHitPrimitiveIndex(hit->primitiveIndex, -1);
  setHitPrimitiveIdentity(hit->primitiveIdentity.identity, -1);
}

#else

extern uint getHitStructSize(HitStructType type);

extern string getHitStructName(HitStructType type);

extern void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type);

#endif

#pragma pack(pop)

#endif
