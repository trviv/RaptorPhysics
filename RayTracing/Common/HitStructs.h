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
struct DEFAULT_ALIGN HitInfoDistance_t
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          padding;
};

typedef struct HitInfoDistance_t HitInfoDistance;


/*!
@struct Hit Info containing distance information.
*/
struct DEFAULT_ALIGN HitInfoDistanceIndexNormal_t
{
  float         distance;
  uint          primitiveIndex;
  IdentityInfo  primitiveIdentity;
  uint          padding;
  float3        normal;
};

typedef struct HitInfoDistanceIndexNormal_t HitInfoDistanceIndexNormal;


#ifdef COMPUTE_SHADER_SCOPE

inline void initializeHit(Thread HitStruct* hit)
{
  hit->distance       = INFINITY;
  hit->primitiveIndex = -1;
#ifndef HitStructIndex
  hit->primitiveIdentity.identity = -1;
#endif
}

#ifdef IntersectionTypeClosest
#define setHitDistance(hitDistance, time) hitDistance = time
#else
#define setHitDistance(hitDistance, time)
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

#else

extern uint getHitStructSize(HitStructType type);

extern string getHitStructName(HitStructType type);

extern void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type);

#endif

#pragma pack(pop)

#endif
