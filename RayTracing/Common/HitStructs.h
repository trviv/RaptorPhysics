#ifndef HIT_STRUCTS_H
#define HIT_STRUCTS_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#pragma pack(push, 4)

enum HitStructType
{
  HitStructDistanceIndex,
  HitStructDistanceIndexNormal,
  HitStructTypeMax
};


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
  hit->primitiveIdentity.identity = -1;
}

#else

extern uint getHitStructSize(HitStructType type);

extern string getHitStructName(HitStructType type);

extern void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type);

#endif

#pragma pack(pop)

#endif
