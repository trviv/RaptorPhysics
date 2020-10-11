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
  float distance;
  uint  primitiveIndex;
};

typedef struct HitInfoDistance_t HitInfoDistance;


/*!
@struct Hit Info containing distance information.
*/
struct DEFAULT_ALIGN HitInfoDistanceIndexNormal_t
{
  float distance;
  uint  primitiveIndex;
  uint  padding[2];
  float3 normal;
};

typedef struct HitInfoDistanceIndexNormal_t HitInfoDistanceIndexNormal;


#ifdef COMPUTE_SHADER_SCOPE

inline void initializeHit(Thread HitStruct* hit)
{
  hit->distance       = INFINITY;
  hit->primitiveIndex = -1;
}

#else

static uint getHitStructSize(HitStructType type)
{
  switch (type)
  {
    case HitStructDistanceIndex:
      return sizeof(HitInfoDistance);
      break;
    case HitStructDistanceIndexNormal:
      return sizeof(HitInfoDistanceIndexNormal);
      break;
    default:
      return 0;
      break;
  }

  return 0;
}

static string getHitStructName(HitStructType type)
{
  switch (type)
  {
    case HitStructDistanceIndex:
      return "HitInfoDistance";
      break;
    case HitStructDistanceIndexNormal:
      return "HitInfoDistanceIndexNormal";
      break;
    default:
      return "";
      break;
  }

  return "";
}

static void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type)
{
  switch (type)
  {
  case HitStructDistanceIndexNormal:
    oldType.push_back("HitStructNormal");
    newType.push_back("");
    break;
  default:
    break;
  }
}

#endif

#pragma pack(pop)

#endif
