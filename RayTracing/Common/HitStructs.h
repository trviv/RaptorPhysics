#ifndef HIT_STRUCTS_H
#define HIT_STRUCTS_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#pragma pack(push, 4)

enum HitStructType
{
  HitStructDistanceIndex,
  HitStructTypeMax
};


/*!
@struct Hit Info containing distance information.
*/
struct ALIGN(8) HitInfoDistance_t
{
  float distance;
  uint  primitiveIndex;
};

typedef struct HitInfoDistance_t HitInfoDistance;

#ifndef COMPUTE_SHADER_SCOPE

static uint getHitStructSize(HitStructType type)
{
  switch (type)
  {
    case HitStructDistanceIndex:
      return sizeof(HitInfoDistance_t);
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
    default:
      return "";
      break;
  }

  return "";
}

#endif

#pragma pack(pop)

#endif
