/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "HitStructs.h"

uint getHitStructSize(HitStructType type)
{
  switch (type)
  {
    case HitStructDistanceIdentity:
      return sizeof(HitInfoIdentity);
      break;
    case HitStructDistanceIndexIdentity:
      return sizeof(HitInfoDistanceIndexIdentity);
      break;
    case HitStructDistanceBVHHits:
      return sizeof(HitInfoDistanceBVHHits);
      break;
    case HitStructDistanceIndexIdentityNormal:
      return sizeof(HitInfoDistanceIndexIdentityNormal);
      break;
    case HitStructDistanceIndexIdentityNormalUV:
      return sizeof(HitInfoDistanceIndexIdentityNormalUV);
      break;
    default:
      return 0;
      break;
  }

  return 0;
}

string getHitStructName(HitStructType type)
{
  switch (type)
  {
    case HitStructDistanceIdentity:
      return "HitInfoIdentity";
      break;
    case HitStructDistanceIndexIdentity:
      return "HitInfoDistanceIndexIdentity";
      break;
    case HitStructDistanceBVHHits:
      return "HitInfoDistanceBVHHits";
      break;
    case HitStructDistanceIndexIdentityNormal:
      return "HitInfoDistanceIndexIdentityNormal";
      break;
    case HitStructDistanceIndexIdentityNormalUV:
      return "HitInfoDistanceIndexIdentityNormalUV";
      break;
    default:
      return "";
      break;
  }

  return "";
}

void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type)
{
  switch (type)
  {
    case HitStructDistanceIdentity:
      oldType.push_back("HitStructIdentity");
      newType.push_back("");
      break;
    case HitStructDistanceIndexIdentity:
      oldType.push_back("HitStructIndex");
      oldType.push_back("HitStructIdentity");
      newType.push_back("");
      newType.push_back("");
      break;
    case HitStructDistanceBVHHits:
      oldType.push_back("HitStructIdentity");
      oldType.push_back("HitStructBVHHits");
      newType.push_back("");
      newType.push_back("");
      break;
    case HitStructDistanceIndexIdentityNormal:
      oldType.push_back("HitStructIndex");
      oldType.push_back("HitStructIdentity");
      oldType.push_back("HitStructNormal");
      newType.push_back("");
      newType.push_back("");
      newType.push_back("");
      break;
    case HitStructDistanceIndexIdentityNormalUV:
      oldType.push_back("HitStructIndex");
      oldType.push_back("HitStructIdentity");
      oldType.push_back("HitStructNormal");
      oldType.push_back("HitStructUV");
      newType.push_back("");
      newType.push_back("");
      newType.push_back("");
      newType.push_back("");
      break;
    default:
      break;
  }
}
