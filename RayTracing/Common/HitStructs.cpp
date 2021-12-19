#include "HitStructs.h"

uint getHitStructSize(HitStructType type)
{
  switch (type)
  {
    case HitStructDistanceIndex:
      return sizeof(HitInfoIndex);
      break;
    case HitStructDistanceIdentity:
      return sizeof(HitInfoIdentity);
      break;
    case HitStructDistanceIndexIdentity:
      return sizeof(HitInfoDistanceIndexIdentity);
      break;
    case HitStructDistanceBVHHits:
      return sizeof(HitInfoDistanceBVHHits);
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
    case HitStructDistanceIndex:
      return "HitInfoIndex";
      break;
    case HitStructDistanceIdentity:
      return "HitInfoIdentity";
      break;
    case HitStructDistanceIndexIdentity:
      return "HitInfoDistanceIndexIdentity";
      break;
    case HitStructDistanceBVHHits:
      return "HitInfoDistanceBVHHits";
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
  case HitStructDistanceIndex:
    oldType.push_back("HitStructIndex");
    newType.push_back("");
    break;
  case HitStructDistanceIdentity:
    oldType.push_back("HitStructIdentity");
    newType.push_back("");
    break;
  case HitStructDistanceIndexIdentity:
    oldType.push_back("HitStructIndexIdentity");
    newType.push_back("");
    break;
  case HitStructDistanceBVHHits:
    oldType.push_back("HitStructBVHHits");
    newType.push_back("");
    break;
  default:
    break;
  }
}
