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
    case HitStructDistanceIndexNormal:
      return sizeof(HitInfoDistanceIndexNormal);
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
    case HitStructDistanceIndexNormal:
      return "HitInfoDistanceIndexNormal";
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
  case HitStructDistanceIndexNormal:
    oldType.push_back("HitStructNormal");
    newType.push_back("");
    break;
  default:
    break;
  }
}
