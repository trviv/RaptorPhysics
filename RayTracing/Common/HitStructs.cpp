#include "HitStructs.h"

uint getHitStructSize(HitStructType type)
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

string getHitStructName(HitStructType type)
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

void getHitStructDefines(vector<string>& oldType, vector<string>& newType, HitStructType type)
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
