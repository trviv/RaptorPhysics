#include "RayStructs.h"

uint getRayStructSize(RayStructType type)
{
  switch (type)
  {
    case RayStructPositionDirection:
      return sizeof(Ray);
      break;
    case RayStructPositionDirectionColor:
      return sizeof(RayColor);
      break;
    default:
      return 0;
      break;
  }

  return 0;
}

string getRayStructName(RayStructType type)
{
  switch (type)
  {
    case RayStructPositionDirection:
      return "Ray";
      break;
    case RayStructPositionDirectionColor:
      return "RayColor";
      break;
    default:
      return "";
      break;
  }

  return "";
}

void getRayStructDefines(vector<string>& oldType, vector<string>& newType, RayStructType type)
{
  switch (type)
  {
  case RayStructPositionDirectionColor:
    oldType.push_back("RayStructColor");
    newType.push_back("");
    break;
  default:
    break;
  }
}
