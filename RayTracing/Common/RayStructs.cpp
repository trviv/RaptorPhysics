/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "RayStructs.h"

uint getRayStructSize(RayStructType type)
{
  switch (type)
  {
    case RayStructPositionDirection:
      return sizeof(RayBase);
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
      return "RayBase";
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
    case RayStructPositionDirection:
      oldType.push_back("RayStructBase");
      newType.push_back("");
      break;
    case RayStructPositionDirectionColor:
      oldType.push_back("RayStructColor");
      newType.push_back("");
      break;
    default:
      break;
  }
}
