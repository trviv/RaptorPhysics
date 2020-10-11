#ifndef RAY_STRUCTS_H
#define RAY_STRUCTS_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#pragma pack(push, 4)

enum RayType
{
  RayTypePrimary  = 1
};

enum RayStructType
{
  RayStructPositionDirection,
  RayStructPositionDirectionColor,
  RayStructTypeMax
};


/*!
@struct Ray containing origin and direction.
*/
struct DEFAULT_ALIGN Ray_t
{
  union
  {
    struct
    {
      float3  origin;
    };
    struct
    {
      uint  reserved[3];
      uint  type;
    };
  };

  union
  {
    struct
    {
      float3  direction;
    };
    struct
    {
      uint  reserved1[3];
      uint  rayIndex;
    };
  };
};

typedef struct Ray_t Ray;

/*!
@struct Ray containing origin and direction.
*/
struct DEFAULT_ALIGN RayColor_t
{
  union
  {
    float3  origin;
    struct
    {
      uint  reserved[3];
      uint  type;
    };
  };

  union
  {
    float3  direction;
    struct
    {
      uint  reserved1[3];
      uint  rayIndex;
    };
  };
  float3  color;
};

typedef struct RayColor_t RayColor;


#ifndef COMPUTE_SHADER_SCOPE

static uint getRayStructSize(RayStructType type)
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

static string getRayStructName(RayStructType type)
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

static void getRayStructDefines(vector<string>& oldType, vector<string>& newType, RayStructType type)
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

#endif

#pragma pack(pop)

#endif
