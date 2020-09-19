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
  RayStructTypeMax
};


/*!
@struct Ray containing origin and direction.
*/
struct DEFAULT_ALIGN Ray_t
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
    };
  };
};

typedef struct Ray_t Ray;


#ifndef COMPUTE_SHADER_SCOPE

static uint getRayStructSize(RayStructType type)
{
  switch (type)
  {
    case RayStructPositionDirection:
      return sizeof(Ray_t);
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
    default:
      return "";
      break;
  }

  return "";
}

#endif

#pragma pack(pop)

#endif
