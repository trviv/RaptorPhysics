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

extern uint getRayStructSize(RayStructType type);

extern string getRayStructName(RayStructType type);

extern void getRayStructDefines(vector<string>& oldType, vector<string>& newType, RayStructType type);

#endif

#pragma pack(pop)

#endif
