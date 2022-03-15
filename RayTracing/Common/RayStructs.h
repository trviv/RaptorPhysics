#ifndef RAY_STRUCTS_H
#define RAY_STRUCTS_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#else
#include "ComputeHeader.shader"
#include "ComputeShared.h"
#endif

#pragma pack(push, 4)

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
#ifdef COMPUTE_SHADER_SCOPE
  packed_float3 origin;
  float         maxDistance;
  packed_float3 direction;
  uint          rayIndex;
#else
  union
  {
    struct
    {
      float3  origin;
      float3  direction;
    };
    struct
    {
      uint    reserved[3];
      float   maxDistance;
      uint    reserved1[3];
      uint    rayIndex;
    };
  };
#endif
};

typedef struct Ray_t RayBase;

/*!
@struct Ray containing origin and direction.
*/
struct DEFAULT_ALIGN RayColor_t
{
#ifdef COMPUTE_SHADER_SCOPE
  packed_float3 origin;
  float         maxDistance;
  packed_float3 direction;
  uint          rayIndex;
#else
  union
  {
    struct
    {
      float3  origin;
      float3  direction;
    };
    struct
    {
      uint    reserved[3];
      float   maxDistance;
      uint    reserved1[3];
      uint    rayIndex;
    };
  };
#endif
  colorType4 color;
};

typedef struct RayColor_t RayColor;


#ifndef COMPUTE_SHADER_SCOPE

extern uint getRayStructSize(RayStructType type);

extern string getRayStructName(RayStructType type);

extern void getRayStructDefines(vector<string>& oldType, vector<string>& newType, RayStructType type);

#endif

#pragma pack(pop)

#endif
