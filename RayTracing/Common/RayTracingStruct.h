#ifndef RAY_TRACING_STRUCT_H
#define RAY_TRACING_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include "RayStructs.h"
#include "HitStructs.h"
#endif

#pragma pack(push, 4)

#ifdef COMPUTE_SHADER_SCOPE

//// also known as slabs method
//inline bool isRayXABIntersecting(const XAB xab, const float3 rayOrigin, const float3 invRayDirection)
//{
//  const float3 t0 = (xab.min - rayOrigin) * invRayDirection;
//  const float3 t1 = (xab.max - rayOrigin) * invRayDirection;
//  const float3 tmin = min(t0, t1), tmax = max(t0, t1);
//  return max_component(tmin) <= min_component(tmax);
//}


// Returns the i'th element of the Halton sequence using the d'th prime number as a
// base. The Halton sequence is a "low discrepency" sequence: the values appear
// random but are more evenly distributed then a purely random sequence. Each random
// value used to render the image should use a different independent dimension 'd',
// and each sample (frame) should use a different index 'i'. To decorrelate each
// pixel, a random offset can be applied to 'i'.
float halton(Const uint* primes, uint i, uint d)
{
  uint b = primes[d];

  float f = 1.0f;
  float invB = 1.0f / b;
  float r = 0;

  while (i > 0)
  {
    f = f * invB;
    r = r + f * (i % b);
    i = i / b;
  }

  return r;
}

#endif


struct PackingInfo_t
{
  ushort strideIn4Bytes;
  ushort offsetIn4Bytes;

#ifndef COMPUTE_SHADER_SCOPE
  PackingInfo_t(ushort strideIn4Bytes = 0, ushort offsetIn4Bytes = 0)
  {
    this->strideIn4Bytes = strideIn4Bytes;
    this->offsetIn4Bytes = offsetIn4Bytes;
  }
#endif
};

typedef struct PackingInfo_t PackingInfo;


/*!
@struct Base data for a ray traced primitive.
@note   Should have same structure as PositionStruct. May cause issues otherwise.
*/
struct DEFAULT_ALIGN PrimitiveStruct_t
{
  union
  {
    float3  position;
    struct
    {
      uint  reserved[3];
      uint  identity;
    };
  };
};

typedef struct PrimitiveStruct_t PrimitiveStruct;


/*!
@struct Shared Camera information.
*/
struct DEFAULT_ALIGN CameraStruct_t
{
  union
  {
    float3  deltaX;  // shift in x axis per pixel
    struct
    {
      uint    reserved1[3];
      uint    width;
    };
  };
  union
  {
    float3  deltaY;  // shift in y axis per pixel
    struct
    {
      uint    reserved2[3];
      uint    height;
    };
  };
  float3 topLeft; // top left position
  float3 origin;  // camera starting position
};

typedef struct CameraStruct_t CameraStruct;


/*!
@struct Shared Light information.
*/
struct DEFAULT_ALIGN LightStruct_t
{
  Color3 color;
};

typedef struct LightStruct_t LightStruct;

#pragma pack(pop)

#endif
