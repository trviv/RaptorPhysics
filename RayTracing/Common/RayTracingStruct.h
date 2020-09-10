#ifndef RAY_TRACING_STRUCT_H
#define RAY_TRACING_STRUCT_H

#ifndef COMPUTE_SHADER_SCOPE
#include <Core.h>
#endif

#pragma pack(push, 4)

#ifdef COMPUTE_SHADER_SCOPE

// Returns the i'th element of the Halton sequence using the d'th prime number as a
// base. The Halton sequence is a "low discrepency" sequence: the values appear
// random but are more evenly distributed then a purely random sequence. Each random
// value used to render the image should use a different independent dimension 'd',
// and each sample (frame) should use a different index 'i'. To decorrelate each
// pixel, a random offset can be applied to 'i'.
float halton(Const* primes, uint i, uint d)
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

enum RayType
{
  RayTypePrimary = 0x1
};

struct Ray_t
{
  union
  {
    struct
    {
      float3  origin;
    };
    struct
    {
      uint    reserved[3];
      uint    type;
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
      uint    reserved1[3];
    };
  };
};

typedef struct Ray_t Ray;


/*!
@struct Shared Camera information.
*/
struct CameraStruct_t
{
  union
  {
    struct
    {
      float3  deltaX;  // shift in x axis per pixel
    };
    struct
    {
      uint    reserved1[3];
      uint    width;
    };
  };
  union
  {
    struct
    {
      float3  deltaY;  // shift in y axis per pixel
    };
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
struct LightStruct_t
{
  Color3 color;
};

typedef struct LightStruct_t LightStruct;

#pragma pack(pop)

#endif
