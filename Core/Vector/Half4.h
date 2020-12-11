#ifndef HALF4
#define HALF4

#include <half.hpp>
#include "Real3.h"


typedef half_float::half half;

class Half4;

// vector structor
struct ALIGN(8) half4
{
  half x, y, z, a;

  operator Half4&()
  {
    return *((Half4*)&x);
  }

  operator Half4&()const
  {
    return *((Half4*)&x);
  }
};

class Half4
#ifdef USE_METAL_COMPUTE
  : public half4
#else
  : public float4
#endif
{
public:

  Half4()
  {

  }

  Half4(const Half4& ref)
  {
    x = ref.x;
    y = ref.y;
    z = ref.z;
    a = ref.a;
  }

  Half4(real x, real y, real z, real a)
  {
    this->x = x;
    this->y = y;
    this->z = z;
    this->a = a;
  }

  operator Real3()const
  {
    Real3 ret((float)x, (float)y, (float)z);
    ret.a = a;
    return ret;
  }

  void operator = (const Real3& ref)
  {
    x = ref.x;
    y = ref.y;
    z = ref.z;
    a = ref.a;
  }

#ifdef USE_METAL_COMPUTE
  const half& operator[](const int& index)const
  {
    return (&this->x)[index];
  }

  half& operator[](const int& index)
  {
    return (&this->x)[index];
  }
#else
  const float& operator[](const int& index)const
  {
    return (&this->x)[index];
  }

  float& operator[](const int& index)
  {
    return (&this->x)[index];
  }
#endif
};

#endif
