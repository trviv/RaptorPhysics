#ifndef HALF4
#define HALF4

#include <half.hpp>
#include "Real3.h"


typedef half_float::half half;

class Half4;

// vector structor
struct half4
{
  ushort x, y, z, a;

  operator Half4&()
  {
    return *((Half4*)&x);
  }

  operator Half4&()const
  {
    return *((Half4*)&x);
  }
};

class Half4 : public half4
{
  half& hx = (half&)x;
  half& hy = (half&)y;
  half& hz = (half&)z;
  half& ha = (half&)a;

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
    Real3 ret((float)hx, (float)hy, (float)hz);
    ret.a = ha;
    return ret;
  }

  void operator = (const Real3& ref)
  {
    hx = ref.x;
    hy = ref.y;
    hz = ref.z;
    ha = ref.a;
  }

  const half& operator[](const int& index)const
  {
    return (&this->hx)[index];
  }

  half& operator[](const int& index)
  {
    return (&this->hx)[index];
  }
};

#endif
