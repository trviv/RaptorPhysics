#ifndef RX_REAL3
#define RX_REAL3

#include "../Header/Math.h"

enum AXIS
{
  X, Y, Z
};

/// The vector class
ALIGN(4) CDEF class Real3

#if RX_CUDA
#if RX_PREC_DOUBLE
  : private double3
#elif RX_PREC_FLOAT
  : private float3
#endif
#endif
{
protected:
#ifndef RX_CUDA
  real x, y, z, w;
#endif

public:
  /// Default constructor
  FORCE_INLINE CU_DEV_HOST Real3()
  {}

  /// Copy one value to all
  FORCE_INLINE CU_DEV_HOST Real3(const real& val)
  {
    *this = val;
  }

  /// Copy constructor
  FORCE_INLINE CU_DEV_HOST Real3(const Real3& val)
  {
    *this = val;
  }

  /// Construct from vector
  //FORCE_INLINE Real3(const Vec3& val)
  //{ *this = val;}

  /// Copy different values
  FORCE_INLINE CU_DEV_HOST Real3(const real& x, const real& y, const real& z)
  {
    set(x, y, z);
  }

  /// Assignment operator
  FORCE_INLINE CU_DEV_HOST Real3& operator=(const real& val)
  {
    set(val, val, val); return *this;
  }

  /// Assignment operator
  FORCE_INLINE CU_DEV_HOST Real3& operator=(const Real3& val)
  {
    set(val.x, val.y, val.z); return *this;
  }

  /// Assignment operator
  /*
  FORCE_INLINE  CU_DEV_HOST Real3&  operator=(const Vec3& val)
  {
  #if USING_SSE2
  #if PREC_DOUBLE
  xy[0] = val[X]; xy[1] = val[Y]; z = val[Z]; return *this;
  #else
  x = val[X]; y = val[Y]; z = val[Z]; return *this;
  #endif
  #else
  xy[0] = val[X]; xy[1] = val[Y]; z = val[Z]; return *this;
  x = val[X]; y = val[Y]; z = val[Z]; return *this;
  #endif
  }
  */

  /// Addition operator (for vector)
  FORCE_INLINE CU_DEV_HOST Real3 operator+(const Real3& val)const
  {
    Real3 ret;
    ret.x = x + val.x;
    ret.y = y + val.y;
    ret.z = z + val.z;
    return ret;
  }

  /// Subtraction operator (for vector)
  FORCE_INLINE CU_DEV_HOST Real3 operator-(const Real3& val)const
  {
    Real3 ret;
    ret.x = x - val.x;
    ret.y = y - val.y;
    ret.z = z - val.z;
    return ret;
  }

  /// Multiplication operator (for vector)
  FORCE_INLINE CU_DEV_HOST Real3 operator*(const Real3& val)const
  {
    Real3 ret;
    ret.x = x*val.x;
    ret.y = y*val.y;
    ret.z = z*val.z;
    return ret;
  }

  /// Multiplication operator (for value)
  FORCE_INLINE CU_DEV_HOST Real3 operator*(const real& val)const
  {
    Real3 ret;
    ret.x = x*val;
    ret.y = y*val;
    ret.z = z*val;
    return ret;
  }

  /// Division operator (for vector)
  FORCE_INLINE CU_DEV_HOST Real3 operator/(const Real3& val)const
  {
    Real3 ret;
    ret.x = x / val.x;
    ret.y = y / val.y;
    ret.z = z / val.z;
    return ret;
  }

  /// Division operator (for value)
  FORCE_INLINE CU_DEV_HOST Real3 operator/(const real& val)const
  {
    prompt(val);
    return (*this)*(real(1) / val);
  }

  /// Increment operator (for vector)
  FORCE_INLINE CU_DEV_HOST void operator+=(const Real3& val)
  {
    x += val.x; y += val.y; z += val.z;
  }

  /// Increment operator (for value)
  FORCE_INLINE CU_DEV_HOST void operator+=(const real& val)
  {
    x += val; y += val; z += val;
  }

  /// Decrement operator (for vector)
  FORCE_INLINE CU_DEV_HOST void operator-=(const Real3& val)
  {
    x -= val.x; y -= val.y; z -= val.z;
  }

  /// Scale multiply operator (for vector)
  FORCE_INLINE CU_DEV_HOST void operator*=(const Real3& val)
  {
    x *= val.x; y *= val.y; z *= val.z;
  }

  /// Scale multiply operator (for value)
  FORCE_INLINE CU_DEV_HOST void operator*=(const real& val)
  {
    x *= val; y *= val; z *= val;
  }

  /// Scale divide operator (for value)
  FORCE_INLINE CU_DEV_HOST void operator /= (const real& val)
  {
    prompt(val); return (*this) *= (real(1) / val);
  }

  /// Access operator
  FORCE_INLINE CU_DEV_HOST const real& operator[](const Counter& index)const
  {
    return (&this->x)[index];
  }

  /// Access operator
  FORCE_INLINE CU_DEV_HOST real& operator [](const Counter& index)
  {
    return (&this->x)[index];
  }

  /// Get cross product
  FORCE_INLINE CU_DEV_HOST Real3 cross(const Real3& val)const
  {
    return  Real3(y*val.z - z*val.y,
      z*val.x - x*val.z,
      x*val.y - y*val.x);
  }

  /// Get square of length of vector
  FORCE_INLINE  CU_DEV_HOST real  lengthSq()const
  {
    return  x * x + y * y + z * z;
  }

  /// Get length of the vector
  FORCE_INLINE  CU_DEV_HOST real  length()const
  {
    return  mSqrt(lengthSq());
  }

  /// Get square of distance from the point
  FORCE_INLINE  CU_DEV_HOST real  distanceSq(const Real3& point)const
  {
    return  (point - *this).lengthSq();
  }

  /// Get distance from point
  FORCE_INLINE  CU_DEV_HOST real  distance(const Real3& point)const
  {
    return  mSqrt(distanceSq(point));
  }

  /// Dot product of vectors
  FORCE_INLINE  CU_DEV_HOST real  dot(const Real3& val)const
  {
    return x * val.x + y * val.y + z * val.z;
  }

  /// Get additive inverse
  FORCE_INLINE  CU_DEV_HOST Real3 additiveInv()const
  {
    return  Real3(-x, -y, -z);
  }

  /// Get multiplicative inverse
  FORCE_INLINE  CU_DEV_HOST Real3 multiplicativeInv()const
  {
    return  Real3(1 / x, 1 / y, 1 / z);
  }

  /// Get the axis with maximum value
  FORCE_INLINE  CU_DEV_HOST AXIS  longestAxis()const
  {
    if (x >= y && x >= z)      return X;
    else if (y >= x && y >= z) return Y;
    return Z;
  }

  ///// Add the product of b and c ( += b*c )
  //FORCE_INLINE void addMul(const Real3& b, const real c)
  //{ x += b.x * c; y += b.y * c; z += b.z * c;}

  ///// Set the addition of product ( = a + b*c )
  //FORCE_INLINE void setAddMul(const Real3& a, const Real3& b,
  //  const real c)
  //{
  //  x = a.x + b.x * c;
  //  y = a.y + b.y * c;
  //  z = a.z + b.z * c;
  //}

  /// Set all values to Zero
  FORCE_INLINE  CU_DEV_HOST void  setNull()
  {
    *this = 0;
  }

  ///// Set the product ( = a * b )
  //FORCE_INLINE void setMul(const Real3& a, const real b)
  //{ set(a.x * b, a.y * b, a.z * b);}

  ///// Set the product ( = a * b )
  //FORCE_INLINE void setMul(const Real3& a, const Real3& b)
  //{ x = a.x * b.x;  y = a.y * b.y;  z = a.z * b.z;}

  /// Set normalized vector
  FORCE_INLINE  CU_DEV_HOST Real3&  normalize()
  {
    operator/=(length()); return *this;
  }

  /// Set normalized vector ( return time , set inverse time)
  FORCE_INLINE  CU_DEV_HOST real  normalize(real& inv_time)
  {
    real t = length();
    inv_time = real(1) / t;
    operator*=(inv_time);
    return t;
  }

  /*
  /// Randomize this vector
  /// and check whether it is along normal or not
  /// if not then inverse its direction
  FORCE_INLINE  CU_DEV_HOST void  randomize(const Real3& normal)
  {
  srand(timeGetTime());
  // get x,y,z random numbers between -1 to 1 for float
  real l;
  do
  {
  x = (-1 + 2*(real)rx_system.random()*INV_RAND_MAX);
  y = (-1 + 2*(real)rx_system.random()*INV_RAND_MAX);
  z = (-1 + 2*(real)rx_system.random()*INV_RAND_MAX);
  l = x*x + y*y + z*z;
  }
  while ( l > 1 || l < MIN);
  this->operator /=(l);
  if( this->dot(normal)<real(0) ) //if enterng inside object
  *this = additiveInv();
  }
  */

  FORCE_INLINE  CU_DEV_HOST Real3 getReflected(
    const Real3&  normal, //normal
    const real&   dot_n_id)const  //dot of normal and incident dir
  {
    prompt(dot_n_id > real(0)); //if dot is -ve report
    return *this + (normal*(real(2)*dot_n_id));
  }

  /// Set from different values
  FORCE_INLINE  CU_DEV_HOST void  set(const real& x, const real& y,
    const real& z)
  {
    this->x = x;  this->y = y;  this->z = z;
  }

  FORCE_INLINE  CU_DEV_HOST Real3 sqrt()const
  {
    Real3  ret;
    ret.x = mSqrt(x);
    ret.y = mSqrt(y);
    ret.z = mSqrt(z);
    return ret;
  }

  ///// set addition of two ( = a + b )
  //FORCE_INLINE void setAdd(const Real3& a, const Real3& b)
  //{ x = a.x + b.x;  y = a.y + b.y;  z = a.z + b.z;}

  ///// Set subtraction ( = a - b )
  //FORCE_INLINE void setSub(const Real3& a, const Real3& b)
  //{ x = a.x - b.x;  y = a.y - b.y;  z = a.z - b.z;}

  ///// Set subtraction and normalize
  //FORCE_INLINE void setNormalizedSub(const Real3& a, const Real3& b)
  //{
  //  setSub(a, b);
  //  this->normalize();
  //}

  ///// Set subtraction and multiply ( = (this - a) * b )
  //FORCE_INLINE void subMul(const Real3& a, const real b)
  //{ x = (x - a.x) * b;  y = (y - a.y) * b;  z = (z - a.z) * b;}

  ///// Set multiply and subtract ( = a * b - c )
  //FORCE_INLINE void setMulSub(const Real3& a, const real b, const Real3& c)
  //{
  //  x = (a.x * b) - c.x;
  //  y = (a.y * b) - c.y;
  //  z = (a.z * b) - c.z;
  //}
};
/*
FORCE_INLINE CU_DEV_HOST std::string operator << (const Real3& prefix, const std::string& suffix)
{
char str[1024];
str[0] = '\0';
sprintf(str, "\nx: %f, y: %f, z: %f\n", prefix[0], prefix[1], prefix[2]);
return std::string(str) + suffix;
}
*/
FORCE_INLINE CU_HOST std::ostream& operator<<(std::ostream& prefix, const Real3& suffix)
{
  char str[1024];
  str[0] = '\0';
  sprintf(str, "\nx: %f, y: %f, z: %f\n", suffix[0], suffix[1], suffix[2]);
  prefix << std::string(str).c_str();
  return prefix;
}

/// Multiplication operator (for value)
FORCE_INLINE CU_DEV_HOST Real3 operator*(const real& val1, const Real3& val2)
{
  return Real3(val2)*val1;
}

#endif