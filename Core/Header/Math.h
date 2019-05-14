#ifndef MATH_H
#define MATH_H

#include "Root.h"

#if PREC_FLOAT

#define mACos(x)      acosf(x)
#define mATan2(x,y)   atan2f(x,y)
#define mCos(x)       cosf(x)
#define mSin(x)       sinf(x)
#define mSqrt(x)      sqrtf(x)
#define mSqr(x)       x*x

#else

#define mACos(x)      acos(x)
#define mATan2(x,y)   atan2(x,y)
#define mCos(x)       cos(x)
#define mSin(x)       sin(x)
#define mSqrt(x)      sqrt(x)
#define mSqr(x)       x*x

#endif

template<class T>T mAbs(const T& x)
{
  return abs(x);
}

/// Function to a clamp value
template<class T>T mCrop(const T v, const T lb, const T ub)
{
  return v < lb ? lb : (ub < v ? ub : v);
}

template<class T>T mRand(const T& min, const T& max)
{
  srand((int)min);
  return min + (max - min) / rand();
}

/// Function to get the minimum of two
template<class T>T mMin(T a, T b)
{
  return a < b ? a : b;
}

/// Fucntion to get the maximum of two
template<class T>T mMax(T a, T b)
{
  return a > b ? a : b;
}

template<class T>T mMin(T a, T b, T c)
{
  return mMin(mMin(a, b), c);
}

template<class T>T mPow(T a, T b)
{
  return pow(a, b);
}

template<class T>T mFloor(T a)
{
  return floor(a);
}

template<class T>T mCeil(T a)
{
  return ceil(a);
}

/// Function to find the number as an expoenent of 2
template<class T>T mCeilExpOf2(T integer)
{
  T exp;
  T backup = integer;
  for (exp = -1; integer; integer >>= 1)
  {
    exp++;
  }
  if ((((1 << exp) - 1) & backup)) exp++;
  return exp;
}

#endif
