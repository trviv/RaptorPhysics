#ifndef ANIMATOR_H
#define ANIMATOR_H

#include "../Vector/Real3.h"

/*!
@class Class to hold an object which can be moved from one point to another.
*/
template<class ClassType> class AnimationElement
{
  ClassType pointA, pointB, current;

public:

  AnimationElement()
  {}

  ClassType& begin()
  {
    return this->pointA;
  }

  ClassType& end()
  {
    return this->pointB;
  }

  ClassType& interpolate(float timeStep)
  {
    current = pointA + (pointB - pointA) * timeStep;
    return current;
  }

  operator ClassType&()
  {
    return current;
  }

  operator const ClassType&()const
  {
    return current;
  }

  ClassType& operator = (const ClassType& ref)
  {
    ((ClassType&)*this) = ref;
    return *this;
  }

  ClassType operator + (const ClassType& ref)const
  {
    return ((ClassType&)*this) + ref;
  }

  ClassType operator * (const ClassType& ref)const
  {
    return ((ClassType&)*this) * ref;
  }

  void operator += (const ClassType& ref)const
  {
    ((ClassType&)*this) * ref;
  }

  ClassType& operator()()
  {
    return *this;
  }
};

#endif
