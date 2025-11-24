/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef ANIMATOR_H
#define ANIMATOR_H

#include <Vector/Real3.h>

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
    current = pointA + (pointB - pointA) * mCrop(timeStep, 0.f, 1.f);
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
    current = ref;
    return current;
  }

  ClassType operator + (const ClassType& ref)const
  {
    return current + ref;
  }

  ClassType operator * (const ClassType& ref)const
  {
    return current * ref;
  }

  void operator += (const ClassType& ref)
  {
    current += ref;
  }

  ClassType& operator()()
  {
    return current;
  }
};

#endif
