#ifndef CLOTH_H
#define CLOTH_H

#include "PhysicsEntity.h"
#include "../Solvers/DistanceSolver.h"

/*!
@class Class representing a cloth.
*/
class Cloth : public PhysicsEntity
{
public:

  Cloth();

  void initXY(const real dimensions[], const uint subdivision[], const real mass);

  void initXY(const real dimensions[], const real particleRadius, const real mass);

#ifdef ENABLE_RENDERING
  void render(ParticleStruct* particles);
#endif

};

#endif