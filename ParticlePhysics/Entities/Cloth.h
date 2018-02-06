#ifndef CLOTH_H
#define CLOTH_H

#include "PhysicsEntity.h"
#include "../Solvers/DistanceSolver.h"

/*!
@class Class representing a cloth
*/
class Cloth : public PhysicsEntity
{
public:

  Cloth();

  void init(const Matrix4& transform, const real dim[],
    const uint subdivision[], const real mass);

  void init(const Matrix4& transform, const real dim[],
    const real particleRadius, const real mass);

#ifdef ENABLE_RENDERING
  void render(ParticleStruct* particles);
#endif

};

#endif