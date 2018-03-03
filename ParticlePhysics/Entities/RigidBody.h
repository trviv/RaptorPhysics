#ifndef RIGID_BODY_H
#define RIGID_BODY_H

#include "PhysicsEntity.h"
#include "../Solvers/RigidSolver.h"

/*!
@class Class representing a rigid body.
*/
class RigidBody : public PhysicsEntity
{
public:

  /*@constructor Create a new rigid entity.*/
  RigidBody();

  /*@function Initialize a cube.*/
  void init(const Matrix4& transform, const real dim[],
    const uint subdivision[], const real mass);

  /*@function Initialize a cube.*/
  void init(const Matrix4& transform, const real dim[],
    const real particleRadius, const real mass);

#ifdef ENABLE_RENDERING
  void render(ParticleStruct* particles);
#endif

};

#endif