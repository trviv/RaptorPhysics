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

  /*!@constructor Create a new rigid entity.*/
  RigidBody();

  /*!
  @function Initialize a cube.
  @param dimensions Cube dimensions.
  @param particleRadius Radius for each particle.
  @param mass Entity mass.
  */
  void initCube(const real dimensions[], const real particleRadius, const real mass);
};

#endif
