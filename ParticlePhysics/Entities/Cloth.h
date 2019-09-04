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

  /*!@constructor Create a new cloth entity.*/
  Cloth();

  /*!
  @function Initialize a 2D cloth based on the parameters.
  @param dimension Cloth xy dimension.
  @param subdivision Cloth xy dimension grid size.
  @param mass Cloth mass.
  */
  void initXY(const real dimensions[], const uint subdivision[], const real mass);

  /*!
  @function Initialize a 2D cloth based on the parameters.
  @param dimension Cloth xy dimension.
  @param particleRadius Radius of cloth particles.
  @param mass Cloth mass.
  */
  void initXY(const real dimensions[], const real particleRadius, const real mass);

#ifdef ENABLE_RENDERING
  void render(ParticleStruct* particles);
#endif

};

#endif
