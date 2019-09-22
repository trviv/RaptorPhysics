#ifndef FLUID_H
#define FLUID_H

#include "PhysicsEntity.h"
#include "../Solvers/FluidSolver.h"

/*!
@class Class representing a Fluid.
*/
class Fluid : public PhysicsEntity
{
public:

  /*!@constructor Create a new fluid entity.*/
  Fluid();

  /*!
  @function Initialize a cube.
  @param dimensions Cube dimensions.
  @param particleRadius Radius for each particle.
  @param mass Entity mass.
  */
  void initFluid(const real dimensions[], const real particleRadius, const real mass, float kernelRadius);

#ifdef ENABLE_RENDERING
  void render(ParticleStruct* particles);
#endif

};

#endif
