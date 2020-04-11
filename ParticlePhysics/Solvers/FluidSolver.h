#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

#include "LinearSolver.h"
#include "FluidSolverCommon.h"
#include "../Common/ParticleStruct.h"
#include "Collision/UniformGridCollisionSolver.h"

/*!
@class Class to solve fluid constraints.
*/
class FluidSolver : public EntitySolver<uint, real, Real3>, protected UniformGridCollisionSolver
{
protected:

  DeviceArray <float> particlesDensity;
  DeviceArray <float> particlesLambda;
  DeviceArray <float>&particlesColor = particlesLambda;

  void update();

  void updateRadius();

  FluidSolver(ComputeInterface* compute, SharedAllocator* allocator, bool noCreate);

public:

  FluidSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve(float timeStep);
};

#endif
