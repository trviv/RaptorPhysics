#ifndef FLUID_SOLVER_PBF_H
#define FLUID_SOLVER_PBF_H

#include "FluidSolver.h"

/*!
@class Class to solve fluid constraints using Position Based Fluids paper.
*/
class FluidSolverPBF : public FluidSolver
{
protected:

  DeviceArray <ParticleStruct> &particlesOmega  = particlesTemp[1];

  void update();

public:

  FluidSolverPBF(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve(float timeStep);

  void postCollisionSolve(float timeStep);
};

#endif
