#ifndef FLUID_SOLVER_PBF_H
#define FLUID_SOLVER_PBF_H

#include "FluidSolver.h"

/*!
@class Class to solve fluid constraints using Position Based Fluids paper.
*/
class FluidSolverPBF : public FluidSolver
{
protected:

  void update();

public:

  FluidSolverPBF(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve(float timeStep);
};

#endif
