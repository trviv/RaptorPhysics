#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

#include "LinearSolver.h"
#include "../Common/ParticleStruct.h"

/*!
@class Class to fluid constraints.
*/
class FluidSolver : public Solver<uint, real, Real3>
{
  void update();

public:

  FluidSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve();
};

#endif
