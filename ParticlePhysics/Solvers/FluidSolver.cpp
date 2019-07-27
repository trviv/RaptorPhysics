#include "FluidSolver.h"

#define DEBUG_FLUID_SOLVER

FluidSolver::FluidSolver(ComputeInterface* compute, SharedAllocator* allocator)
  : Solver(compute, allocator, SOLVER_FLUID)
{
  type = SOLVER_FLUID;
  iterations = 1;
  create(compute);
}

void FluidSolver::create(ComputeInterface* compute)
{
}

void FluidSolver::solve()
{
  uint count = lastPartition().end();

  if (!count) return;
}

void FluidSolver::update()
{
  if (!updates.size()) return;

  Solver::update();
}
