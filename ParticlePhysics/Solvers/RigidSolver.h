#ifndef RIGID_SOLVER_H
#define RIGID_SOLVER_H

#include "Solver.h"
#include "../Common/ParticleStruct.h"

class RigidSolver : public Solver<uint, real, Real3>
{
  DeviceArray<real> covarianceMatrix;

  void update();

public:

  RigidSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve();
};

#endif
