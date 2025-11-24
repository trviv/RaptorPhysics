/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef RIGID_SOLVER_H
#define RIGID_SOLVER_H

#include "Solver.h"
#include "../Common/ParticleStruct.h"

/*!
@class Class to solve rigid body constraints.
*/
class RigidSolver : public EntitySolverType
{
  DeviceArray<real> covarianceMatrix;

  void update();

public:

  RigidSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve(float timeStep);
};

#endif
