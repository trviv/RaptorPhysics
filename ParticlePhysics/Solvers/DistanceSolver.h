/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef DISTANCE_SOLVER_H
#define DISTANCE_SOLVER_H

#include "LinearSolver.h"
#include "../Common/ParticleStruct.h"

/*!
@class Class to solve distance constraints.
*/
class DistanceSolver : public LinearSolver<uint, real, Real3>
{
  typedef uint  IndexType;
  typedef real  CoefficientType;
  typedef Real3 VariableType;

  void update();

public:

  DistanceSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void addDistance(IndexType index, IndexType connectionIndex, CoefficientType distance);

  void create(ComputeInterface* compute);

  void solve(float timeStep);
};

#endif
