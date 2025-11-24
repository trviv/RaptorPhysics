/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "Solver.h"

/*!
@class Class to solve linear constraints.
*/
template<class IndexType, class CoefficientType, class VariableType>
class LinearSolver : public EntitySolver<IndexType, CoefficientType, VariableType>
{
protected:

  void update();

public:

  LinearSolver(ComputeInterface* compute, SharedAllocator* allocator);

  void create(ComputeInterface* compute);

  void solve();
};

#endif
