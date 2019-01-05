#ifndef LBVH_SOLVER_H
#define LBVH_SOLVER_H

#include "CollisionSolver.h"
#include "CollisionSolverShared.h"

/*!
@class Class to solve collision using LBVH paper
@info Fast BVH Construction on GPUs C. Lauterbach and M. Garland and S. Sengupta and D. Luebke and D. Manocha.
*/
class LBVHSolver : public CollisionSolver
{
public:

  ~LBVHSolver();

  void init(ComputeInterface* compute, SharedAllocator* allocator);

  void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);
};

#endif