#ifndef COLLISION_SOLVER_H
#define COLLISION_SOLVER_H

#include "../Common/ParticleStruct.h"
#include "SharedAllocator.h"

/*!
@class Class to solve collision.
*/
class CollisionSolver : public ShaderEntity
{

  ComputeInterface* compute;
  SharedAllocator*  allocator;

public:

  ~CollisionSolver();

  void init(ComputeInterface* compute, SharedAllocator* allocator);

  void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);
};

#endif