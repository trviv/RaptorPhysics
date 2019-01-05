#ifndef COLLISION_SOLVER_H
#define COLLISION_SOLVER_H

#include "../../Common/ParticleStruct.h"
#include "../SharedAllocator.h"
#include "CollisionSolverShared.h"

/*!
@class Class to solve collisions.
*/
class CollisionSolver : public ShaderEntity
{
protected:

  ComputeInterface*         compute;
  SharedAllocator*          allocator;
  DeviceArray <BVHLeafInfo> particleData;

public:

  ~CollisionSolver();

  virtual void init(ComputeInterface* compute, SharedAllocator* allocator);

  virtual void build(uint instanceNodeCount);

  virtual void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);
};

#endif