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

  ComputeInterface* compute;
  SharedAllocator*  allocator;
  ComputeHeap*      solverHeap;

public:

  virtual ~CollisionSolver();

  virtual void init(ComputeInterface* compute, SharedAllocator* allocator) = 0;

  virtual void build(uint instanceNodeCount, ComputeMemory* globalOffsets) = 0;

  virtual void solve(uint instanceNodeCount, ComputeMemory* globalOffsets) = 0;

  virtual DeviceArray<XAB>* getBoundingBoxes();
};

#endif
