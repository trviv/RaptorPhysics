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
  ComputeHeap*              solverHeap;
  DeviceArray <BVHLeafInfo> particleData;
  DeviceArray <uint>        gridCompactCellCount;
  DeviceArray <uint>        gridCompactCellIndices;
  DeviceArray <uint>        gridParticleCellIndex;
  DeviceArray <uint>        gridCellParticleCount;
  DeviceArray <uint>        gridCellParticleOffsets;
  DeviceArray <uint>        gridCellParticleIndices;
  uint                      gridSize;

public:

  ~CollisionSolver();

  virtual void init(ComputeInterface* compute, SharedAllocator* allocator);

  virtual void build(uint instanceNodeCount, ComputeMemory* globalOffsets);

  virtual void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);
};

#endif