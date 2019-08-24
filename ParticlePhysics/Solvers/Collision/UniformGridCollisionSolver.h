#ifndef UNIFORM_GRID_COLLISION_SOLVER_H
#define UNIFORM_GRID_COLLISION_SOLVER_H

#include "CollisionSolver.h"

/*!
@class Class to solve collisions using a uniform grid.
*/
class UniformGridCollisionSolver : public CollisionSolver
{
protected:

  DeviceArray <uint>  gridCompactCellCount;
  DeviceArray <uint>  gridCompactCellIndices;
  DeviceArray <uint>  gridParticleCellIndex;
  DeviceArray <uint>  gridCellParticleCount;
  DeviceArray <uint>  gridCellParticleOffsets;
  DeviceArray <uint>  gridCellParticleIndices;
  uint                gridSize;

public:

  ~UniformGridCollisionSolver();

  void init(ComputeInterface* compute, SharedAllocator* allocator);

  void build(uint instanceNodeCount, ComputeMemory* globalOffsets);

  void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);

  DeviceArray<XAB>* getBoundingBoxes();
};

#endif
