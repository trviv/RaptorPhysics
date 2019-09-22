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

  void build(uint instanceNodeCount, ComputeMemory* globalOffsets);

public:

  /*!
  @constructor Construct a Grid solver object.
  @param compute Compute interface to be used for the solver.
  @param allocator Shared memory allocator for the solver.
  */
  UniformGridCollisionSolver(ComputeInterface* compute, SharedAllocator* allocator);

  /*!@destructor Destroy a Grid solver object.*/
  ~UniformGridCollisionSolver();

  /*!@function Initialize a Grid solver object.*/
  void init();

  void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);

  DeviceArray<XAB>* getBoundingBoxes();
};

#endif
