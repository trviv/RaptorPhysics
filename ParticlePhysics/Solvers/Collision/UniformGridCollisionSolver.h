#ifndef UNIFORM_GRID_COLLISION_SOLVER_H
#define UNIFORM_GRID_COLLISION_SOLVER_H

#include "CollisionSolver.h"

/*!
@class Class to solve collisions using a uniform grid.
*/
class UniformGridCollisionSolver : public CollisionSolver
{
protected:

  static uint gridXABComputeUtilId;
  static uint gridComputeUtilId;

  DeviceArray <uint>  gridParticleCellIndex;
  DeviceArray <uint> &gridCellParticleOffsets;
  DeviceArray <uint>  gridCellParticleIndices;

  void build(uint instanceNodeCount, ComputeMemory* systemSettings, ComputeMemory* particleBuffer);

  void createUtilities();

public:

  uint                gridSize;
  uint                gridSizeExp;
  DeviceArray <uint>  gridCellParticleCount;

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

  void solve(uint instanceNodeCount, ComputeMemory* systemSettings);
};

#endif
