/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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

  /* Kernels common to uniform grid solvers */
  ComputeKernel createBoundingBoxes;
  ComputeKernel createGridCellHistogram;
  ComputeKernel createGridCellArrays;
  ComputeKernel createBoundaryGridCellHistogram;
  ComputeKernel createBoundaryGridCellArrays;

  DeviceArray <uint>  gridParticleCellIndex;
  DeviceArray <uint> &gridCellParticleOffsets;
  DeviceArray <uint>  gridCellParticleIndices;

  void build(uint instanceNodeCount, ComputeMemory* systemSettings, ComputeMemory* particleBuffer);

  void createUtilities();

public:

  ushort              gridSize;
  ushort              gridSizeExp;
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
