/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef COLLISION_SOLVER_H
#define COLLISION_SOLVER_H

#include "../Solver.h"

/*!
@class Class to solve collisions.
*/
class CollisionSolver : virtual public Solver
{
protected:

  ComputeHeap*                  solverHeap;
  DeviceArray <ParticleStruct>  particlesBufferTemp;

  virtual void build(uint instanceNodeCount, ComputeMemory* systemSettings, ComputeMemory* particleBuffer) = 0;

public:

  DeviceArray <float> invMaxRadius;
  DeviceArray <XAB>   systemBoundingBox;
  DeviceArray <XAB>   particleGroupBoundingBoxes;

  /*!
  @constructor Construct a collision solver object.
  @param compute Compute interface to be used for the solver.
  @param allocator Shared memory allocator for the solver.
  */
  CollisionSolver(ComputeInterface* compute, SharedAllocator* allocator);

  virtual ~CollisionSolver();

  /*!@function Initialize a collision solver object.*/
  virtual void init() = 0;

  virtual void solve(uint instanceNodeCount, ComputeMemory* systemSettings) = 0;
};

#endif
