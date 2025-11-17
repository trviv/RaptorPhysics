/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "CollisionSolver.h"

CollisionSolver::CollisionSolver(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;
}

CollisionSolver::~CollisionSolver()
{
  particlesBufferTemp.free();
  particleGroupBoundingBoxes.free();
  systemBoundingBox.free();
  invMaxRadius.free();

  delete solverHeap;
  solverHeap = NULL;
}
