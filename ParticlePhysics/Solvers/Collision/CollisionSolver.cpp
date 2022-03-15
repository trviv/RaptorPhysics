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
