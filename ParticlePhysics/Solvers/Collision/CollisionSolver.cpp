#include "CollisionSolver.h"

CollisionSolver::CollisionSolver(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");
  includeFiles.push_back("CollisionSolver.shader");
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
