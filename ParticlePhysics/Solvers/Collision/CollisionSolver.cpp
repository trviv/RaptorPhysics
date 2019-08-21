#include "CollisionSolver.h"

CollisionSolver::~CollisionSolver()
{
  empty.free();
  particlesTemp.free();
  particleGroupBoundingBoxes.free();
  systemBoundingBox.free();

  delete solverHeap;
  solverHeap = NULL;
}

void CollisionSolver::init(ComputeInterface* compute, SharedAllocator* allocator)
{
  this->compute = compute;
  this->allocator = allocator;

  includeFiles.push_back("ComputeHeader.shader");
  includeFiles.push_back("ComputeShared.h");
  includeFiles.push_back("ParticleStruct.h");
  includeFiles.push_back("CollisionSolverShared.h");
}

DeviceArray<XAB>* CollisionSolver::getBoundingBoxes()
{
  return NULL;
}
