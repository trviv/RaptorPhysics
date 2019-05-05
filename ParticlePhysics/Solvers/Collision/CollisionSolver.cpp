#include "CollisionSolver.h"

CollisionSolver::~CollisionSolver()
{
  delete solverHeap;
}

DeviceArray<XAB>* CollisionSolver::getBoundingBoxes()
{
  return NULL;
}