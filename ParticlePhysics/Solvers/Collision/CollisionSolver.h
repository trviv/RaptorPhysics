#ifndef COLLISION_SOLVER_H
#define COLLISION_SOLVER_H

#include "../Solver.h"
#include "CollisionSolverShared.h"

/*!
@class Class to solve collisions.
*/
class CollisionSolver : virtual public Solver
{
protected:

  ComputeHeap*                  solverHeap;
  DeviceArray <ParticleStruct>  empty;
  DeviceArray <ParticleStruct>  particlesTemp;
  DeviceArray <XAB>             systemBoundingBox;
  DeviceArray <XAB>             particleGroupBoundingBoxes;
  DeviceArray <float>           maxRadius;

public:

  virtual ~CollisionSolver();

  virtual void init(ComputeInterface* compute, SharedAllocator* allocator);

  virtual void build(uint instanceNodeCount, ComputeMemory* globalOffsets) = 0;

  virtual void solve(uint instanceNodeCount, ComputeMemory* globalOffsets) = 0;

  virtual DeviceArray<XAB>* getBoundingBoxes();
};

#endif
