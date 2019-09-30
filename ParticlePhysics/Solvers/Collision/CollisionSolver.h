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
  DeviceArray <ParticleStruct>  empty;
  DeviceArray <ParticleStruct>  particlesCurrentTemp;
  DeviceArray <ParticleStruct>  particlesPredictedTemp;
  DeviceArray <XAB>             systemBoundingBox;
  DeviceArray <XAB>             particleGroupBoundingBoxes;
  DeviceArray <float>           maxRadius;

  virtual void build(uint instanceNodeCount, ComputeMemory* systemSettings) = 0;

public:

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

  virtual DeviceArray<XAB>* getBoundingBoxes();
};

#endif
