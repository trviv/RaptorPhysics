#ifndef LBVH_SOLVER_H
#define LBVH_SOLVER_H

#include "CollisionSolver.h"

/*!
@class Class to solve collisions using Binary Radix Tree
@info Fast BVH Construction on GPUs C. Lauterbach and M. Garland and S. Sengupta and D. Luebke and D. Manocha.
*/
class LBVHSolver : public CollisionSolver
{
protected:

  DeviceArray <ParticleStruct>  particlesTemp2;
  DeviceArray <BVHNodeInfo>     treeInternalNodes;
  DeviceArray <BVHLeafInfo>     particleLeafData;
  DeviceArray <BVHLeafInfo>     particleLeafDataSorted;
  DeviceArray <uint>            visitedInternalNodes;
  DeviceArray <uint>            leafParentNodeIndices;
  DeviceArray <uint>            nodeParentNodeIndices;
  DeviceArray <XAB>             particleBoundingBoxes;
  DeviceArray <XAB>             treeInternalNodeBoundingBoxes;

  void build(uint instanceNodeCount, ComputeMemory* globalOffsets);

public:

  /*!
  @constructor Construct a LBVH solver object.
  @param compute Compute interface to be used for the solver.
  @param allocator Shared memory allocator for the solver.
  */
  LBVHSolver(ComputeInterface* compute, SharedAllocator* allocator);

  /*!@destructor Destroy a LBVH solver object.*/
  ~LBVHSolver();

  /*!@function Initialize a LBVH solver object.*/
  void init();

  void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);

  DeviceArray<XAB>* getBoundingBoxes();
};

#endif
