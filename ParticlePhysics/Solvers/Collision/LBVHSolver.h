#ifndef LBVH_SOLVER_H
#define LBVH_SOLVER_H

#include "CollisionSolver.h"
#include "CollisionSolverShared.h"

/*!
@class Class to solve collisions using Binary Radix Tree
@info Fast BVH Construction on GPUs C. Lauterbach and M. Garland and S. Sengupta and D. Luebke and D. Manocha.
*/
class LBVHSolver : public CollisionSolver
{
  DeviceArray <ParticleStruct>  particlesTemp2;
  DeviceArray <BVHNodeInfo>     treeInternalNodes;
  DeviceArray <BVHLeafInfo>     particleLeafData;
  DeviceArray <BVHLeafInfo>     particleLeafDataSorted;
  DeviceArray <uint>            visitedInternalNodes;
  DeviceArray <uint>            leafParentNodeIndices;
  DeviceArray <uint>            nodeParentNodeIndices;
  DeviceArray <XAB>             particleBoundingBoxes;
  DeviceArray <XAB>             treeInternalNodeBoundingBoxes;

public:

  LBVHSolver();

  ~LBVHSolver();

  void init(ComputeInterface* compute, SharedAllocator* allocator);

  void build(uint instanceNodeCount, ComputeMemory* globalOffsets);

  void solve(uint instanceNodeCount, ComputeMemory* globalOffsets);

  DeviceArray<XAB>* getBoundingBoxes();
};

#endif
