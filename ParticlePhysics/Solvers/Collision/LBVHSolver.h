/*
 * Copyright (c) Vivek Trivedi, 2025
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

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

  DeviceArray <BVHNodeInfo>     treeInternalNodes;
  DeviceArray <BVHLeafInfo>     particleLeafData;
  DeviceArray <BVHLeafInfo>     particleLeafDataSorted;
  DeviceArray <uint>            visitedInternalNodes;
  DeviceArray <uint>            leafParentNodeIndices;
  DeviceArray <uint>            nodeParentNodeIndices;
  DeviceArray <XAB>             particleBoundingBoxes;
  DeviceArray <XAB>             treeInternalNodeBoundingBoxes;

  void build(uint instanceNodeCount, ComputeMemory* systemSettings, ComputeMemory* particleBuffer);

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

  void solve(uint instanceNodeCount, ComputeMemory* systemSettings);
};

#endif
