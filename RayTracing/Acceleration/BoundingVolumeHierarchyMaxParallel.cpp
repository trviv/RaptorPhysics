/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#include "BoundingVolumeHierarchyADS.h"

//#define DEBUG_BVH_ADS

void BoundingVolumeHierarchyADS::initializeDataMaxParallelTree()
{
  MaximizingParallelismTreeData* treePtr = (MaximizingParallelismTreeData*)treeDataPtr;
  treePtr->visitedInternalNodes.create(compute);
}

void BoundingVolumeHierarchyADS::constructMaxParallelTree()
{
  size_t workgroupSize[3], workgroupCount[3];
  compute->configureSize(workgroupSize, workgroupCount, primitiveCount);

  MaximizingParallelismTreeData* treePtr = (MaximizingParallelismTreeData*)treeDataPtr;

  // create binary radix tree
  treePtr->constructBinaryTree.setArg(pointerTreeInternalNodes, 0);
  treePtr->constructBinaryTree.setArg(treePtr->visitedInternalNodes.device(), 1);
  treePtr->constructBinaryTree.setArg(pointerLeafParentNodeIndices, 2);
  treePtr->constructBinaryTree.setArg(pointerNodeParentNodeIndices, 3);
  treePtr->constructBinaryTree.setArg(primitiveLeafDataSorted.device(), 4);
  treePtr->constructBinaryTree.setArg(&primitiveCount, 5);

  compute->execute(treePtr->constructBinaryTree, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
  treeInternalNodes.syncHost();
  leafParentNodeIndices.syncHost();
  nodeParentNodeIndices.syncHost();
  compute->sync();
#endif

  // calculate bounding boxes for the tree
  treePtr->constructTreeBoundingBox.setArg(pointerTreeNodeBoundingBoxes, 0);
  treePtr->constructTreeBoundingBox.setArg(treePtr->visitedInternalNodes.device(), 1);
  treePtr->constructTreeBoundingBox.setArg(pointerTreeInternalNodes, 2);
  treePtr->constructTreeBoundingBox.setArg(pointerLeafParentNodeIndices, 3);
  treePtr->constructTreeBoundingBox.setArg(pointerNodeParentNodeIndices, 4);
  treePtr->constructTreeBoundingBox.setArg(pointerLeafNodeBoundingBoxes, 5);
  treePtr->constructTreeBoundingBox.setArg(&primitiveCount, 6);

  compute->execute(treePtr->constructTreeBoundingBox, workgroupSize, workgroupCount);

#ifdef DEBUG_BVH_ADS
  treeNodeBoundingBoxes.syncHost();
  treePtr->visitedInternalNodes.syncHost();
  compute->sync();
#endif
}

void BoundingVolumeHierarchyADS::resizeBuffersMaxParallelTree()
{
  MaximizingParallelismTreeData* treePtr = (MaximizingParallelismTreeData*)treeDataPtr;
  treePtr->visitedInternalNodes.resize(primitiveCount, false);
}
