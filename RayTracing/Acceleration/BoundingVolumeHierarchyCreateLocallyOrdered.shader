/*
 * RaptorPhysics
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 */

#ifndef BOUNDING_VOLUME_HIERARCHY_CREATE_LOCALLY_ORDERED_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_CREATE_LOCALLY_ORDERED_SHADER_H

#include "BoundingVolumeHierarchyADSCreate.shader"

/*
@kernel Create initial clusters using leaf data.
@param clusterNodes Cluster nodes data.
@param clusterBoundingBoxes Cluster node bounding boxes.
@param nodeCount Total number of leaves in the tree.
*/
Kernel void createLeafClustersLocallyOrdered(
  Device BVHClusterStruct*  clusterNodes,
  const Device XAB*         clusterBoundingBoxes,
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const int index = threadIndex();

  if (index >= nodeCount) return;

  const XAB selfBoundingBox = clusterBoundingBoxes[index];

  BVHClusterStruct newCluster;
  newCluster.center       = (selfBoundingBox.min + selfBoundingBox.max) * 0.5f;
  newCluster.clusterIndex = index;

  clusterNodes[index] = newCluster;
}

/*
@kernel Find the nearest neighbour for each cluster.
@param treeInternalNodes Binary radix tree internal connectivity data.
@param visitedInternalNodes Counter for tracking threads visited during bottom up traversal.
@param leafParentNodeIndices Immediate parents to particle leaf data.
@param bvhLeafs Particle position and index data for bounding volume hierarchy.
@param nodeCount Total nodes in the solver.
*/
Kernel void findNearestClusterLocallyOrdered(
  Device int*                     nearestNeighbour,
  const Device BVHClusterStruct*  clusterNodes,
  constantKernelInput(int,        nodeCount),
  constantKernelInput(int,        searchRadius)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const int index = threadIndex();

  if (index >= nodeCount) return;

  const float3 selfCenter = clusterNodes[index].center;
  float minDistance = INFINITY;

  int closestNeighbourIndex = -1;

  for (int i = max(0, index - searchRadius); i <= min(nodeCount - 1, index + searchRadius); i++)
  {
    if (i == index) continue;

    const float distance = lengthSq(clusterNodes[i].center - selfCenter);

    if (minDistance > distance)
    {
      closestNeighbourIndex = i;
      minDistance = distance;
    }
  }

  nearestNeighbour[index] = closestNeighbourIndex;
}

/*
@kernel Create boundig box of tree nodes.
@param treeInternalNodeBoundingBoxes Bounding box for tree nodes.
@param visitedInternalNodes Counter for tracking threads visited during bottom up traversal.
@param treeInternalNodes Binary radix tree internal connectivity data.
@param leafParentNodeIndices Immediate parents to particle leaf data.
@param primitiveBoundingBoxes Particle bounding box array.
@param nodeCount Total nodes in the solver.
*/
Kernel void mergeClustersLocallyOrdered(
  Device int*               clusterValidFlags,
  Device int*               clusterNewFlags,
  Device BVHClusterStruct*  oldClusterNodes,
  Device XAB*               clusterBoundingBoxes,
  Device BVHNodeInfo*       treeInternalNodes,
  const Device int*         nearestNeighbour,
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const int index = threadIndex();

  if (index >= nodeCount) return;

  const int neighbourIndex = nearestNeighbour[index];

  int clusterValidFlag  = 1;
  int clusterNewFlag    = 0;

  if (nearestNeighbour[neighbourIndex] == index)
  {
    if (index < neighbourIndex)
    {
      BVHClusterStruct selfClusterNode  = oldClusterNodes[index];
      BVHClusterStruct otherClusterNode = oldClusterNodes[neighbourIndex];

      BVHNodeInfo internalNode;
      internalNode.childLeft   = selfClusterNode.clusterIndex;
      internalNode.childRight  = otherClusterNode.clusterIndex;
      treeInternalNodes[index] = internalNode;

      XAB selfBoundingBox         = clusterBoundingBoxes[index];
      const XAB otherBoundingBox  = clusterBoundingBoxes[neighbourIndex];
      mergeXAB(&selfBoundingBox, &otherBoundingBox);

      clusterBoundingBoxes[index] = selfBoundingBox;

      BVHClusterStruct newCluster;
      newCluster.center = (selfBoundingBox.min + selfBoundingBox.max) * 0.5f;
      newCluster.clusterIndex = -2;

      oldClusterNodes[index]  = newCluster;
      oldClusterNodes[neighbourIndex].clusterIndex = -1;
      clusterNewFlag = 1;
    }
    else
    {
      clusterValidFlag = 0;
    }
  }

  clusterNewFlags[index]   = clusterNewFlag;
  clusterValidFlags[index] = clusterValidFlag;
}

Kernel void compactClustersLocallyOrdered(
  Device XAB*                     newClusterBoundingBoxes,
  Device BVHClusterStruct*        newClusterNodes,
  Device BVHNodeInfo*             treeInternalNodes,
  Device int*                     newInternalNodeCount,
  Device int*                     newClusterCount,
  Device uint*                    leafParentNodeIndices,
  Device uint*                    nodeParentNodeIndices,
  Device XAB*                     treeInternalNodeBoundingBoxes,
  const Device XAB*               oldClusterBoundingBoxes,
  const Device BVHClusterStruct*  oldClusterNodes,
  const Device BVHNodeInfo*       oldTreeInternalNodes,
  constantKernelInput(int,        oldInternalNodeCount),
  constantKernelInput(int,        nodeCount),
  const Device int*               clusterValidFlags,
  const Device int*               clusterNewFlags,
  constantKernelInput(int,        primitiveCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const int index = threadIndex();

  if (index >= nodeCount) return;

  BVHClusterStruct clusterNode = oldClusterNodes[index];

  const int totalNewClusters = clusterNewFlags[nodeCount-1] + (oldClusterNodes[nodeCount-1].clusterIndex == -2);

  if (clusterNode.clusterIndex != -1)
  {
    const XAB clusterBoundingBox = oldClusterBoundingBoxes[index];

    if (clusterNode.clusterIndex == -2)
    {
      //const int internalNodeIndex = oldInternalNodeCount + clusterNewFlags[index];
      const int internalNodeIndex = primitiveCount - (1 + totalNewClusters + oldInternalNodeCount) + clusterNewFlags[index];
      const BVHNodeInfo internalNode = oldTreeInternalNodes[index];

      treeInternalNodes[internalNodeIndex] = internalNode;
      treeInternalNodeBoundingBoxes[internalNodeIndex] = clusterBoundingBox;

      const int parentNodeIndex   = setBVHInternalNodeMarker(false, internalNodeIndex);

      if (isBVHLeafNode(internalNode.childLeft))
      {
        leafParentNodeIndices[internalNode.childLeft] = parentNodeIndex;
      }
      else
      {
        nodeParentNodeIndices[removeBVHInternalNodeMarker(internalNode.childLeft)] = parentNodeIndex;
      }

      if (isBVHLeafNode(internalNode.childRight))
      {
        leafParentNodeIndices[internalNode.childRight] = parentNodeIndex;
      }
      else
      {
        nodeParentNodeIndices[removeBVHInternalNodeMarker(internalNode.childRight)] = parentNodeIndex;
      }

      clusterNode.clusterIndex = parentNodeIndex;
    }

    const int newIndex = clusterValidFlags[index];

    newClusterNodes[newIndex] = clusterNode;
    newClusterBoundingBoxes[newIndex] = clusterBoundingBox;
  }

  if (index == (nodeCount-1))
  {
    const int clusterCount = clusterValidFlags[index] + (clusterNode.clusterIndex != -1);
    newClusterCount[0] = clusterCount;
    newInternalNodeCount[0] = oldInternalNodeCount + (nodeCount - clusterCount);
  }
}

#endif
