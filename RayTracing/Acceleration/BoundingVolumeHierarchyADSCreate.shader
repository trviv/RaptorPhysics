#ifndef BOUNDING_VOLUME_HIERARCHY_ADS_CREATE_SHADER_H
#define BOUNDING_VOLUME_HIERARCHY_ADS_CREATE_SHADER_H

//#define BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION

#define BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER  ((uint)-1)

// The most significant bit(0x80000000) of a uint32 is used to distinguish between leaf and internal nodes.
// If it is set, then the index is for an internal node; otherwise, it is a leaf node.
// In both cases, the bit should be cleared to access the actual node index.
inline bool isBVHLeafNode(const uint index)
{
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION
  return index < 1000000000;
#else
  return (index & 0x80000000) == 0;
#endif
}

inline uint setBVHInternalNodeMarker(const bool isLeaf, uint index)
{
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION
  return select(index + 1000000000, index, isLeaf);
#else
  return select(index | 0x80000000, index, isLeaf);
#endif
}

inline uint removeBVHInternalNodeMarker(uint index)
{
#ifdef BOUNDING_VOLUME_HIERARCHY_ADS_DEBUG_TREE_CREATION
  return select(index - 1000000000, index, index < 1000000000);
#else
  return index & 0x7FFFFFFF;
#endif
}

inline short getBVHCommonPrefixLength(const uint2 left, const uint2 right)
{
  short ret = clz(left.x ^ right.x);
  return select(ret, (short)(clz(left.y ^ right.y) + 32), (short)(ret == 32));
}

inline short getBVHCommonPrefixLengthHalf(const uint2 left, const uint2 right)
{
  return clz(left.x ^ right.x);
}


/*
@kernel Create tree from the leaf data.
@param treeInternalNodes Binary radix tree internal connectivity data.
@param visitedInternalNodes Counter for tracking threads visited during bottom up traversal.
@param leafParentNodeIndices Immediate parents to particle leaf data.
@param bvhLeafs Particle position and index data for bounding volume hierarchy.
@param nodeCount Total nodes in the solver.
*/
Kernel void constructBinaryTree(
  Device BVHNodeInfo*       treeInternalNodes,
  Device uint*              visitedInternalNodes,
  Device uint*              leafParentNodeIndices,
  Device uint*              nodeParentNodeIndices,
  const Device BVHLeafInfo* bvhLeafs,
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const int internalNodeIndex = threadIndex();

  if (internalNodeIndex >= (nodeCount - 1))
    return;

  // code of leaf before this
  const uint2 prevNodePrefix = constructUint2(select(0xFFFFFFFF, bvhLeafs[internalNodeIndex - 1].mortonCode, internalNodeIndex > 0), internalNodeIndex - 1);

  // code of this leaf
  const uint2 currNodePrefix = constructUint2(bvhLeafs[internalNodeIndex].mortonCode, internalNodeIndex);

  // code of leaf after this
  const uint2 nextNodePrefix = constructUint2(bvhLeafs[internalNodeIndex + 1].mortonCode, internalNodeIndex + 1);

  const short nodeLeftSharedPrefixLength  = select((short)-1, (short)getBVHCommonPrefixLengthHalf(currNodePrefix, prevNodePrefix), (short)(internalNodeIndex > 0));
  const short nodeRightSharedPrefixLength = getBVHCommonPrefixLengthHalf(currNodePrefix, nextNodePrefix);

  const int direction   = select(-1, 1, nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength);
  const short minLength = select(nodeRightSharedPrefixLength, nodeLeftSharedPrefixLength, (short)(nodeRightSharedPrefixLength > nodeLeftSharedPrefixLength));

  int maxStep = 8;
  int currInternalNodeIndex;

  // Compute upper bound for the length of the range
  do
  {
    maxStep <<= 1;
    currInternalNodeIndex = internalNodeIndex + maxStep * direction;
  }
  while (currInternalNodeIndex >= 0 && currInternalNodeIndex < nodeCount && getBVHCommonPrefixLengthHalf(currNodePrefix, constructUint2(bvhLeafs[currInternalNodeIndex].mortonCode, currInternalNodeIndex)) > minLength);

  int indexOffset = 0;

  // Find the exact offset using binary search
  for (int currentStep = maxStep >> 1; currentStep != 0; currentStep >>= 1)
  {
    currInternalNodeIndex = internalNodeIndex + (currentStep + indexOffset) * direction;
    if (currInternalNodeIndex >= 0 && currInternalNodeIndex < nodeCount && getBVHCommonPrefixLengthHalf(currNodePrefix, constructUint2(bvhLeafs[currInternalNodeIndex].mortonCode, currInternalNodeIndex)) > minLength)
    {
      indexOffset += currentStep;
    }
  }

  const int otherInternalNodeIndex = internalNodeIndex + indexOffset * direction;
  const short splitLength = getBVHCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[otherInternalNodeIndex].mortonCode, otherInternalNodeIndex));

  int splitOffset = 0;

  // Find the split position using binary search
  do
  {
    indexOffset = (indexOffset >> 1) + (indexOffset & 1);
    currInternalNodeIndex = internalNodeIndex + (indexOffset + splitOffset) * direction;
    if (currInternalNodeIndex >= 0 && currInternalNodeIndex < nodeCount && getBVHCommonPrefixLength(currNodePrefix, constructUint2(bvhLeafs[currInternalNodeIndex].mortonCode, currInternalNodeIndex)) > splitLength)
    {
      splitOffset += indexOffset;
    }
  }
  while (indexOffset > 1);

  splitOffset = internalNodeIndex + splitOffset * direction + min(direction, 0);

  const bool leftIsLeaf  = (min(internalNodeIndex, otherInternalNodeIndex) == splitOffset);
  const bool rightIsLeaf = (max(internalNodeIndex, otherInternalNodeIndex) == (splitOffset + 1));

  BVHNodeInfo bvhNode;
  bvhNode.childLeft  = setBVHInternalNodeMarker(leftIsLeaf, splitOffset);
  bvhNode.childRight = setBVHInternalNodeMarker(rightIsLeaf, splitOffset + 1);

  treeInternalNodes[internalNodeIndex] = bvhNode;

  // mark internal node at zero'th index as the root
  if (internalNodeIndex == 0)
  {
    nodeParentNodeIndices[internalNodeIndex] = BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER;
  }

  if (!leftIsLeaf)  nodeParentNodeIndices[splitOffset]     = setBVHInternalNodeMarker(false, internalNodeIndex);
  if (!rightIsLeaf) nodeParentNodeIndices[splitOffset + 1] = setBVHInternalNodeMarker(false, internalNodeIndex);

  if (leftIsLeaf)   leafParentNodeIndices[splitOffset]     = setBVHInternalNodeMarker(false, internalNodeIndex);
  if (rightIsLeaf)  leafParentNodeIndices[splitOffset + 1] = setBVHInternalNodeMarker(false, internalNodeIndex);

  // reset visited counter for use in next dispatch
  visitedInternalNodes[internalNodeIndex] = 0;
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
Kernel void constructTreeBoundingBox(
  Device XAB*               treeInternalNodeBoundingBoxes,
  atomicKernelInput(uint,   visitedInternalNodes),
  const Device BVHNodeInfo* treeInternalNodes,
  const Device uint*        leafParentNodeIndices,
  const Device uint*        nodeParentNodeIndices,
  const Device XAB*         primitiveBoundingBoxes,
  constantKernelInput(int,  nodeCount)
  KERNEL_GLOBAL_ARGUMENTS)
{
  const uint index = threadIndex();

  if (index >= nodeCount)
    return;

  // begin with leaf node bounding boxes
  uint currentNodeIndex = leafParentNodeIndices[index];

  // exit if root
  while (currentNodeIndex != BOUNDING_VOLUME_HIERARCHY_ADS_ROOT_NODE_MARKER)
  {
    currentNodeIndex = removeBVHInternalNodeMarker(currentNodeIndex);
    // get the visited order
    const uint visited = atomicAdd(&visitedInternalNodes[currentNodeIndex], 1);

    // only process if visited after
    if (visited == 0)
    {
      break;
    }

    // fetch node data
    const BVHNodeInfo internalNode = treeInternalNodes[currentNodeIndex];

    // bounding box accumulated by the thread
    XAB mergedBoundingBox;
    clearXAB(&mergedBoundingBox, INFINITY);

    // fetched bounding box
    XAB childBoundingBox;
    if (isBVHLeafNode(internalNode.childLeft))
    {
      childBoundingBox = primitiveBoundingBoxes[internalNode.childLeft];
    }
    else
    {
      childBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(internalNode.childLeft)];
    }
    mergeXAB(&mergedBoundingBox, &childBoundingBox);

    if (isBVHLeafNode(internalNode.childRight))
    {
      childBoundingBox = primitiveBoundingBoxes[internalNode.childRight];
    }
    else
    {
      childBoundingBox = treeInternalNodeBoundingBoxes[removeBVHInternalNodeMarker(internalNode.childRight)];
    }
    mergeXAB(&mergedBoundingBox, &childBoundingBox);

    // save the internal node information
    treeInternalNodeBoundingBoxes[currentNodeIndex] = mergedBoundingBox;

    // process the parent node next
    currentNodeIndex = nodeParentNodeIndices[currentNodeIndex];
  }
}

#endif
